import os

import pytest
import torch
import torch.distributed as dist

from fla.cp.conv import causal_conv1d_cp
from fla.modules.convolution import causal_conv1d
from fla.ops.cp import get_cp_context, set_cp_context
from fla.utils import device


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        return False

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed():
    dist.destroy_process_group()


def assert_close(name, ref, tri, atol=1e-3, rtol=1e-3):
    """辅助断言函数"""
    diff = (ref - tri).abs().max().item()
    print(f"  Checking {name:<10} | Max Diff: {diff:.6f}")
    assert diff < atol, f"{name} mismatch. Max diff: {diff}"


def run_cp_test(dtype=torch.float32):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # -------------------------------------------------------
    # 1. 准备全局数据 (所有 Rank 使用相同种子生成相同数据)
    # -------------------------------------------------------
    torch.manual_seed(42)
    B, T, D, W = 1, 1024, 128, 4  # Varlen 模式 B 必须为 1

    # 全局数据
    x_global = torch.randn(B, T, D, device=device, dtype=dtype)
    dy_global = torch.randn(B, T, D, device=device, dtype=dtype)

    # weight 和 bias
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)

    # 广播权重以确保完全一致
    dist.broadcast(weight, src=0)
    dist.broadcast(bias, src=0)

    # 构造 Varlen 序列长度 (跨越 Rank 边界)
    lengths = [300, 400, T - 700]
    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

    activation = 'swish'

    # -------------------------------------------------------
    # 2. Reference Run (仅在 Rank 0 上计算，作为基准)
    # -------------------------------------------------------
    ref_out, ref_dx, ref_dw, ref_db = None, None, None, None

    if rank == 0:
        x_ref = x_global.clone().detach().requires_grad_(True)
        weight_ref = weight.clone().detach().requires_grad_(True)
        bias_ref = bias.clone().detach().requires_grad_(True)

        # 普通前向 (使用 triton backend)
        y_ref, _ = causal_conv1d(
            x=x_ref,
            weight=weight_ref,
            bias=bias_ref,
            activation=activation,
            backend='triton',
            cu_seqlens=cu_seqlens_global,
        )

        # 普通反向
        y_ref.backward(dy_global)

        ref_out = y_ref.detach()
        ref_dx = x_ref.grad.detach()
        ref_dw = weight_ref.grad.detach()
        ref_db = bias_ref.grad.detach()

    # -------------------------------------------------------
    # 3. Context Parallel Run (所有 Rank 并行)
    # -------------------------------------------------------
    dist.barrier()

    # A. 激活 CP Context (group=None 表示使用默认的 world group)
    # 注意：set_gdn_cp_context 中 group=None 会返回空 context，需要传入实际的 group
    set_cp_context(cu_seqlens_global, group=dist.group.WORLD, kernel_size=W)
    context = get_cp_context()

    # B. 数据切分 (Slice Input)
    chunk_size = T // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    dy_local = dy_global[:, start_idx:end_idx, :].clone()
    weight_local = weight.clone().detach().requires_grad_(True)
    bias_local = bias.clone().detach().requires_grad_(True)

    # === DEBUG: 打印 context 信息 ===
    print(f"[Rank {rank}] cu_seqlens_global: {cu_seqlens_global.tolist()}")
    print(f"[Rank {rank}] context.cu_seqlens: {context.cu_seqlens.tolist()}")
    print(f"[Rank {rank}] pre_num_ranks: {context.pre_num_ranks}")
    print(f"[Rank {rank}] pre_num_conv_tokens: {context.pre_num_conv_tokens}")
    print(f"[Rank {rank}] start_idx: {start_idx}, end_idx: {end_idx}")
    print(f"[Rank {rank}] x_local shape: {x_local.shape}")
    dist.barrier()

    # C. CP 前向 (使用 causal_conv1d_cp)
    y_local = causal_conv1d_cp(
        x=x_local,
        weight=weight_local,
        bias=bias_local,
        activation=activation,
        cu_seqlens=context.cu_seqlens,
    )

    # === DEBUG: 比较 forward output ===
    dist.barrier()
    if rank == 0:
        # Rank 0 的 output 应该和 ref_out[:, :chunk_size, :] 一致
        ref_local = ref_out[:, :chunk_size, :]
        diff = (y_local - ref_local).abs().max().item()
        print(f"[Rank 0] Forward diff (local vs ref[:chunk_size]): {diff:.6f}")
    dist.barrier()

    # D. CP 反向
    y_local.backward(dy_local)

    # -------------------------------------------------------
    # 4. 结果聚合与验证
    # -------------------------------------------------------

    # 聚合 Output
    y_gathered = [torch.zeros_like(y_local) for _ in range(world_size)]
    dist.all_gather(y_gathered, y_local)
    y_cp_global = torch.cat(y_gathered, dim=1)

    # 聚合 dx (Input Gradient)
    dx_gathered = [torch.zeros_like(x_local.grad) for _ in range(world_size)]
    dist.all_gather(dx_gathered, x_local.grad)
    dx_cp_global = torch.cat(dx_gathered, dim=1)

    # 聚合 dw, db (Weight Gradient - AllReduce Sum)
    dw_cp = weight_local.grad.clone()
    db_cp = bias_local.grad.clone()
    dist.all_reduce(dw_cp, op=dist.ReduceOp.SUM)
    dist.all_reduce(db_cp, op=dist.ReduceOp.SUM)

    # 仅在 Rank 0 进行断言检查
    if rank == 0:
        print(f"\n[CP Conv1d] Verification Results (CP Size={world_size}):")

        # 1. 检查 Output
        assert_close("Output", ref_out, y_cp_global, atol=1e-3)

        # 2. 检查 dx
        assert_close("dx (Input)", ref_dx, dx_cp_global, atol=1e-3)

        # 3. 检查 dw
        assert_close("dw (Weight)", ref_dw, dw_cp, atol=1e-3)

        # 4. 检查 db
        assert_close("db (Bias)", ref_db, db_cp, atol=1e-3)

        print("✅ [CP Conv1d] Test Passed!\n")

    dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cp_convolution():
    """
    该测试需要通过 torchrun 启动才能运行:
    torchrun --nproc_per_node=2 test_cp_conv.py
    """
    if not dist.is_initialized():
        if not setup_distributed():
            pytest.skip("Distributed environment not available. Run with torchrun.")

    try:
        run_cp_test(dtype=torch.float32)
    finally:
        pass


if __name__ == "__main__":
    if setup_distributed():
        try:
            print("Running CP Conv1d Test...")
            run_cp_test(dtype=torch.float32)
        finally:
            cleanup_distributed()
    else:
        print("Please run with torchrun:")
        print("torchrun --nproc_per_node=2 test_cp_conv.py")
