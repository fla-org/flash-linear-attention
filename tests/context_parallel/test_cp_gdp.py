# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
from fla.utils import assert_close, device, device_torch_lib

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format="%(message)s")


def init_distributed(rank, world_size):
    """Initialize distributed environment for a single process."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29504"  # Different port from other CP tests
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device_torch_lib.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_gather_cat(x: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=1)


def run_cp_gdp_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    num_householder: int,
) -> None:
    """
    Worker function for CP GDP test.
    Runs in a spawned process with the given rank.
    """
    try:
        init_distributed(rank, world_size)
        current_device = torch.device(f"{device}:{rank}")
        dtype = torch.bfloat16
        B, T, H, K, V = 1, 256, 2, 64, 80
        lengths = [80, 110, 66]
        scale = K ** -0.5

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, H={H}, K={K}, V={V}, P={num_householder}, world_size={world_size}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data (all generated on rank 0, broadcast to all)
        q_global = torch.empty(B, T, H, K, dtype=dtype, device=current_device)
        k_global = torch.empty(B, T * num_householder, H, K, dtype=dtype, device=current_device)
        v_global = torch.empty(B, T * num_householder, H, V, dtype=dtype, device=current_device)
        beta_global = torch.empty(B, T * num_householder, H, dtype=dtype, device=current_device)
        g_global = torch.empty(B, T, H, dtype=torch.float32, device=current_device)
        do_global = torch.empty(B, T, H, V, dtype=dtype, device=current_device)

        if rank == 0:
            torch.manual_seed(42)
            q_global.copy_(
                F.normalize(
                    torch.randn(B, T, H, K, dtype=torch.float32, device=current_device), p=2, dim=-1
                ).to(dtype)
            )
            k_global.copy_(
                F.normalize(
                    torch.randn(B, T * num_householder, H, K, dtype=torch.float32, device=current_device),
                    p=2,
                    dim=-1,
                ).to(dtype)
            )
            v_global.copy_(torch.randn_like(v_global))
            beta_global.copy_(torch.randn_like(beta_global).sigmoid())
            g_global.copy_(F.logsigmoid(torch.randn_like(g_global)))
            do_global.copy_(torch.randn_like(do_global))

        for tensor in (q_global, k_global, v_global, beta_global, g_global, do_global):
            dist.broadcast(tensor, src=0)

        # Prepare cu_seqlens
        sequence_offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        cu_seqlens_global = torch.tensor(sequence_offsets, dtype=torch.long, device=current_device)

        # Step 2: Reference Run (recurrent, varlen, no CP)
        ref = None
        if rank == 0:
            q_ref = q_global.detach().clone().requires_grad_(True)
            k_ref = k_global.detach().clone().requires_grad_(True)
            v_ref = v_global.detach().clone().requires_grad_(True)
            beta_ref = beta_global.detach().clone().requires_grad_(True)
            g_ref = g_global.detach().clone().requires_grad_(True)
            outputs = []
            for bos, eos in zip(sequence_offsets[:-1], sequence_offsets[1:]):
                bos_dp, eos_dp = bos * num_householder, eos * num_householder
                o_ref_i, _ = naive_recurrent_gated_delta_product(
                    q=q_ref[:, bos:eos],
                    k=k_ref[:, bos_dp:eos_dp],
                    v=v_ref[:, bos_dp:eos_dp],
                    g=g_ref[:, bos:eos],
                    beta=beta_ref[:, bos_dp:eos_dp],
                    scale=scale,
                    cu_seqlens=None,
                    num_householder=num_householder,
                )
                outputs.append(o_ref_i)
            # The recurrent reference accepts scale but does not apply it.
            o_ref = torch.cat(outputs, dim=1) * scale
            o_ref.backward(do_global)
            ref = {
                "o": o_ref.detach(),
                "dq": q_ref.grad.detach(),
                "dk": k_ref.grad.detach(),
                "dv": v_ref.grad.detach(),
                "dbeta": beta_ref.grad.detach(),
                "dg": g_ref.grad.detach(),
            }

        # Step 3: Context Parallel Run
        dist.barrier()
        cp_context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)
        local_T = T // world_size
        start = rank * local_T
        end = start + local_T
        start_dp, end_dp = start * num_householder, end * num_householder

        # Get local slices
        q_local = q_global[:, start:end].detach().clone().requires_grad_(True)
        k_local = k_global[:, start_dp:end_dp].detach().clone().requires_grad_(True)
        v_local = v_global[:, start_dp:end_dp].detach().clone().requires_grad_(True)
        beta_local = beta_global[:, start_dp:end_dp].detach().clone().requires_grad_(True)
        g_local = g_global[:, start:end].detach().clone().requires_grad_(True)
        do_local = do_global[:, start:end].clone()

        print(
            f"[Rank {rank}] chunk: [{start}, {end}), "
            f"cu_seqlens: {cp_context.cu_seqlens.tolist()}, "
            f"pre_num_ranks: {cp_context.pre_num_ranks}"
        )
        dist.barrier()

        # CP Forward
        o_local, final_state = chunk_gated_delta_product(
            q=q_local,
            k=k_local,
            v=v_local,
            g=g_local,
            beta=beta_local,
            num_householder=num_householder,
            scale=scale,
            cp_context=cp_context,
        )
        assert final_state is None

        # CP Backward
        o_local.backward(do_local)

        # Step 4: Result Aggregation and Verification
        actual = {
            "o": all_gather_cat(o_local, world_size),
            "dq": all_gather_cat(q_local.grad, world_size),
            "dk": all_gather_cat(k_local.grad, world_size),
            "dv": all_gather_cat(v_local.grad, world_size),
            "dbeta": all_gather_cat(beta_local.grad, world_size),
            "dg": all_gather_cat(g_local.grad, world_size),
        }

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verifying results...")
            ratios = {"o": 5e-3, "dq": 8e-3, "dk": 8e-3, "dv": 8e-3, "dbeta": 2e-2, "dg": 2e-2}
            try:
                for name, expected in ref.items():
                    assert_close(name, expected, actual[name], ratio=ratios[name], warning=False)
                print(f"[{test_name}] Test Passed!\n")
            except AssertionError as error:
                print(f"[{test_name}] Test Failed: {error}\n")
                test_passed = False
        result = torch.tensor(test_passed, dtype=torch.int32, device=current_device)
        dist.broadcast(result, src=0)
        if not result.item():
            raise AssertionError("GDP context parallel result did not match the single-device reference")
    finally:
        cleanup_distributed()


@pytest.mark.parametrize("num_householder", [2, 3], ids=["n2", "n3"])
def test_cp_gdp_sequence_cut(num_householder: int) -> None:
    """CP2: sequences cut across rank boundary."""
    world_size = 2
    if device_torch_lib.device_count() < world_size:
        pytest.skip("At least 2 GPUs are required")
    test_name = f"CP2_SequenceCut_N{num_householder}"
    mp.start_processes(
        run_cp_gdp_test_worker,
        args=(world_size, test_name, num_householder),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
