# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
from fla.utils import assert_close, device
from tests.cp.utils import gather_global, shard_tensor

pytestmark = pytest.mark.skipif(device != "cuda", reason="requires CUDA or ROCm GPUs")


def _run_chunk(
    rank: int,
    world_size: int,
    num_householder: int,
) -> None:
    current_device = torch.device(f"{device}:{rank}")
    dtype = torch.bfloat16
    B, T, H, K, V = 1, 256, 2, 64, 80
    lengths = [80, 110, 66]
    scale = K ** -0.5

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

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
                input=torch.randn(B, T, H, K, dtype=torch.float32, device=current_device),
                p=2,
                dim=-1,
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

    # offsets count output tokens; k/v/beta carry num_householder updates per token
    sequence_offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    cu_seqlens_global = torch.tensor(sequence_offsets, dtype=torch.long, device=current_device)

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
        # the recurrent reference applies scale to q before reading the state
        o_ref = torch.cat(outputs, dim=1)
        o_ref.backward(do_global)
        ref = {
            "o": o_ref.detach(),
            "dq": q_ref.grad.detach(),
            "dk": k_ref.grad.detach(),
            "dv": v_ref.grad.detach(),
            "dbeta": beta_ref.grad.detach(),
            "dg": g_ref.grad.detach(),
        }

    dist.barrier()
    cp_context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)
    q_local = shard_tensor(q_global, rank=rank, world_size=world_size).detach().clone().requires_grad_(True)
    k_local = shard_tensor(k_global, rank=rank, world_size=world_size).detach().clone().requires_grad_(True)
    v_local = shard_tensor(v_global, rank=rank, world_size=world_size).detach().clone().requires_grad_(True)
    beta_local = shard_tensor(beta_global, rank=rank, world_size=world_size).detach().clone().requires_grad_(True)
    g_local = shard_tensor(g_global, rank=rank, world_size=world_size).detach().clone().requires_grad_(True)
    do_local = shard_tensor(do_global, rank=rank, world_size=world_size).clone()

    dist.barrier()

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

    o_local.backward(do_local)

    actual = {
        "o": gather_global(o_local, world_size),
        "dq": gather_global(q_local.grad, world_size),
        "dk": gather_global(k_local.grad, world_size),
        "dv": gather_global(v_local.grad, world_size),
        "dbeta": gather_global(beta_local.grad, world_size),
        "dg": gather_global(g_local.grad, world_size),
    }

    if rank == 0:
        ratios = {"o": 5e-3, "dq": 8e-3, "dk": 8e-3, "dv": 8e-3, "dbeta": 2e-2, "dg": 2e-2}
        for name, expected in ref.items():
            assert_close(name, expected, actual[name], ratio=ratios[name], warning=False)


@pytest.mark.parametrize("num_householder", [2, 3], ids=["n2", "n3"])
def test_chunk(run_distributed, num_householder: int) -> None:
    run_distributed(worker=_run_chunk, world_size=2, num_householder=num_householder)
