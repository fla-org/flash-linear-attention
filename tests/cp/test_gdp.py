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
from tests.cp.utils import assert_chunk_matches_cp, gather_global


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
    local_T = T // world_size
    start = rank * local_T
    end = start + local_T
    start_dp, end_dp = start * num_householder, end * num_householder

    q_local = q_global[:, start:end].detach().clone().requires_grad_(True)
    k_local = k_global[:, start_dp:end_dp].detach().clone().requires_grad_(True)
    v_local = v_global[:, start_dp:end_dp].detach().clone().requires_grad_(True)
    beta_local = beta_global[:, start_dp:end_dp].detach().clone().requires_grad_(True)
    g_local = g_global[:, start:end].detach().clone().requires_grad_(True)
    do_local = do_global[:, start:end].clone()

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

    test_passed = True
    if rank == 0:
        ratios = {"o": 5e-3, "dq": 8e-3, "dk": 8e-3, "dv": 8e-3, "dbeta": 2e-2, "dg": 2e-2}
        try:
            for name, expected in ref.items():
                assert_close(name, expected, actual[name], ratio=ratios[name], warning=False)
        except AssertionError:
            test_passed = False
    result = torch.tensor(test_passed, dtype=torch.int32, device=current_device)
    dist.broadcast(result, src=0)
    if not result.item():
        raise AssertionError("GDP context parallel result did not match the single-device reference")


@pytest.mark.parametrize('num_householder', [2, 3], ids=['n2', 'n3'])
def test_chunk(run_distributed, num_householder):
    run_distributed(_run_chunk, world_size=2, num_householder=num_householder)


def _run_chunk_gradients(rank: int, world_size: int, num_householder: int):
    torch.manual_seed(42)
    current_device = torch.device(device, rank)
    T, H, K, V = 256, 2, 64, 80
    q = F.normalize(torch.randn(1, T, H, K, device=current_device), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, T * num_householder, H, K, device=current_device), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, T * num_householder, H, V, device=current_device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, H, device=current_device))
    beta = torch.randn(1, T * num_householder, H, device=current_device, dtype=torch.bfloat16).sigmoid()
    do = torch.randn(1, T, H, V, device=current_device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 80, 190, T], device=current_device)
    assert_chunk_matches_cp(
        op=chunk_gated_delta_product,
        inputs=dict(q=q, k=k, v=v, g=g, beta=beta),
        do=do,
        cu_seqlens=cu_seqlens,
        rank=rank,
        world_size=world_size,
        num_householder=num_householder,
        ratio=0.005,
    )


@pytest.mark.parametrize('num_householder', [2, 3], ids=['n2', 'n3'])
def test_chunk_gradients(run_distributed, num_householder: int):
    run_distributed(_run_chunk_gradients, world_size=2, num_householder=num_householder)
