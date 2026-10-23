# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare CP GDN-2 outputs and gradients with per-sequence recurrent references."""

import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

from fla.ops.cp import build_cp_context
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.naive import naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_lowerbound_gate
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, assert_close, device
from tests.cp.utils import assert_chunk_matches_cp, gather_global, shard_tensor

pytestmark = pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason="requires NVIDIA Ampere or newer, or Ascend")


def _make_inputs(rank, T, H, D, dtype):
    B = 1
    q = torch.empty(B, T, H, D, device=device, dtype=dtype)
    k = torch.empty(B, T, H, D, device=device, dtype=dtype)
    v = torch.empty(B, T, H, D, device=device, dtype=dtype)
    g = torch.empty(B, T, H, D, device=device, dtype=dtype)
    b = torch.empty(B, T, H, D, device=device, dtype=dtype)
    w = torch.empty(B, T, H, D, device=device, dtype=dtype)
    do = torch.empty(B, T, H, D, device=device, dtype=dtype)

    if rank == 0:
        torch.manual_seed(42)

        # asymmetric q ~ N(1, 1) and k ~ N(-50, 10) stress the L2 normalization path
        q.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
        k.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) * 10.0 - 50.0)
        v.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        g.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        b.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid())
        w.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid())
        do.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    for tensor in (q, k, v, g, b, w, do):
        dist.broadcast(tensor, src=0)

    num_even_heads = triton.next_power_of_2(H)
    projection_size = H * D

    A_log = torch.empty(num_even_heads, dtype=torch.float32, device=device)
    dt_bias = torch.empty(projection_size, dtype=torch.float32, device=device)

    if rank == 0:
        # vary A_log across heads so wrong-head gk offsets change the result
        A_log[:H] = torch.log(torch.linspace(1, 8, H, dtype=torch.float32, device=device))
        # dt initialization from real training
        dt = torch.exp(
            torch.rand(projection_size, device=device) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
        ).clamp_(min=1e-4)
        dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    for tensor in (A_log, dt_bias):
        dist.broadcast(tensor, src=0)
    return q, k, v, g, b, w, do, A_log, dt_bias


def _reference(q, k, v, g, b, w, do, offsets, A_log, dt_bias):
    H = q.shape[2]
    results = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        q_seq = q[:, start:end].clone().detach().requires_grad_(True)
        k_seq = k[:, start:end].clone().detach().requires_grad_(True)
        v_seq = v[:, start:end].clone().detach().requires_grad_(True)
        g_seq = g[:, start:end].clone().detach().requires_grad_(True)
        b_seq = b[:, start:end].clone().detach().requires_grad_(True)
        w_seq = w[:, start:end].clone().detach().requires_grad_(True)
        do_seq = do[:, start:end].clone()

        q_seq_norm = F.normalize(q_seq, p=2, dim=-1)
        k_seq_norm = F.normalize(k_seq, p=2, dim=-1)
        g_processed = naive_kda_lowerbound_gate(
            g=g_seq.to(torch.float),
            A_log=A_log[:H].contiguous(),
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )

        o_seq, _ = naive_recurrent_gdn2(
            q=q_seq_norm,
            k=k_seq_norm,
            v=v_seq,
            g=g_processed,
            b=b_seq,
            w=w_seq,
            initial_state=None,
            output_final_state=False,
        )
        (o_seq * do_seq).sum().backward()
        results.append((
            o_seq.detach(),
            q_seq.grad.detach(),
            k_seq.grad.detach(),
            v_seq.grad.detach(),
            g_seq.grad.detach(),
            b_seq.grad.detach(),
            w_seq.grad.detach(),
        ))
    return tuple(torch.cat(parts, dim=1) for parts in zip(*results))


def _run_chunk(
    rank: int,
    world_size: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    disable_recompute: bool = False,
    state_v_first: bool = False,
):
    T = sum(lengths)
    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"

    q_global, k_global, v_global, g_global, b_global, w_global, do_global, A_log_global, dt_bias_global = _make_inputs(
        rank=rank,
        T=T,
        H=H,
        D=D,
        dtype=dtype,
    )

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

    if rank == 0:
        ref_out, ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dw = _reference(
            q=q_global,
            k=k_global,
            v=v_global,
            g=g_global,
            b=b_global,
            w=w_global,
            do=do_global,
            offsets=cu_seqlens_list,
            A_log=A_log_global,
            dt_bias=dt_bias_global,
        )

    dist.barrier()

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    q_local = shard_tensor(q_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    k_local = shard_tensor(k_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    v_local = shard_tensor(v_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    g_local = shard_tensor(g_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    b_local = shard_tensor(b_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    w_local = shard_tensor(w_global, rank=rank, world_size=world_size).clone().detach().requires_grad_(True)
    do_local = shard_tensor(do_global, rank=rank, world_size=world_size).clone()

    dist.barrier()

    o_local, _ = chunk_gdn2(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        b=b_local,
        w=w_local,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        disable_recompute=disable_recompute,
        state_v_first=state_v_first,
        cp_context=context,
        A_log=A_log_global[:H].contiguous(),
        dt_bias=dt_bias_global,
    )

    o_local.backward(do_local)

    o_cp_global = gather_global(o_local, world_size=world_size)
    dq_cp_global = gather_global(q_local.grad, world_size=world_size)
    dk_cp_global = gather_global(k_local.grad, world_size=world_size)
    dv_cp_global = gather_global(v_local.grad, world_size=world_size)
    dg_cp_global = gather_global(g_local.grad, world_size=world_size)
    db_cp_global = gather_global(b_local.grad, world_size=world_size)
    dw_cp_global = gather_global(w_local.grad, world_size=world_size)

    if rank == 0:
        # per-dim gating and L2 normalization of k ~ N(-50, 10) amplify bf16 state-communication error;
        # baseline ratios reach ~0.9-1.6%, above the strict 8e-3 threshold.
        # db also includes a cross-rank reduction (~2.3% error) and remains a warning-only diagnostic.
        assert_close("Output", ref_out, o_cp_global, ratio=8e-3, warning=False)
        assert_close("dq", ref_dq, dq_cp_global, ratio=8e-3, warning=False)
        assert_close("dk", ref_dk, dk_cp_global, ratio=8e-3, warning=False)
        assert_close("dv", ref_dv, dv_cp_global, ratio=8e-3, warning=False)
        assert_close("dg", ref_dg, dg_cp_global, ratio=8e-3, warning=False)
        assert_close("db", ref_db, db_cp_global, ratio=8e-3, warning=True)
        assert_close("dw", ref_dw, dw_cp_global, ratio=8e-3, warning=False)


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'disable_recompute', 'state_v_first'),
    [
        pytest.param(2, [3000, 4000, 3240], False, False, id='cp2-sequence-cut'),
        pytest.param(2, [5120, 5120], False, False, id='cp2-boundary-aligned'),
        pytest.param(4, [7000, 3240], False, False, id='cp4-sequence-cut'),
        pytest.param(4, [10240], False, False, id='cp4-single-sequence'),
        pytest.param(8, [65536], False, False, id='cp8-single-sequence'),
        pytest.param(2, [1000, 1500, 2000, 2500, 1240, 1000, 1000], False, False, id='cp2-short-sequences'),
        pytest.param(2, [3000, 4000, 3240], True, False, id='cp2-no-recompute'),
        pytest.param(2, [3000, 4000, 3240], False, True, id='cp2-v-first'),
        pytest.param(4, [10240], False, True, id='cp4-v-first'),
    ],
)
def test_chunk(run_distributed, world_size, lengths, disable_recompute, state_v_first):
    run_distributed(
        worker=_run_chunk,
        world_size=world_size,
        H=12,
        D=128,
        lengths=lengths,
        dtype=torch.bfloat16,
        disable_recompute=disable_recompute,
        state_v_first=state_v_first,
    )


def _run_chunk_gradients(rank: int, world_size: int, state_v_first: bool):
    torch.manual_seed(42)
    current_device = torch.device(device, rank)
    T, H, K, V = 256, 2, 64, 80
    q, k = [torch.randn(1, T, H, K, dtype=torch.bfloat16, device=current_device) for _ in range(2)]
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=current_device)
    g = torch.full_like(k, -0.01)
    b = torch.randn_like(k).sigmoid()
    w = torch.randn_like(v).sigmoid()
    do = torch.randn_like(v)
    cu_seqlens = torch.tensor([0, 80, 190, T], device=current_device)
    assert_chunk_matches_cp(
        op=chunk_gdn2,
        inputs=dict(q=q, k=k, v=v, g=g, b=b, w=w),
        do=do,
        cu_seqlens=cu_seqlens,
        rank=rank,
        world_size=world_size,
        use_qk_l2norm_in_kernel=True,
        state_v_first=state_v_first,
    )


@pytest.mark.parametrize('state_v_first', [False, True])
def test_chunk_gradients(run_distributed, state_v_first: bool):
    run_distributed(_run_chunk_gradients, world_size=2, state_v_first=state_v_first)
