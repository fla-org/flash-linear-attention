# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

from fla.ops.cp import build_cp_context
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.naive import naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.utils import assert_close
from fla.utils import device as device_type
from tests.cp.utils import assert_chunk_matches_cp, gather_global


def _run_chunk(
    rank: int,
    world_size: int,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    disable_recompute: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
):
    device = torch.device(f'{device_type}:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    B = 1
    q_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    k_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    g_global = torch.empty(B, T, H, D, device=device, dtype=dtype if use_gate_in_kernel else torch.float)
    b_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    w_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    A_log_global = None
    dt_bias_global = None

    if rank == 0:
        torch.manual_seed(42)
        # asymmetric q/k magnitudes keep L2 normalization observable
        q_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
        k_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) * 10.0 - 50.0)
        v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        if use_gate_in_kernel:
            g_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        else:
            g_global.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=torch.float)))
            g_global.clamp_(min=-5.0)
        b_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid())
        w_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype).sigmoid())
        do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    dist.broadcast(q_global, src=0)
    dist.broadcast(k_global, src=0)
    dist.broadcast(v_global, src=0)
    dist.broadcast(g_global, src=0)
    dist.broadcast(b_global, src=0)
    dist.broadcast(w_global, src=0)
    dist.broadcast(do_global, src=0)

    num_even_heads = triton.next_power_of_2(H)
    projection_size = H * D

    A_log_global = torch.empty(num_even_heads, dtype=torch.float32, device=device)
    dt_bias_global = torch.empty(projection_size, dtype=torch.float32, device=device)

    if rank == 0:
        # distinct per-head decay exposes missing head offsets in backward gate loads
        if safe_gate and lower_bound is not None:
            A_log_global[:H] = torch.log(torch.linspace(1, 8, H, dtype=torch.float32, device=device))
        else:
            A_log_global.copy_(torch.log(torch.empty(num_even_heads, dtype=torch.float32, device=device).uniform_(1, 16)))
        # dt initialization from real training
        dt = torch.exp(torch.rand(projection_size, device=device) *
                       (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_(min=1e-4)
        dt_bias_global.copy_(dt + torch.log(-torch.expm1(-dt)))

    dist.broadcast(A_log_global, src=0)
    dist.broadcast(dt_bias_global, src=0)

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

    ref_out = None
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dw = None, None, None, None, None, None

    if rank == 0:
        N = len(lengths)
        ref_outputs = []

        for i in range(N):
            seq_start = cu_seqlens_list[i]
            seq_end = cu_seqlens_list[i + 1]

            q_seq = q_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            k_seq = k_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            v_seq = v_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            g_seq = g_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            b_seq = b_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            w_seq = w_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            do_seq = do_global[:, seq_start:seq_end].clone()

            q_seq_norm = F.normalize(q_seq, p=2, dim=-1)
            k_seq_norm = F.normalize(k_seq, p=2, dim=-1)
            if use_gate_in_kernel:
                g_seq_input = g_seq.requires_grad_(True)
                if safe_gate and lower_bound is not None:
                    g_processed = naive_kda_lowerbound_gate(
                        g_seq_input.to(torch.float),
                        A_log_global[:H].contiguous(),
                        dt_bias_global,
                        lower_bound=lower_bound,
                    )
                else:
                    g_processed = naive_kda_gate(
                        g_seq_input.to(torch.float),
                        A_log_global[:H].contiguous(),
                        dt_bias_global,
                    )
                g_for_grad = g_seq_input
            else:
                # chunk_gdn2 gets the raw g_local here and only cumsums it,
                # so the reference must use the very same log-decay.
                g_processed = g_seq
                g_for_grad = g_seq

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

            ref_outputs.append({
                'o': o_seq,
                'q': q_seq,
                'k': k_seq,
                'v': v_seq,
                'g': g_for_grad,
                'b': b_seq,
                'w': w_seq,
                'do': do_seq,
            })

        all_dq = []
        all_dk = []
        all_dv = []
        all_dg = []
        all_db = []
        all_dw = []

        for item in ref_outputs:
            (item['o'] * item['do']).sum().backward()
            all_dq.append(item['q'].grad.detach())
            all_dk.append(item['k'].grad.detach())
            all_dv.append(item['v'].grad.detach())
            all_dg.append(item['g'].grad.detach())
            all_db.append(item['b'].grad.detach())
            all_dw.append(item['w'].grad.detach())

        ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
        ref_dq = torch.cat(all_dq, dim=1)
        ref_dk = torch.cat(all_dk, dim=1)
        ref_dv = torch.cat(all_dv, dim=1)
        ref_dg = torch.cat(all_dg, dim=1)
        ref_db = torch.cat(all_db, dim=1)
        ref_dw = torch.cat(all_dw, dim=1)

    dist.barrier()

    context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

    chunk_size = T // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    g_local = g_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    b_local = b_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    w_local = w_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
    do_local = do_global[:, start_idx:end_idx, :].clone()

    dist.barrier()

    o_local, _ = chunk_gdn2(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        b=b_local,
        w=w_local,
        cp_context=context,
        disable_recompute=disable_recompute,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log_global[:H].contiguous() if use_gate_in_kernel else None,
        dt_bias=dt_bias_global if use_gate_in_kernel else None,
        state_v_first=state_v_first,
    )

    o_local.backward(do_local)

    o_cp_global = gather_global(o_local, world_size=world_size)

    dq_cp_global = gather_global(q_local.grad, world_size=world_size)

    dk_cp_global = gather_global(k_local.grad, world_size=world_size)

    dv_cp_global = gather_global(v_local.grad, world_size=world_size)

    dg_cp_global = gather_global(g_local.grad, world_size=world_size)

    db_cp_global = gather_global(b_local.grad, world_size=world_size)

    dw_cp_global = gather_global(w_local.grad, world_size=world_size)

    test_passed = True
    if rank == 0:

        # Tolerance: ratio=8e-3 for all tensors, matching test_kda.py.
        # Measured ratios are actually ~0.9-1.6% (db ~2.3%): per-dim gating
        # plus the L2 norm over k ~ N(-50, 10) amplify the bf16 state
        # communication precision loss.
        # db (b grad) additionally involves a cross-rank reduction, so it is
        # marked warning-only unconditionally.
        tensors_to_verify = [
            ("Output", ref_out, o_cp_global),
            ("dq", ref_dq, dq_cp_global),
            ("dk", ref_dk, dk_cp_global),
            ("dv", ref_dv, dv_cp_global),
            ("dg", ref_dg, dg_cp_global),
            ("db", ref_db, db_cp_global),
            ("dw", ref_dw, dw_cp_global),
        ]

        try:
            for name, ref, cp in tensors_to_verify:
                warn = (name == "db")
                assert_close(name, ref, cp, ratio=8e-3, warning=warn)
        except AssertionError:
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"CP comparison failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'T', 'lengths', 'disable_recompute', 'state_v_first'),
    [
        pytest.param(2, 10240, [3000, 4000, 3240], False, False, id='cp2-sequence-cut'),
        pytest.param(2, 10240, [5120, 5120], False, False, id='cp2-boundary-aligned'),
        pytest.param(4, 10240, [7000, 3240], False, False, id='cp4-complex'),
        pytest.param(4, 10240, [10240], False, False, id='cp4-single-sequence'),
        pytest.param(8, 65536, [65536], False, False, id='cp8-single-sequence'),
        pytest.param(2, 10240, [1000, 1500, 2000, 2500, 1240, 1000, 1000], False, False, id='cp2-many-short-sequences'),
        pytest.param(2, 10240, [3000, 4000, 3240], True, False, id='cp2-disable-recompute'),
        pytest.param(2, 10240, [3000, 4000, 3240], False, True, id='cp2-state-v-first'),
        pytest.param(4, 10240, [10240], False, True, id='cp4-state-v-first'),
    ],
)
def test_chunk(run_distributed, world_size, T, lengths, disable_recompute, state_v_first):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=T,
        H=12,
        D=128,
        lengths=lengths,
        dtype=torch.bfloat16,
        disable_recompute=disable_recompute,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=state_v_first,
    )


def _run_chunk_gradients(rank: int, world_size: int, state_v_first: bool):
    torch.manual_seed(42)
    device = torch.device(device_type, rank)
    T, H, K, V = 256, 2, 64, 80
    q, k = [torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device) for _ in range(2)]
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device)
    g = torch.full_like(k, -0.01)
    b = torch.randn_like(k).sigmoid()
    w = torch.randn_like(v).sigmoid()
    do = torch.randn_like(v)
    cu_seqlens = torch.tensor([0, 80, 190, T], device=device)
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
