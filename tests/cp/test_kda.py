# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import contextlib
import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

with contextlib.suppress(ImportError):
    import torch_npu  # noqa: F401

from cp.utils import assert_chunk_matches_cp, gather_global, slice_local
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close
from fla.utils import device as device_type


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
    layout: str = 'contiguous',
):
    device = torch.device(f'{device_type}:{rank}')

    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    if layout == 'zigzag':
        assert T % (2 * world_size) == 0, f"T={T} must be divisible by 2*world_size={2 * world_size} for zigzag"
    assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

    B = 1
    q_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    k_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    v_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    g_global = torch.empty(B, T, H, D, device=device, dtype=dtype if use_gate_in_kernel else torch.float)
    beta_global = torch.empty(B, T, H, device=device, dtype=dtype)
    do_global = torch.empty(B, T, H, D, device=device, dtype=dtype)
    A_log_global = None
    dt_bias_global = None

    if rank == 0:
        torch.manual_seed(42)
        # q gets a +1.0 offset to keep L2 normalization observable; extreme k
        # magnitudes inflate chunk-vs-naive bf16 error past 8e-3 even without CP.
        q_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
        k_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        v_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        if use_gate_in_kernel:
            g_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        else:
            g_global.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=torch.float)))
            g_global.clamp_(min=-5.0)
        beta_global.copy_(torch.randn(B, T, H, device=device, dtype=dtype).sigmoid())
        do_global.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    dist.broadcast(q_global, src=0)
    dist.broadcast(k_global, src=0)
    dist.broadcast(v_global, src=0)
    dist.broadcast(g_global, src=0)
    dist.broadcast(beta_global, src=0)
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
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db = None, None, None, None, None

    if rank == 0:
        N = len(lengths)
        ref_outputs = []

        for i in range(N):
            seq_start = cu_seqlens_list[i]
            seq_end = cu_seqlens_list[i + 1]

            q_seq = q_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            k_seq = k_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            v_seq = v_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
            g_seq = g_global[:, seq_start:seq_end].clone().detach()
            beta_seq = beta_global[:, seq_start:seq_end].clone().detach().requires_grad_(True)
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
                g_processed = g_seq.to(torch.float).requires_grad_(True)
                g_for_grad = g_processed

            o_seq, _ = naive_recurrent_kda(
                q=q_seq_norm,
                k=k_seq_norm,
                v=v_seq,
                g=g_processed,
                beta=beta_seq,
                initial_state=None,
                output_final_state=False,
            )

            ref_outputs.append({
                'o': o_seq,
                'q': q_seq,
                'k': k_seq,
                'v': v_seq,
                'g': g_for_grad,
                'beta': beta_seq,
                'do': do_seq,
            })

        all_dq = []
        all_dk = []
        all_dv = []
        all_dg = []
        all_db = []

        for item in ref_outputs:
            (item['o'] * item['do']).sum().backward()
            all_dq.append(item['q'].grad.detach())
            all_dk.append(item['k'].grad.detach())
            all_dv.append(item['v'].grad.detach())
            all_dg.append(item['g'].grad.detach())
            all_db.append(item['beta'].grad.detach())

        ref_out = torch.cat([item['o'].detach() for item in ref_outputs], dim=1)
        ref_dq = torch.cat(all_dq, dim=1)
        ref_dk = torch.cat(all_dk, dim=1)
        ref_dv = torch.cat(all_dv, dim=1)
        ref_dg = torch.cat(all_dg, dim=1)
        ref_db = torch.cat(all_db, dim=1)

    dist.barrier()

    context = build_cp_context(
        cu_seqlens=cu_seqlens_global,
        group=dist.group.WORLD,
        layout=layout,
    )

    q_local = slice_local(q_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    k_local = slice_local(k_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    v_local = slice_local(v_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    g_local = slice_local(g_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    beta_local = slice_local(
        beta_global,
        rank=rank,
        world_size=world_size,
        layout=layout,
    ).clone().detach().requires_grad_(True)
    do_local = slice_local(do_global, rank=rank, world_size=world_size, layout=layout).clone()

    dist.barrier()

    o_local, _ = chunk_kda(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        beta=beta_local,
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

    o_cp_global = gather_global(o_local, world_size=world_size, layout=layout)
    dq_cp_global = gather_global(q_local.grad, world_size=world_size, layout=layout)
    dk_cp_global = gather_global(k_local.grad, world_size=world_size, layout=layout)
    dv_cp_global = gather_global(v_local.grad, world_size=world_size, layout=layout)
    dg_cp_global = gather_global(g_local.grad, world_size=world_size, layout=layout)
    db_cp_global = gather_global(beta_local.grad, world_size=world_size, layout=layout)

    test_passed = True
    if rank == 0:

        # Tolerance: ratio=8e-3 for all tensors; dg/db warning-only: dg's
        # single-GPU chunk-vs-naive error is already ~0.0085; db sees cross-rank
        # reduction (~1.5-2.4%).
        tensors_to_verify = [
            ("Output", ref_out, o_cp_global),
            ("dq", ref_dq, dq_cp_global),
            ("dk", ref_dk, dk_cp_global),
            ("dv", ref_dv, dv_cp_global),
            ("dg", ref_dg, dg_cp_global),
            ("db", ref_db, db_cp_global),
        ]

        try:
            for name, ref, cp in tensors_to_verify:
                warn = name in ("dg", "db")
                assert_close(name, ref, cp, ratio=8e-3, warning=warn)
        except AssertionError:
            test_passed = False

    dist.barrier()

    if not test_passed:
        raise AssertionError(f"CP comparison failed on rank {rank}")


@pytest.mark.parametrize(
    ('world_size', 'T', 'lengths', 'disable_recompute', 'state_v_first', 'layout',
     'use_gate_in_kernel', 'safe_gate', 'lower_bound'),
    [
        pytest.param(2, 10240, [3000, 4000, 3240], False, False, 'contiguous', True, True, -5.0, id='cp2-sequence-cut'),
        pytest.param(2, 10240, [5120, 5120], False, False, 'contiguous', True, True, -5.0, id='cp2-boundary-aligned'),
        pytest.param(4, 10240, [7000, 3240], False, False, 'contiguous', True, True, -5.0, id='cp4-complex'),
        pytest.param(4, 10240, [10240], False, False, 'contiguous', True, True, -5.0, id='cp4-single-sequence'),
        pytest.param(8, 65536, [65536], False, False, 'contiguous', True, True, -5.0, id='cp8-single-sequence'),
        pytest.param(2, 10240, [1000, 1500, 2000, 2500, 1240, 1000, 1000],
                     False, False, 'contiguous', True, True, -5.0, id='cp2-many-short-sequences'),
        pytest.param(2, 10240, [3000, 4000, 3240], True, False, 'contiguous', True, True, -5.0, id='cp2-disable-recompute'),
        pytest.param(2, 10240, [3000, 4000, 3240], False, True, 'contiguous', True, True, -5.0, id='cp2-state-v-first'),
        pytest.param(4, 10240, [10240], False, True, 'contiguous', True, True, -5.0, id='cp4-state-v-first'),
        pytest.param(2, 10240, [10240], False, False, 'zigzag', True, True, -5.0, id='cp2-zigzag-single-sequence'),
        pytest.param(2, 10240, [3000, 4000, 3240], False, False, 'zigzag', True, True, -5.0, id='cp2-zigzag-sequence-cut'),
        pytest.param(4, 10240, [1500, 2600, 3000, 3140], False, False, 'zigzag', True, True, -5.0, id='cp4-zigzag-complex'),
        pytest.param(4, 10240, [2560, 2560, 2560, 2560], False, False, 'zigzag',
                     True, True, -5.0, id='cp4-zigzag-boundary-aligned'),
        pytest.param(2, 10240, [3000, 4000, 3240], False, False, 'contiguous',
                     False, False, None, id='cp2-pregated-nonsafe'),
        pytest.param(2, 10240, [3000, 4000, 3240], False, False, 'contiguous',
                     False, True, -5.0, id='cp2-pregated-safe'),
    ],
)
def test_chunk(
    run_distributed, world_size, T, lengths, disable_recompute, state_v_first, layout,
    use_gate_in_kernel, safe_gate, lower_bound,
):
    run_distributed(
        _run_chunk,
        world_size=world_size,
        T=T,
        H=12,
        D=128,
        lengths=lengths,
        dtype=torch.bfloat16,
        disable_recompute=disable_recompute,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
        layout=layout,
    )


def _run_chunk_gradients(rank: int, world_size: int, layout: str, state_v_first: bool):
    torch.manual_seed(42)
    device = torch.device(device_type, rank)
    T, H, K, V = 256, 2, 64, 80
    q, k = [torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device) for _ in range(2)]
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device)
    g = torch.full_like(k, -0.01)
    beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=device).sigmoid()
    do = torch.randn_like(v)
    cu_seqlens = torch.tensor([0, 80, 190, T], device=device)

    assert_chunk_matches_cp(
        op=chunk_kda,
        inputs=dict(q=q, k=k, v=v, g=g, beta=beta),
        do=do,
        cu_seqlens=cu_seqlens,
        rank=rank,
        world_size=world_size,
        layout=layout,
        use_qk_l2norm_in_kernel=True,
        state_v_first=state_v_first,
    )


@pytest.mark.parametrize('layout', ['contiguous', 'zigzag'])
@pytest.mark.parametrize('state_v_first', [False, True])
def test_chunk_gradients(run_distributed, layout: str, state_v_first: bool):
    run_distributed(_run_chunk_gradients, world_size=2, layout=layout, state_v_first=state_v_first)
