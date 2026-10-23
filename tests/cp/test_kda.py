# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare CP KDA outputs and gradients with per-sequence recurrent references."""

import contextlib
import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

with contextlib.suppress(ImportError):
    import torch_npu  # noqa: F401

from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, assert_close, device
from tests.cp.utils import assert_chunk_matches_cp, gather_global, shard_tensor

pytestmark = pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason="requires NVIDIA Ampere or newer, or Ascend")


def _make_inputs(rank, T, H, D, dtype, use_gate_in_kernel, safe_gate, lower_bound):
    B = 1
    q = torch.empty(B, T, H, D, device=device, dtype=dtype)
    k = torch.empty(B, T, H, D, device=device, dtype=dtype)
    v = torch.empty(B, T, H, D, device=device, dtype=dtype)
    g = torch.empty(B, T, H, D, device=device, dtype=dtype if use_gate_in_kernel else torch.float)
    beta = torch.empty(B, T, H, device=device, dtype=dtype)
    do = torch.empty(B, T, H, D, device=device, dtype=dtype)

    if rank == 0:
        torch.manual_seed(42)

        # q gets a +1.0 offset to keep L2 normalization observable;
        # extreme k magnitudes inflate chunk-vs-naive bf16 error past 8e-3 even without CP.
        q.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype) + 1.0)
        k.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        v.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        if use_gate_in_kernel:
            g.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))
        else:
            g.copy_(F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=torch.float)))
            g.clamp_(min=-5.0)
        beta.copy_(torch.randn(B, T, H, device=device, dtype=dtype).sigmoid())
        do.copy_(torch.randn(B, T, H, D, device=device, dtype=dtype))

    for tensor in (q, k, v, g, beta, do):
        dist.broadcast(tensor, src=0)

    num_even_heads = triton.next_power_of_2(H)
    projection_size = H * D

    A_log = torch.empty(num_even_heads, dtype=torch.float32, device=device)
    dt_bias = torch.empty(projection_size, dtype=torch.float32, device=device)

    if rank == 0:
        # vary A_log across heads so wrong-head gk offsets change the result
        if safe_gate and lower_bound is not None:
            A_log[:H] = torch.log(torch.linspace(1, 8, H, dtype=torch.float32, device=device))
        else:
            A_log.copy_(torch.log(torch.empty(num_even_heads, dtype=torch.float32, device=device).uniform_(1, 16)))
        # dt initialization from real training
        dt = torch.exp(
            torch.rand(projection_size, device=device) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
        ).clamp_(min=1e-4)
        dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    for tensor in (A_log, dt_bias):
        dist.broadcast(tensor, src=0)
    return q, k, v, g, beta, do, A_log, dt_bias


def _reference(q, k, v, g, beta, do, offsets, A_log, dt_bias, use_gate_in_kernel, lower_bound):
    H = q.shape[2]
    results = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        q_seq = q[:, start:end].clone().detach().requires_grad_(True)
        k_seq = k[:, start:end].clone().detach().requires_grad_(True)
        v_seq = v[:, start:end].clone().detach().requires_grad_(True)
        g_seq = g[:, start:end].clone().detach()
        beta_seq = beta[:, start:end].clone().detach().requires_grad_(True)
        do_seq = do[:, start:end].clone()

        q_seq_norm = F.normalize(q_seq, p=2, dim=-1)
        k_seq_norm = F.normalize(k_seq, p=2, dim=-1)

        if use_gate_in_kernel:
            g_seq_input = g_seq.requires_grad_(True)
            g_processed = naive_kda_lowerbound_gate(
                g=g_seq_input.to(torch.float),
                A_log=A_log[:H].contiguous(),
                dt_bias=dt_bias,
                lower_bound=lower_bound,
            )
            g_for_grad = g_seq_input
        else:
            # pre-gated inputs already contain per-token log decay
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
        (o_seq * do_seq).sum().backward()
        results.append((
            o_seq.detach(),
            q_seq.grad.detach(),
            k_seq.grad.detach(),
            v_seq.grad.detach(),
            g_for_grad.grad.detach(),
            beta_seq.grad.detach(),
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
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    use_tf32x3_affine_chain: bool | None = None,
    layout: str = 'contiguous',
):
    T = sum(lengths)
    assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
    if layout == 'zigzag':
        assert T % (2 * world_size) == 0, f"T={T} must be divisible by 2*world_size={2 * world_size} for zigzag"

    q_global, k_global, v_global, g_global, beta_global, do_global, A_log_global, dt_bias_global = _make_inputs(
        rank=rank,
        T=T,
        H=H,
        D=D,
        dtype=dtype,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
    cu_seqlens_global = torch.tensor(cu_seqlens_list, device=device, dtype=torch.long)

    if rank == 0:
        ref_out, ref_dq, ref_dk, ref_dv, ref_dg, ref_db = _reference(
            q=q_global,
            k=k_global,
            v=v_global,
            g=g_global,
            beta=beta_global,
            do=do_global,
            offsets=cu_seqlens_list,
            A_log=A_log_global,
            dt_bias=dt_bias_global,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
        )

    dist.barrier()

    context = build_cp_context(
        cu_seqlens=cu_seqlens_global,
        group=dist.group.WORLD,
        layout=layout,
        **({'use_tf32x3_affine_chain': use_tf32x3_affine_chain} if use_tf32x3_affine_chain is not None else {}),
    )

    q_local = shard_tensor(q_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    k_local = shard_tensor(k_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    v_local = shard_tensor(v_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    g_local = shard_tensor(g_global, rank=rank, world_size=world_size, layout=layout).clone().detach().requires_grad_(True)
    beta_local = shard_tensor(
        tensor=beta_global,
        rank=rank,
        world_size=world_size,
        layout=layout,
    ).clone().detach().requires_grad_(True)
    do_local = shard_tensor(do_global, rank=rank, world_size=world_size, layout=layout).clone()

    dist.barrier()

    o_local, _ = chunk_kda(
        q=q_local,
        k=k_local,
        v=v_local,
        g=g_local,
        beta=beta_local,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
        state_v_first=state_v_first,
        cp_context=context,
        A_log=A_log_global[:H].contiguous() if use_gate_in_kernel else None,
        dt_bias=dt_bias_global if use_gate_in_kernel else None,
    )

    o_local.backward(do_local)

    o_cp_global = gather_global(o_local, world_size=world_size, layout=layout)
    dq_cp_global = gather_global(q_local.grad, world_size=world_size, layout=layout)
    dk_cp_global = gather_global(k_local.grad, world_size=world_size, layout=layout)
    dv_cp_global = gather_global(v_local.grad, world_size=world_size, layout=layout)
    dg_cp_global = gather_global(g_local.grad, world_size=world_size, layout=layout)
    db_cp_global = gather_global(beta_local.grad, world_size=world_size, layout=layout)

    if rank == 0:
        # dg remains diagnostic: single-GPU chunk-vs-naive error is already ~0.0085;
        # db's cross-rank reduction adds ~1.5-2.4% error, so it also remains warning-only.
        assert_close("Output", ref_out, o_cp_global, ratio=8e-3, warning=False)
        assert_close("dq", ref_dq, dq_cp_global, ratio=8e-3, warning=False)
        assert_close("dk", ref_dk, dk_cp_global, ratio=8e-3, warning=False)
        assert_close("dv", ref_dv, dv_cp_global, ratio=8e-3, warning=False)
        assert_close("dg", ref_dg, dg_cp_global, ratio=8e-3, warning=True)
        assert_close("db", ref_db, db_cp_global, ratio=8e-3, warning=True)


@pytest.mark.parametrize(
    ('world_size', 'lengths', 'use_gate_in_kernel', 'safe_gate',
     'disable_recompute', 'state_v_first', 'use_tf32x3_affine_chain', 'layout'),
    [
        pytest.param(2, [3000, 4000, 3240], True, True, False, False, None, 'contiguous', id='cp2-sequence-cut'),
        pytest.param(2, [5120, 5120], True, True, False, False, None, 'contiguous', id='cp2-boundary-aligned'),
        pytest.param(4, [7000, 3240], True, True, False, False, None, 'contiguous', id='cp4-sequence-cut'),
        pytest.param(4, [10240], True, True, False, False, None, 'contiguous', id='cp4-single-sequence'),
        pytest.param(8, [65536], True, True, False, False, None, 'contiguous', id='cp8-single-sequence'),
        pytest.param(2, [1000, 1500, 2000, 2500, 1240, 1000, 1000], True, True,
                     False, False, None, 'contiguous', id='cp2-short-sequences'),
        pytest.param(2, [3000, 4000, 3240], True, True, True, False, None, 'contiguous', id='cp2-no-recompute'),
        pytest.param(2, [3000, 4000, 3240], False, False, False, False, None, 'contiguous', id='cp2-pregated'),
        pytest.param(2, [3000, 4000, 3240], False, True, False, False, None, 'contiguous', id='cp2-pregated-safe'),
        pytest.param(2, [3000, 4000, 3240], True, True, False, True, None, 'contiguous', id='cp2-v-first'),
        pytest.param(4, [10240], True, True, False, True, None, 'contiguous', id='cp4-v-first'),
        pytest.param(2, [3000, 4000, 3240], True, True, False, False, True, 'contiguous',
                     id='cp2-sequence-cut-legacy-precision-true'),
        pytest.param(4, [10240], True, True, False, False, True, 'contiguous', id='cp4-single-sequence-legacy-precision-true'),
        pytest.param(2, [3000, 4000, 3240], True, True, False, True, True,
                     'contiguous', id='cp2-v-first-legacy-precision-true'),
        pytest.param(2, [10240], True, True, False, False, None, 'zigzag', id='cp2-zigzag-single-sequence'),
        pytest.param(2, [3000, 4000, 3240], True, True, False, False, None, 'zigzag', id='cp2-zigzag-sequence-cut'),
        pytest.param(4, [1500, 2600, 3000, 3140], True, True, False, False, None, 'zigzag', id='cp4-zigzag-sequence-cut'),
        pytest.param(4, [2560, 2560, 2560, 2560], True, True, False, False, None, 'zigzag', id='cp4-zigzag-boundary-aligned'),
    ],
)
def test_chunk(
    run_distributed,
    world_size,
    lengths,
    use_gate_in_kernel,
    safe_gate,
    disable_recompute,
    state_v_first,
    use_tf32x3_affine_chain,
    layout,
):
    run_distributed(
        worker=_run_chunk,
        world_size=world_size,
        H=12,
        D=128,
        lengths=lengths,
        dtype=torch.bfloat16,
        disable_recompute=disable_recompute,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
        state_v_first=state_v_first,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
        layout=layout,
    )


def _run_chunk_gradients(rank: int, world_size: int, layout: str, state_v_first: bool):
    torch.manual_seed(42)
    current_device = torch.device(device, rank)
    T, H, K, V = 256, 2, 64, 80
    q, k = [torch.randn(1, T, H, K, dtype=torch.bfloat16, device=current_device) for _ in range(2)]
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=current_device)
    g = torch.full_like(k, -0.01)
    beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=current_device).sigmoid()
    do = torch.randn_like(v)
    cu_seqlens = torch.tensor([0, 80, 190, T], device=current_device)

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
