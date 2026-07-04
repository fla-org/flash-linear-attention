# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Bitwise batch-invariance tests for the fused recurrent GDN kernel.

Under `fla.utils.batch_invariant_mode`, `fused_recurrent_gated_delta_rule`
guarantees two invariants, both checked here with `torch.equal` (bitwise):

1. Split invariance: processing a full sequence in one call produces exactly
   the same outputs and final state as a prefill call followed by
   token-by-token decode calls, with the fp32 recurrent state carried
   unmodified between calls.
2. Batch invariance: a sequence's outputs do not depend on the other
   sequences present in the batch.
"""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from fla.utils import batch_invariant_mode, device, is_batch_invariant_mode_enabled, set_batch_invariant_mode


def make_inputs(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    dtype: torch.dtype,
    gate_fusion: bool,
    with_initial_state: bool,
    seed: int = 42,
):
    """Build GDN inputs; with `gate_fusion`, `g`/`beta` are raw pre-activations."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    if gate_fusion:
        g = torch.randn(B, T, HV, dtype=torch.float32, device=device)
        beta = torch.randn(B, T, HV, dtype=dtype, device=device)
        A_log = torch.randn(HV, dtype=torch.float32, device=device)
        dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    else:
        g = F.logsigmoid(torch.randn(B, T, HV, dtype=torch.float32, device=device))
        beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
        A_log, dt_bias = None, None
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=device) if with_initial_state else None
    return q, k, v, g, beta, A_log, dt_bias, h0


def run(q, k, v, g, beta, A_log, dt_bias, h0, use_qk_l2norm: bool, gate_fusion: bool, state_v_first: bool = False):
    return fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        use_gate_in_kernel=gate_fusion,
        A_log=A_log,
        dt_bias=dt_bias,
        use_beta_sigmoid_in_kernel=gate_fusion,
        state_v_first=state_v_first,
    )


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'prefill_len', 'use_qk_l2norm', 'gate_fusion', 'with_initial_state', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-prefill{}-l2norm{}-gatefusion{}-h0{}-{}".format(*test),
        )
        for test in [
            (2, 64, 2, 4, 64, 37, False, False, False, torch.bfloat16),
            (2, 64, 2, 4, 64, 37, True, True, False, torch.bfloat16),
            (2, 64, 2, 4, 64, 37, True, False, True, torch.bfloat16),
            (2, 64, 2, 4, 64, 37, False, True, True, torch.bfloat16),
            (1, 100, 1, 1, 128, 63, True, True, False, torch.bfloat16),
            (2, 64, 2, 2, 100, 37, True, False, False, torch.float32),
            (2, 64, 2, 4, 64, 37, True, True, True, torch.float32),
        ]
    ],
)
def test_split_invariance_full_vs_prefill_decode(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    prefill_len: int,
    use_qk_l2norm: bool,
    gate_fusion: bool,
    with_initial_state: bool,
    dtype: torch.dtype,
):
    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H, HV, D, dtype, gate_fusion, with_initial_state)

    with batch_invariant_mode():
        o_full, ht_full = run(q, k, v, g, beta, A_log, dt_bias, h0, use_qk_l2norm, gate_fusion)

        # prefill, then token-by-token decode with the state carried between calls
        chunks = []
        state = h0
        splits = [(0, prefill_len)] + [(t, t + 1) for t in range(prefill_len, T)]
        for s, e in splits:
            o_step, state = run(
                q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                A_log, dt_bias, state, use_qk_l2norm, gate_fusion,
            )
            chunks.append(o_step)
        o_split = torch.cat(chunks, dim=1)

    assert torch.equal(o_full, o_split), \
        f"outputs diverge: max abs diff {(o_full.float() - o_split.float()).abs().max().item():.3e}"
    assert torch.equal(ht_full, state), \
        f"final states diverge: max abs diff {(ht_full - state).abs().max().item():.3e}"


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'use_qk_l2norm', 'gate_fusion', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-l2norm{}-gatefusion{}-{}".format(*test))
        for test in [
            (4, 64, 2, 4, 64, True, True, torch.bfloat16),
            (4, 64, 2, 2, 64, False, False, torch.float32),
        ]
    ],
)
def test_batch_invariance_batched_vs_single(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_qk_l2norm: bool,
    gate_fusion: bool,
    dtype: torch.dtype,
):
    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H, HV, D, dtype, gate_fusion, with_initial_state=True)

    with batch_invariant_mode():
        o_batch, ht_batch = run(q, k, v, g, beta, A_log, dt_bias, h0, use_qk_l2norm, gate_fusion)
        for b in range(B):
            o_one, ht_one = run(
                q[b:b + 1], k[b:b + 1], v[b:b + 1], g[b:b + 1], beta[b:b + 1],
                A_log, dt_bias, h0[b:b + 1], use_qk_l2norm, gate_fusion,
            )
            assert torch.equal(o_batch[b:b + 1], o_one), f"outputs of sequence {b} depend on the rest of the batch"
            assert torch.equal(ht_batch[b:b + 1], ht_one), f"final state of sequence {b} depends on the rest of the batch"


def test_split_invariance_state_v_first():
    """Split invariance for the V-first state layout, with K != V so that a
    transposed or mis-shaped state materialization cannot go unnoticed."""
    B, T, H, HV, K, V, prefill_len = 2, 64, 2, 4, 64, 32, 37
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, HV, dtype=torch.float32, device=device))
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(B, HV, V, K, dtype=torch.float32, device=device)

    with batch_invariant_mode():
        o_full, ht_full = run(q, k, v, g, beta, None, None, h0, False, False, state_v_first=True)

        chunks, state = [], h0
        splits = [(0, prefill_len)] + [(t, t + 1) for t in range(prefill_len, T)]
        for s, e in splits:
            o_step, state = run(
                q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                None, None, state, False, False, state_v_first=True,
            )
            chunks.append(o_step)

    assert torch.equal(o_full, torch.cat(chunks, dim=1))
    assert torch.equal(ht_full, state)


def test_prefill_single_decode_batched():
    """Crossing both invariants at once: prefill each sequence in its own call,
    then decode all sequences batched, against one full-batch full-sequence call."""
    B, T, H, HV, D, prefill_len = 4, 64, 2, 4, 64, 49
    q, k, v, g, beta, A_log, dt_bias, _ = make_inputs(
        B, T, H, HV, D, torch.bfloat16, gate_fusion=True, with_initial_state=False,
    )

    with batch_invariant_mode():
        o_full, ht_full = run(q, k, v, g, beta, A_log, dt_bias, None, True, True)

        prefill_outs, prefill_states = [], []
        for b in range(B):
            o_b, s_b = run(
                q[b:b + 1, :prefill_len], k[b:b + 1, :prefill_len], v[b:b + 1, :prefill_len],
                g[b:b + 1, :prefill_len], beta[b:b + 1, :prefill_len],
                A_log, dt_bias, None, True, True,
            )
            prefill_outs.append(o_b)
            prefill_states.append(s_b)

        chunks = [torch.cat(prefill_outs, dim=0)]
        state = torch.cat(prefill_states, dim=0)
        for t in range(prefill_len, T):
            o_t, state = run(
                q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], beta[:, t:t + 1],
                A_log, dt_bias, state, True, True,
            )
            chunks.append(o_t)

    assert torch.equal(o_full, torch.cat(chunks, dim=1))
    assert torch.equal(ht_full, state)


def test_varlen_split_invariance():
    """Split invariance also holds for varlen (`cu_seqlens`) prefill vs. batched decode."""
    T0, T1, H, HV, D = 37, 51, 2, 4, 64
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(1, T0 + T1, H, D, dtype=dtype, device=device)
    k = F.normalize(torch.randn(1, T0 + T1, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn(1, T0 + T1, HV, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, T0 + T1, HV, dtype=torch.float32, device=device))
    beta = torch.rand(1, T0 + T1, HV, dtype=dtype, device=device).sigmoid()
    cu_seqlens = torch.tensor([0, T0, T0 + T1], dtype=torch.long, device=device)

    with batch_invariant_mode():
        o_full, ht_full = fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

        # per-sequence prefill of all but the last token, then one decode step each
        o_parts, ht_parts = [], []
        for s, e in ((0, T0), (T0, T0 + T1)):
            o_pre, state = fused_recurrent_gated_delta_rule(
                q=q[:, s:e - 1], k=k[:, s:e - 1], v=v[:, s:e - 1], g=g[:, s:e - 1], beta=beta[:, s:e - 1],
                output_final_state=True,
            )
            o_dec, state = fused_recurrent_gated_delta_rule(
                q=q[:, e - 1:e], k=k[:, e - 1:e], v=v[:, e - 1:e], g=g[:, e - 1:e], beta=beta[:, e - 1:e],
                initial_state=state,
                output_final_state=True,
            )
            o_parts += [o_pre, o_dec]
            ht_parts.append(state)

    assert torch.equal(o_full, torch.cat(o_parts, dim=1))
    assert torch.equal(ht_full, torch.cat(ht_parts, dim=0))


def test_final_state_not_returned_when_not_requested():
    """The invariant-mode internal final-state buffer must not leak to the caller."""
    q, k, v, g, beta, *_ = make_inputs(1, 16, 1, 1, 64, torch.bfloat16, gate_fusion=False, with_initial_state=False)
    with batch_invariant_mode():
        o, final_state = fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, output_final_state=False)
    assert final_state is None
    assert o.shape == v.shape


def test_batch_invariant_mode_api():
    # save/restore so the test is independent of the FLA_BATCH_INVARIANT env var
    initial = is_batch_invariant_mode_enabled()
    try:
        set_batch_invariant_mode(False)
        assert not is_batch_invariant_mode_enabled()
        with batch_invariant_mode():
            assert is_batch_invariant_mode_enabled()
            with batch_invariant_mode(enabled=False):
                assert not is_batch_invariant_mode_enabled()
            assert is_batch_invariant_mode_enabled()
        assert not is_batch_invariant_mode_enabled()

        set_batch_invariant_mode(True)
        assert is_batch_invariant_mode_enabled()
    finally:
        set_batch_invariant_mode(initial)
