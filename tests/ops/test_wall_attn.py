# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
import os

import pytest
import torch

from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.wall_attn import build_wall_kv_cache, naive_wall_attn, parallel_wall_attn, parallel_wall_attn_decode
from fla.utils import assert_close, device

# Wall's log-space `R` factoring and the gate-gradient reverse-cumsum are sensitive
# to TF32 matmuls (catastrophic cancellation for small gates), so force IEEE fp32
# dots for these correctness checks -- matching the convention in `test_attn.py`.
# Read by Triton at kernel-launch time, so setting it after imports is sufficient.
os.environ['TRITON_F32_DEFAULT'] = 'ieee'

# Wall scores with a per-block log-space reference `R` for the `exp2(P_i - P_j)`
# rescaling, where `P = cumsum(g)`. The factoring assumes `P` is monotonically
# decreasing, i.e. the gate is a true log-decay (`g <= 0`) -- which is what the
# layer produces via `logsigmoid` (initialized near zero, so per-step decay is
# small). We seed gates as small negatives accordingly; out-of-domain (two-sided)
# gates break the `R` factoring and are not a supported regime.
RTOL_FWD = 5e-3     # Triton prefill vs. exact eager reference (fp32)
RTOL_GRAD = 5e-3    # autograd dq/dk/dv vs. eager autograd (fp32)
RTOL_FD = 2e-2      # finite-difference gate gradients (inherently noisy)
RTOL_DECODE = 2e-2  # cached decode vs. prefill self-consistency (fp32/bf16)


def log_decay(*shape, scale=0.05, dtype=torch.float32):
    """A valid small Wall log-decay gate (`g <= 0`), as `logsigmoid` produces in practice."""
    return (-torch.randn(*shape, device=device, dtype=torch.float32).abs() * scale).to(dtype)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'K', 'V'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-K{}-V{}".format(*test))
        for test in [
            (1, 48, 2, 4, 32, 16),
            (2, 31, 1, 1, 24, 8),
        ]
    ],
)
@pytest.mark.parametrize('window_size', [None, 8])
def test_parallel_matches_reference(B: int, T: int, H: int, HQ: int, K: int, V: int, window_size):
    assert HQ % H == 0
    torch.manual_seed(0)
    dtype = torch.float32
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)
    scale = K**-0.5

    ref = naive_wall_attn(q, k, v, g, scale=scale, window_size=window_size)
    tri = parallel_wall_attn(q, k, v, g, scale=scale, window_size=window_size)
    assert_close(" o", ref, tri, RTOL_FWD)


def test_parallel_gqa_matches_reference():
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 40, 2, 8, 32, 24
    assert HQ // H == 4
    torch.manual_seed(1)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)
    scale = K**-0.5

    ref = naive_wall_attn(q, k, v, g, scale=scale)
    tri = parallel_wall_attn(q, k, v, g, scale=scale)
    assert_close(" o", ref, tri, RTOL_FWD)


def test_parallel_varlen_matches_reference():
    dtype = torch.float32
    T1, T2 = 17, 23
    T = T1 + T2
    H, HQ, K, V = 1, 2, 16, 12
    torch.manual_seed(2)
    q = torch.randn(1, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(1, T, H, K, device=device, dtype=dtype)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype)
    g = log_decay(1, T, HQ, K, dtype=dtype)
    cu = torch.tensor([0, T1, T], dtype=torch.long, device=device)
    scale = K**-0.5

    ref = naive_wall_attn(q, k, v, g, scale=scale, cu_seqlens=cu)
    tri = parallel_wall_attn(q, k, v, g, scale=scale, cu_seqlens=cu)
    assert_close(" o", ref, tri, RTOL_FWD)


def test_parallel_sink_bias_matches_reference():
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 29, 1, 2, 20, 10
    torch.manual_seed(3)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)
    sink_bias = torch.randn(HQ, device=device, dtype=dtype) * 0.1
    scale = K**-0.5

    ref = naive_wall_attn(q, k, v, g, scale=scale, sink_bias=sink_bias)
    tri = parallel_wall_attn(q, k, v, g, scale=scale, sink_bias=sink_bias)
    assert_close(" o", ref, tri, RTOL_FWD)


def test_parallel_aggressive_gates_long_seq():
    """Strong per-timestep decay; exact reference stays in fp32, kernel uses per-block R."""
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 512, 1, 1, 32, 32
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.full((B, T, HQ, K), math.log2(0.9), device=device, dtype=dtype)
    scale = K**-0.5

    ref = naive_wall_attn(q, k, v, g, scale=scale)
    tri = parallel_wall_attn(q, k, v, g, scale=scale)
    assert_close(" o", ref, tri, RTOL_FWD)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'K', 'V'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-K{}-V{}".format(*test))
        for test in [
            (1, 24, 2, 4, 16, 12),
            (1, 64, 2, 2, 64, 128),
        ]
    ],
)
def test_backward_matches_eager_reference(B: int, T: int, H: int, HQ: int, K: int, V: int):
    dtype = torch.float32
    torch.manual_seed(11)
    q0 = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k0 = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v0 = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g0 = log_decay(B, T, HQ, K, dtype=dtype)
    scale = K**-0.5

    q = q0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    g = g0.clone().requires_grad_(True)

    q2 = q0.clone().requires_grad_(True)
    k2 = k0.clone().requires_grad_(True)
    v2 = v0.clone().requires_grad_(True)
    g2 = g0.clone().requires_grad_(True)

    o = parallel_wall_attn(q, k, v, g, scale=scale)
    ref = naive_wall_attn(q2, k2, v2, g2, scale=scale)
    go = torch.randn_like(o)
    o.backward(go)
    ref.backward(go)

    assert_close("dq", q2.grad, q.grad, RTOL_GRAD)
    assert_close("dk", k2.grad, k.grad, RTOL_GRAD)
    assert_close("dv", v2.grad, v.grad, RTOL_GRAD)
    # The reference does not backprop through `g` (chunk_global_cumsum); the wall
    # `dg` is validated separately in `test_g_gradient_matches_finite_differences`.


def test_dg_nonzero_after_backward():
    torch.manual_seed(3)
    B, T, H, HQ, K, V = 1, 16, 1, 1, 8, 8
    q = torch.randn(B, T, HQ, K, device=device, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, requires_grad=True)
    g = log_decay(B, T, HQ, K).requires_grad_(True)
    o = parallel_wall_attn(q, k, v, g, scale=K**-0.5)
    o.sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


def test_g_gradient_matches_finite_differences():
    """dL/dg for the Triton Wall path vs central finite differences.

    The loss is accumulated in fp64; the step is sized for fp32 softmax logits.
    """
    torch.manual_seed(7)
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 4, 1, 1, 3, 3
    scale = K**-0.5
    eps = 3e-3

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g0 = log_decay(B, T, HQ, K, dtype=dtype)
    go = torch.randn(B, T, HQ, V, device=device, dtype=dtype)

    g = g0.clone().requires_grad_(True)
    o = parallel_wall_attn(q, k, v, g, scale=scale)
    (o * go).sum().backward()
    assert g.grad is not None
    dg_ana = g.grad.detach().clone()

    g_flat = g0.reshape(-1)
    dg_fd = torch.empty_like(g_flat)
    for i in range(g_flat.numel()):
        gp = g_flat.clone()
        gm = g_flat.clone()
        gp[i] += eps
        gm[i] -= eps
        op = parallel_wall_attn(q, k, v, gp.view_as(g0), scale=scale)
        om = parallel_wall_attn(q, k, v, gm.view_as(g0), scale=scale)
        Lp = (op * go).sum().double()
        Lm = (om * go).sum().double()
        dg_fd[i] = ((Lp - Lm) / (2.0 * eps)).to(dtype)

    dg_fd = dg_fd.view_as(dg_ana)
    assert_close("dg", dg_fd, dg_ana, RTOL_FD)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'K', 'V'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-K{}-V{}".format(*test))
        for test in [
            (1, 48, 2, 4, 32, 16),
            (2, 31, 1, 1, 24, 8),
        ]
    ],
)
def test_scalar_gate_matches_reference(B: int, T: int, H: int, HQ: int, K: int, V: int):
    """Wall + FoX-style additive scalar gate: Triton vs reference."""
    dtype = torch.float32
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)
    g_scalar = log_decay(B, T, HQ, dtype=dtype)
    scale = K**-0.5

    ref = naive_wall_attn(q, k, v, g, scale=scale, g_scalar=g_scalar)
    tri = parallel_wall_attn(q, k, v, g, scale=scale, g_scalar=g_scalar)
    assert_close(" o", ref, tri, RTOL_FWD)


def test_scalar_gate_gradient_finite_differences():
    """dL/dg_scalar for Wall + scalar gate via central differences."""
    torch.manual_seed(13)
    dtype = torch.float32
    B, T, H, HQ, K, V = 1, 4, 1, 1, 3, 3
    scale = K**-0.5
    eps = 3e-3

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g0 = log_decay(B, T, HQ, K, dtype=dtype)
    gs0 = log_decay(B, T, HQ, dtype=dtype)
    go = torch.randn(B, T, HQ, V, device=device, dtype=dtype)

    gs = gs0.clone().requires_grad_(True)
    o = parallel_wall_attn(q, k, v, g0, scale=scale, g_scalar=gs)
    (o * go).sum().backward()
    assert gs.grad is not None
    dgs_ana = gs.grad.detach().clone()

    gs_flat = gs0.reshape(-1)
    dgs_fd = torch.empty_like(gs_flat)
    for i in range(gs_flat.numel()):
        gsp = gs_flat.clone()
        gsm = gs_flat.clone()
        gsp[i] += eps
        gsm[i] -= eps
        op = parallel_wall_attn(q, k, v, g0, scale=scale, g_scalar=gsp.view_as(gs0))
        om = parallel_wall_attn(q, k, v, g0, scale=scale, g_scalar=gsm.view_as(gs0))
        Lp = (op * go).sum().double()
        Lm = (om * go).sum().double()
        dgs_fd[i] = ((Lp - Lm) / (2.0 * eps)).to(dtype)

    dgs_fd = dgs_fd.view_as(dgs_ana)
    assert_close("dg_scalar", dgs_fd, dgs_ana, RTOL_FD)


def _decode_at(t, q, k, v, P, scale, C, *, g_scalar_cumsum=None):
    """Decode at position `t` with the cache truncated to [0, t].

    Decode has no intra-cache causal mask (the query is assumed to come after
    everything in the cache), so we truncate the cache to the current position.
    """
    k_c = k[:, : t + 1].contiguous()
    v_c = v[:, : t + 1].contiguous()
    P_c = P[:, : t + 1].contiguous()
    gsc = g_scalar_cumsum[:, : t + 1].contiguous() if g_scalar_cumsum is not None else None

    k_tilde, r_cache = build_wall_kv_cache(k_c, P_c, chunk_size=C)
    o_dec, _ = parallel_wall_attn_decode(
        q=q[:, t: t + 1].contiguous(),
        v=v_c,
        p_curr=P[:, t: t + 1].contiguous(),
        k_tilde=k_tilde,
        r_cache=r_cache,
        sink_bias=None,
        scale=scale,
        cache_chunk_size=C,
        g_scalar_cumsum=gsc,
    )
    return o_dec


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'K', 'V', 'C'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-K{}-V{}-C{}".format(*test))
        for test in [
            (1, 256, 4, 4, 64, 64, 64),   # MHA
            (1, 256, 2, 8, 64, 64, 64),   # GQA, G=4
            (2, 128, 1, 2, 32, 32, 32),   # small
        ]
    ],
)
def test_decode_matches_training_forward(dtype, B: int, T: int, H: int, HQ: int, K: int, V: int, C: int):
    """Decode at position t reproduces the training forward output at that row."""
    torch.manual_seed(0)
    scale = K**-0.5
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)

    o_ref = parallel_wall_attn(q, k, v, g, scale=scale)
    P = chunk_global_cumsum(g, scale=RCP_LN2)

    for t in (T - 1, (T // 2 // C) * C + C - 1):
        o_dec = _decode_at(t, q, k, v, P, scale, C)
        assert_close("o_dec", o_ref[:, t: t + 1], o_dec, RTOL_DECODE)


def test_decode_matches_training_forward_long():
    """Long-context stability: per-block reference must keep exp2 finite."""
    dtype = torch.bfloat16
    torch.manual_seed(1)
    B, T, H, HQ, K, V, C = 1, 4096, 2, 4, 64, 64, 128
    scale = K**-0.5
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)

    o_ref = parallel_wall_attn(q, k, v, g, scale=scale)
    P = chunk_global_cumsum(g, scale=RCP_LN2)

    t = T - 1
    o_dec = _decode_at(t, q, k, v, P, scale, C)
    assert torch.isfinite(o_dec).all(), "decode output must be finite at long context"
    assert_close("o_dec_long", o_ref[:, t: t + 1], o_dec, RTOL_DECODE)


def test_decode_with_scalar_gate():
    """Wall + FoX-style scalar gate: decode matches training forward."""
    torch.manual_seed(2)
    dtype = torch.float32
    B, T, H, HQ, K, V, C = 1, 256, 2, 4, 64, 64, 64
    scale = K**-0.5
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)
    g_scalar = log_decay(B, T, HQ, dtype=dtype)

    o_ref = parallel_wall_attn(q, k, v, g, scale=scale, g_scalar=g_scalar)
    P = chunk_global_cumsum(g, scale=RCP_LN2)
    c = chunk_global_cumsum(g_scalar, scale=RCP_LN2)

    t = T - 1
    o_dec = _decode_at(t, q, k, v, P, scale, C, g_scalar_cumsum=c)
    assert_close("o_dec_scalar", o_ref[:, t: t + 1], o_dec, RTOL_DECODE)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_decode_streaming_matches_full_forward(dtype):
    """End-to-end serving pattern: prefill the cache, then decode token-by-token,
    appending each new (k_tilde, v) to a pre-allocated buffer, and compare each
    step against the batched training forward at that position.
    """
    torch.manual_seed(123)
    B, T_prefill, T_gen, H, HQ, K, V, C = 1, 96, 16, 2, 4, 64, 64, 32
    T = T_prefill + T_gen
    G = HQ // H
    scale = K**-0.5

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = log_decay(B, T, HQ, K, dtype=dtype)

    o_ref = parallel_wall_attn(q, k, v, g, scale=scale)
    P = chunk_global_cumsum(g, scale=RCP_LN2)

    NC_max = (T + C - 1) // C
    k_tilde_buf = torch.zeros(B, T, HQ, K, device=device, dtype=dtype)
    v_buf = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    r_cache_buf = torch.zeros(B, NC_max, HQ, K, device=device, dtype=P.dtype)

    # Prefill: bulk-build the cache for the first T_prefill tokens.
    k_tilde_pre, r_cache_pre = build_wall_kv_cache(
        k[:, :T_prefill].contiguous(), P[:, :T_prefill].contiguous(), chunk_size=C,
    )
    NC_pre = r_cache_pre.shape[1]
    k_tilde_buf[:, :T_prefill] = k_tilde_pre
    v_buf[:, :T_prefill] = v[:, :T_prefill]
    r_cache_buf[:, :NC_pre] = r_cache_pre

    for t in range(T_prefill, T):
        c = t // C
        if t % C == 0:
            r_cache_buf[:, c] = P[:, t]  # new chunk: freeze anchor R_c = P[t]
        R_c = r_cache_buf[:, c: c + 1]
        k_q_t = k[:, t: t + 1].repeat_interleave(G, dim=2)
        k_tilde_buf[:, t: t + 1] = (
            k_q_t.float() * torch.exp2(R_c.float() - P[:, t: t + 1].float())
        ).to(dtype)
        v_buf[:, t: t + 1] = v[:, t: t + 1]

        T_kv = t + 1
        NC_t = (T_kv + C - 1) // C
        o_t, _ = parallel_wall_attn_decode(
            q=q[:, t: t + 1].contiguous(),
            v=v_buf[:, :T_kv].contiguous(),
            p_curr=P[:, t: t + 1].contiguous(),
            k_tilde=k_tilde_buf[:, :T_kv].contiguous(),
            r_cache=r_cache_buf[:, :NC_t].contiguous(),
            sink_bias=None,
            scale=scale,
            cache_chunk_size=C,
        )
        assert_close("o_stream", o_ref[:, t: t + 1], o_t, RTOL_DECODE)


def test_decode_cache_layout_shapes():
    """Pre-rescaled cache has documented shapes; r_cache size == ceil(T/C)."""
    torch.manual_seed(3)
    B, T, H, HQ, K = 2, 200, 2, 8, 32
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    g = log_decay(B, T, HQ, K, dtype=torch.bfloat16)
    P = chunk_global_cumsum(g, scale=RCP_LN2)
    for C in (32, 64, 128):
        k_tilde, r_cache = build_wall_kv_cache(k, P, chunk_size=C)
        NC = (T + C - 1) // C
        assert k_tilde.shape == (B, T, HQ, K)
        assert r_cache.shape == (B, NC, HQ, K)
