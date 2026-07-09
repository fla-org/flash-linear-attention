# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Correctness tests for the GDN-2 (Gated DeltaNet 2) ops.
#
# The ground truth is the pure-PyTorch ``naive_recurrent_gdn2`` reference: a
# direct transcription of
#     S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
# with no chunking, no WY trick, no fused gates. Any kernel that disagrees with
# it is wrong.
#
# The suite exercises the full feature surface of both public kernels:
#   * forward + numeric backward (dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias)
#   * varlen / packed sequences (cu_seqlens) vs per-sequence reference
#   * use_gate_in_kernel (fused -exp(A_log)*softplus path) + dt_bias
#   * safe_gate + lower_bound (bounded gate activation)
#   * use_qk_l2norm_in_kernel
#   * state_v_first ([V, K] state layout)
#   * initial_state propagation
#   * non-chunk-multiple sequence lengths (boundary masking)
#   * fp16 and fp32 dtypes
#   * chunk <-> fused_recurrent cross-agreement
#   * return_intermediate_states (inference mode)
#   * the GatedDeltaNet2 layer (incl. GVA + short conv) end to end
#
# GDN-2 reuses KDA's gate activation verbatim, so the gate-in-kernel reference
# uses ``naive_kda_gate`` / ``naive_kda_lowerbound_gate``.

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.utils import assert_close, device


def _activate_g(g, A_log, dt_bias, safe_gate, lower_bound):
    """Reference gate activation matching the kernel's use_gate_in_kernel path."""
    if safe_gate:
        return naive_kda_lowerbound_gate(g.float(), A_log.float(), dt_bias.float() if dt_bias is not None else None,
                                         lower_bound=lower_bound)
    return naive_kda_gate(g.float(), A_log.float(), dt_bias.float() if dt_bias is not None else None)


def _rand_inputs(B, T, H, K, V, dtype, *, gate_in_kernel=False, b_scale=1.0, seed=42):
    """Well-conditioned GDN-2 inputs.

    q/k drawn raw (kernels L2-normalize via use_qk_l2norm_in_kernel, or the
    caller normalizes for the reference). g is a contracting log-decay so the
    state does not blow up across long sequences; when gate_in_kernel is set, g
    is instead the raw pre-activation and A_log/dt_bias are returned.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.5
    b = torch.rand(B, T, H, K, dtype=dtype, device=device) * b_scale
    w = torch.rand(B, T, H, V, dtype=dtype, device=device)
    A_log, dt_bias = None, None
    if gate_in_kernel:
        g = torch.randn(B, T, H, K, dtype=dtype, device=device)
        A_log = torch.log(torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(H * K, dtype=torch.float32, device=device)
    else:
        g = torch.empty(B, T, H, K, device=device, dtype=torch.float32).uniform_(-5.0, -0.1).to(dtype)
    return q, k, v, g, b, w, A_log, dt_bias


# =============================================================================
# fused_recurrent
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-scale{}-l2norm{}-{}".format(*p))
        for p in [
            (1, 64, 2, 32, 32, 1.0, False, torch.float32),
            (2, 128, 2, 64, 64, 0.5, False, torch.float32),
            (2, 100, 3, 64, 64, 1.0, True, torch.float32),     # non-chunk-multiple T, l2norm
            (1, 130, 2, 64, 128, 1.0, True, torch.float16),    # fp16, V != K
        ]
    ],
)
def test_fused_recurrent(B, T, H, K, V, scale, use_qk_l2norm_in_kernel, dtype):
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, dtype)

    # The naive reference never normalizes internally, so it always receives
    # pre-normalized q/k. The kernel either normalizes raw inputs itself
    # (flag on) or is handed the same pre-normalized tensors (flag off).
    qn = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    kn = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=qn, k=kn, v=v, g=g, b=b, w=w, scale=scale, output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gdn2(
        # Flag on: hand the kernel raw q/k and let it normalize. Flag off: hand
        # it the same pre-normalized tensors the reference saw.
        q=q if use_qk_l2norm_in_kernel else qn,
        k=k if use_qk_l2norm_in_kernel else kn,
        v=v, g=g, b=b, w=w, scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, output_final_state=True,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "has_dt_bias", "safe_gate"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-dt_bias{}-safe_gate{}".format(*p))
        for p in [
            (2, 100, 2, 64, 64, True, False),
            (2, 100, 2, 64, 64, False, False),
            (1, 128, 2, 64, 64, True, True),
        ]
    ],
)
def test_fused_recurrent_gate_in_kernel(B, T, H, K, V, has_dt_bias, safe_gate):
    """use_gate_in_kernel=True must match the manually-activated gate path."""
    dtype = torch.float32
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(B, T, H, K, V, dtype, gate_in_kernel=True)
    if not has_dt_bias:
        dt_bias = None
    lower_bound = -5.0 if safe_gate else None

    g_ref = _activate_g(g, A_log, dt_bias, safe_gate, lower_bound).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.float(), p=2, dim=-1).to(dtype),
        k=F.normalize(k.float(), p=2, dim=-1).to(dtype),
        v=v, g=g_ref, b=b, w=w, output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w, A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, lower_bound=lower_bound,
        output_final_state=True,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_recurrent_state_v_first():
    """state_v_first stores the state transposed to [V, K]; output must match."""
    dtype = torch.float32
    B, T, H, K, V = 2, 64, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, dtype)

    o0, ht0 = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, state_v_first=False, output_final_state=True)
    o1, ht1 = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, state_v_first=True, output_final_state=True)
    assert_close("o", o0, o1, 0.005)
    assert_close("ht", ht0, ht1.transpose(-1, -2), 0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_recurrent_with_initial_state():
    dtype = torch.float32
    B, T, H, K, V = 2, 64, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, dtype)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.float(), p=2, dim=-1).to(dtype),
        k=F.normalize(k.float(), p=2, dim=-1).to(dtype),
        v=v, g=g, b=b, w=w, initial_state=h0, output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w, initial_state=h0,
        use_qk_l2norm_in_kernel=True, output_final_state=True,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("cu_seqlens", "H", "K", "V"),
    [
        pytest.param(*p, id="cu_seqlens{}-H{}-K{}-V{}".format(*p))
        for p in [
            ([0, 64, 128], 2, 64, 64),
            ([0, 15, 100, 256], 2, 64, 64),       # ragged, non-chunk-multiple
        ]
    ],
)
def test_fused_recurrent_varlen(cu_seqlens, H, K, V):
    """Packed varlen recurrent run must equal running each sequence on its own."""
    dtype = torch.float32
    cu = torch.LongTensor(cu_seqlens).to(device)
    T, N = cu[-1].item(), len(cu_seqlens) - 1
    q, k, v, g, b, w, _, _ = _rand_inputs(1, T, H, K, V, dtype)
    h0 = torch.randn(N, H, K, V, device=device, dtype=torch.float32)

    tri, tri_ht = fused_recurrent_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w, initial_state=h0,
        use_qk_l2norm_in_kernel=True, output_final_state=True, cu_seqlens=cu,
    )

    refs, ref_hts = [], []
    for i in range(N):
        s, e = cu[i].item(), cu[i + 1].item()
        o_i, ht_i = naive_recurrent_gdn2(
            q=F.normalize(q[:, s:e].float(), p=2, dim=-1).to(dtype),
            k=F.normalize(k[:, s:e].float(), p=2, dim=-1).to(dtype),
            v=v[:, s:e], g=g[:, s:e], b=b[:, s:e], w=w[:, s:e],
            initial_state=h0[i:i + 1], output_final_state=True,
        )
        refs.append(o_i)
        ref_hts.append(ht_i)
    assert_close("o", torch.cat(refs, 1), tri, 0.005)
    assert_close("ht", torch.cat(ref_hts, 0), tri_ht, 0.005)


# =============================================================================
# chunk — forward + numeric backward
# =============================================================================
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_chunk_invalid_chunk_size(chunk_size):
    B, T, H, K, V = 1, 64, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, torch.float32)

    with pytest.raises(ValueError, match=r"`chunk_size` must be 64 for GDN-2"):
        chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, chunk_size=chunk_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "safe_gate", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-scale{}-l2norm{}-gate{}-safe{}-{}".format(*p))
        for p in [
            (1, 64, 2, 32, 32, 1.0, False, False, False, torch.float32),
            (2, 256, 2, 64, 64, 0.5, True, False, False, torch.float32),
            (2, 100, 3, 64, 64, 1.0, True, False, False, torch.float16),   # non-multiple T, fp16
            (2, 256, 2, 64, 64, 1.0, True, True, False, torch.float32),    # gate-in-kernel
            (1, 128, 2, 64, 64, 1.0, True, True, True, torch.float32),     # gate-in-kernel + safe_gate
        ]
    ],
)
def test_chunk(B, T, H, K, V, scale, use_qk_l2norm_in_kernel, use_gate_in_kernel, safe_gate, dtype):
    """Full forward + gradient comparison of the chunkwise kernel vs autograd
    through the naive recurrent reference."""
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(B, T, H, K, V, dtype, gate_in_kernel=use_gate_in_kernel)
    lower_bound = -5.0 if safe_gate else None
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

    leaves = [q, k, v, g, b, w, h0]
    if use_gate_in_kernel:
        leaves += [A_log, dt_bias]
    for t in leaves:
        t.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # ---- reference (naive recurrent through autograd) ----
    # The reference never normalizes internally, so it always sees normalized
    # q/k. To keep gradients flowing to the q/k leaves either way, the
    # normalization is applied to the leaf tensors directly here, and the
    # kernel below is handed raw leaves (flag on) or the same normalized
    # tensors (flag off).
    g_ref = _activate_g(g, A_log, dt_bias, safe_gate, lower_bound) if use_gate_in_kernel else g
    qn = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    kn = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=qn, k=kn, v=v, g=g_ref, b=b, w=w, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = {n: t.grad.clone() for n, t in
                 zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
    if use_gate_in_kernel:
        ref_grads["A_log"], ref_grads["dt_bias"] = A_log.grad.clone(), dt_bias.grad.clone()
    for t in leaves:
        t.grad = None

    # ---- triton chunk ----
    tri, tri_ht = chunk_gdn2(
        q=q if use_qk_l2norm_in_kernel else qn,
        k=k if use_qk_l2norm_in_kernel else kn,
        v=v, g=g, b=b, w=w, scale=scale,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=safe_gate, lower_bound=lower_bound,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_grads = {n: t.grad.clone() for n, t in
                 zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
    if use_gate_in_kernel:
        tri_grads["A_log"], tri_grads["dt_bias"] = A_log.grad.clone(), dt_bias.grad.clone()

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_grads["q"], tri_grads["q"], 0.01)
    assert_close("dk", ref_grads["k"], tri_grads["k"], 0.01)
    assert_close("dv", ref_grads["v"], tri_grads["v"], 0.01)
    assert_close("db", ref_grads["b"], tri_grads["b"], 0.02)
    assert_close("dw", ref_grads["w"], tri_grads["w"], 0.02)
    assert_close("dg", ref_grads["g"], tri_grads["g"], 0.02)
    assert_close("dh0", ref_grads["h0"], tri_grads["h0"], 0.01)
    if use_gate_in_kernel:
        assert_close("dA_log", ref_grads["A_log"], tri_grads["A_log"], 0.02, warning=True)
        assert_close("dt_bias", ref_grads["dt_bias"], tri_grads["dt_bias"], 0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunk_state_v_first():
    """state_v_first must give the same output and a transposed final state.

    The state-first flag flips the layout of *both* the initial and final state
    to [V, K], so the v-first call is fed a transposed h0 and its returned ht is
    transposed back for comparison.
    """
    dtype = torch.float32
    B, T, H, K, V = 2, 128, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, dtype)
    h0_kv = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()

    o0, ht0 = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, initial_state=h0_kv,
                         use_qk_l2norm_in_kernel=True, output_final_state=True, state_v_first=False)
    o1, ht1 = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, initial_state=h0_vk,
                         use_qk_l2norm_in_kernel=True, output_final_state=True, state_v_first=True)
    assert_close("o", o0, o1, 0.005)
    assert_close("ht", ht0, ht1.transpose(-1, -2), 0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("cu_seqlens", "H", "K", "V", "use_gate_in_kernel", "dtype"),
    [
        pytest.param(*p, id="cu_seqlens{}-H{}-K{}-V{}-gate{}-{}".format(*p))
        for p in [
            ([0, 64, 128], 2, 64, 64, False, torch.float32),
            ([0, 15, 100, 256], 2, 64, 64, False, torch.float16),     # ragged, non-multiple, fp16
            ([0, 100, 300, 512], 2, 64, 64, True, torch.float16),     # gate-in-kernel + varlen
        ]
    ],
)
def test_chunk_varlen(cu_seqlens, H, K, V, use_gate_in_kernel, dtype):
    """Packed varlen chunk run (fwd + grads) must equal per-sequence reference."""
    cu = torch.LongTensor(cu_seqlens).to(device)
    cu_cpu = cu.cpu()
    T, N = cu[-1].item(), len(cu_seqlens) - 1
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(1, T, H, K, V, dtype, gate_in_kernel=use_gate_in_kernel)
    h0 = torch.randn(N, H, K, V, dtype=torch.float32, device=device)

    leaves = [q, k, v, g, b, w, h0] + ([A_log, dt_bias] if use_gate_in_kernel else [])
    for t in leaves:
        t.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    tri, tri_ht = chunk_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=use_gate_in_kernel,
        cu_seqlens=cu, cu_seqlens_cpu=cu_cpu,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_grads = {n: t.grad.clone() for n, t in zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
    for t in leaves:
        t.grad = None

    refs, ref_hts = [], []
    for i in range(N):
        s, e = cu[i].item(), cu[i + 1].item()
        g_ref = (_activate_g(g[:, s:e], A_log, dt_bias, False, None) if use_gate_in_kernel else g[:, s:e])
        o_i, ht_i = naive_recurrent_gdn2(
            q=F.normalize(q[:, s:e].float(), p=2, dim=-1).to(dtype),
            k=F.normalize(k[:, s:e].float(), p=2, dim=-1).to(dtype),
            v=v[:, s:e], g=g_ref, b=b[:, s:e], w=w[:, s:e],
            initial_state=h0[i:i + 1], output_final_state=True,
        )
        refs.append(o_i)
        ref_hts.append(ht_i)
    ref, ref_ht = torch.cat(refs, 1), torch.cat(ref_hts, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = {n: t.grad.clone() for n, t in zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}

    assert_close("o", ref, tri, 0.006)
    assert_close("ht", ref_ht, tri_ht, 0.006)
    assert_close("dq", ref_grads["q"], tri_grads["q"], 0.012)
    assert_close("dk", ref_grads["k"], tri_grads["k"], 0.012)
    assert_close("dv", ref_grads["v"], tri_grads["v"], 0.012)
    assert_close("db", ref_grads["b"], tri_grads["b"], 0.02)
    assert_close("dw", ref_grads["w"], tri_grads["w"], 0.02)
    assert_close("dg", ref_grads["g"], tri_grads["g"], 0.02)
    assert_close("dh0", ref_grads["h0"], tri_grads["h0"], 0.012)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_chunk_matches_fused_recurrent(dtype):
    """The two production kernels must agree with each other (and the naive)."""
    B, T, H, K, V = 2, 256, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, dtype)
    common = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)

    o_chunk, ht_chunk = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **common)
    o_rec, ht_rec = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **common)
    assert_close("o", o_rec, o_chunk, 0.006)
    assert_close("ht", ht_rec, ht_chunk, 0.006)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_chunk_return_intermediate_states():
    """return_intermediate_states yields per-chunk pre-states h; the output must
    still match the normal path."""
    dtype = torch.float32
    B, T, H, K, V = 2, 192, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, K, V, dtype)

    o_ref, ht_ref = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w,
                               use_qk_l2norm_in_kernel=True, output_final_state=True)
    o, ht, h = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w,
                          use_qk_l2norm_in_kernel=True, output_final_state=True,
                          return_intermediate_states=True)
    NT = (T + 63) // 64
    assert h.shape == (B, NT, H, K, V), f"unexpected intermediate-state shape {tuple(h.shape)}"
    assert_close("o", o_ref, o, 0.005)
    assert_close("ht", ht_ref, ht, 0.005)


# =============================================================================
# layer — GatedDeltaNet2 (GVA + short conv) end to end
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("num_heads", "num_v_heads", "use_short_conv"),
    [
        pytest.param(*p, id="H{}-HV{}-conv{}".format(*p))
        for p in [
            (2, 2, False),
            (2, 4, False),    # GVA: num_v_heads > num_heads
            (2, 2, True),     # short conv path
        ]
    ],
)
def test_layer(num_heads, num_v_heads, use_short_conv):
    """The GatedDeltaNet2 layer must run fwd/bwd and produce finite grads for
    every parameter, including the GVA (num_v_heads > num_heads) and short-conv
    paths that the op-level tests do not reach."""
    from fla.layers import GatedDeltaNet2

    torch.manual_seed(0)
    hidden_size, head_dim, B, T = 128, 32, 2, 128
    layer = GatedDeltaNet2(
        hidden_size=hidden_size, head_dim=head_dim,
        num_heads=num_heads, num_v_heads=num_v_heads,
        use_short_conv=use_short_conv,
    ).to(device).to(torch.float32)
    layer.train()

    x = torch.randn(B, T, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == (B, T, hidden_size)
    assert torch.isfinite(o).all()

    o.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(p.grad).all(), f"{name}.grad has non-finite values"
