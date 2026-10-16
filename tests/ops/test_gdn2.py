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
# GDN-2 reuses KDA's gate activation verbatim, so the gate-in-kernel reference
# uses ``naive_kda_gate`` / ``naive_kda_lowerbound_gate``.

import inspect
import math
from functools import partial
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, assert_close, device


def _unwrap_autotuner(fn):
    while not hasattr(fn, 'configs'):
        fn = fn.fn
    return fn


def _activate_g(g, A_log, dt_bias, safe_gate, lower_bound):
    """Reference gate activation matching the kernel's use_gate_in_kernel path."""
    if safe_gate:
        return naive_kda_lowerbound_gate(
            g=g.float(),
            A_log=A_log.float() if A_log is not None else None,
            dt_bias=dt_bias.float() if dt_bias is not None else None,
            lower_bound=lower_bound,
        )
    return naive_kda_gate(
        g=g.float(),
        A_log=A_log.float(),
        dt_bias=dt_bias.float() if dt_bias is not None else None,
    )


def _rand_inputs(B, T, H, HV, K, V, dtype, *, gate_in_kernel=False, b_scale=1.0, seed=42):
    """Well-conditioned GDN-2 inputs: q/k on H heads, the rest on HV (GVA when HV > H).

    g is a contracting log-decay so the recurrent state stays bounded; with
    gate_in_kernel it is the raw pre-activation instead.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.5
    b = torch.rand(B, T, HV, K, dtype=dtype, device=device) * b_scale
    w = torch.rand(B, T, HV, V, dtype=dtype, device=device)
    A_log, dt_bias = None, None
    if gate_in_kernel:
        g = torch.randn(B, T, HV, K, dtype=dtype, device=device)
        A_log = torch.log(torch.empty(HV, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(HV * K, dtype=torch.float32, device=device)
    else:
        g = torch.empty(B, T, HV, K, device=device, dtype=torch.float32).uniform_(-5.0, -0.1).to(dtype)
    return q, k, v, g, b, w, A_log, dt_bias


# =============================================================================
# fused_recurrent
# =============================================================================
@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "scale", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-HV{}-K{}-V{}-scale{}-l2norm{}-{}".format(*p))
        for p in [
            (1, 64, 2, 2, 32, 32, 1.0, False, torch.float32),
            (2, 128, 2, 2, 64, 64, 0.5, False, torch.float32),
            (2, 100, 3, 3, 64, 64, 1.0, True, torch.float32),     # non-chunk-multiple T, l2norm
            (1, 130, 2, 2, 64, 128, 1.0, True, torch.float16),    # fp16, V != K
            (2, 128, 2, 4, 64, 64, 1.0, False, torch.float32),    # GVA: HV > H
            (2, 100, 2, 4, 64, 128, 1.0, True, torch.float16),    # GVA + l2norm + fp16, V != K
            (1, 64, 1, 1, 256, 256, 1.0, True, torch.float16),    # automatic V-first, fp16
            (1, 64, 1, 1, 256, 256, 1.0, True, torch.bfloat16),   # automatic V-first, bf16
            (1, 4, 1, 1, 48, 16, 1.0, False, torch.float32),      # non-power-of-2 K
        ]
    ],
)
def test_fused_recurrent(B, T, H, HV, K, V, scale, use_qk_l2norm_in_kernel, dtype):
    assert HV % H == 0
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, HV, K, V, dtype)
    if K & (K - 1):
        # poison the allocation past g so an unmasked out-of-bounds gate load fails loudly
        numel = B * T * HV * K
        storage = torch.full((numel + 64,), float("nan"), device=device)
        g = storage[:numel].view(B, T, HV, K)
        g.uniform_(-2.0, -0.1)

    # The reference gets normalized q/k expanded to HV heads; the kernel maps
    # value heads to qk heads itself (and normalizes when the flag is on).
    qn = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    kn = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=qn.repeat_interleave(HV // H, dim=2),
        k=kn.repeat_interleave(HV // H, dim=2),
        v=v,
        g=g,
        b=b,
        w=w,
        scale=scale,
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gdn2(
        q=q if use_qk_l2norm_in_kernel else qn,
        k=k if use_qk_l2norm_in_kernel else kn,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not IS_NPU, reason='Triton-Ascend fused recurrent routing is only exercised on NPU')
def test_fused_recurrent_npu_backend_routing(monkeypatch):
    """The public fused-recurrent call must enter the Triton-Ascend backend."""
    from fla.ops.gdn2.backends.triton_ascend import fused_recurrent as npu_impl

    spy = Mock(wraps=npu_impl.fused_recurrent_gdn2_fwd_npu)
    monkeypatch.setattr(npu_impl, 'fused_recurrent_gdn2_fwd_npu', spy)
    q, k, v, g, b, w, _, _ = _rand_inputs(1, 2, 1, 1, 32, 32, torch.float16)
    fused_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        output_final_state=True,
    )
    spy.assert_called_once()


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "has_a_log", "has_dt_bias", "safe_gate"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-a_log{}-dt_bias{}-safe_gate{}".format(*p))
        for p in [
            (2, 100, 2, 64, 64, True, True, False),
            (2, 100, 2, 64, 64, True, False, False),
            (1, 128, 2, 64, 64, True, True, True),
            (2, 100, 2, 64, 64, False, False, True),
            (1, 128, 2, 64, 64, False, True, True),
        ]
    ],
)
def test_fused_recurrent_gate_in_kernel(B, T, H, K, V, has_a_log, has_dt_bias, safe_gate):
    """use_gate_in_kernel=True must match the manually-activated gate path."""
    dtype = torch.float32
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(B, T, H, H, K, V, dtype, gate_in_kernel=True)
    if not has_a_log:
        A_log = None
    if not has_dt_bias:
        dt_bias = None
    lower_bound = -5.0 if safe_gate else None

    g_ref = _activate_g(g, A_log, dt_bias, safe_gate, lower_bound).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.float(), p=2, dim=-1).to(dtype),
        k=F.normalize(k.float(), p=2, dim=-1).to(dtype),
        v=v,
        g=g_ref,
        b=b,
        w=w,
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
def test_fused_recurrent_state_v_first():
    """state_v_first stores the state transposed to [V, K]; output must match."""
    dtype = torch.float32
    B, T, H, K, V = 2, 64, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)

    o0, ht0 = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, output_final_state=True, state_v_first=False)
    o1, ht1 = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, output_final_state=True, state_v_first=True)
    assert_close("o", o0, o1, 0.005)
    assert_close("ht", ht0, ht1.transpose(-1, -2), 0.005)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
def test_fused_recurrent_initial_state():
    dtype = torch.float32
    B, T, H, K, V = 2, 64, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.float(), p=2, dim=-1).to(dtype),
        k=F.normalize(k.float(), p=2, dim=-1).to(dtype),
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
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
    q, k, v, g, b, w, _, _ = _rand_inputs(1, T, H, H, K, V, dtype)
    h0 = torch.randn(N, H, K, V, device=device, dtype=torch.float32)

    tri, tri_ht = fused_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )

    refs, ref_hts = [], []
    for i in range(N):
        s, e = cu[i].item(), cu[i + 1].item()
        o_i, ht_i = naive_recurrent_gdn2(
            q=F.normalize(q[:, s:e].float(), p=2, dim=-1).to(dtype),
            k=F.normalize(k[:, s:e].float(), p=2, dim=-1).to(dtype),
            v=v[:, s:e],
            g=g[:, s:e],
            b=b[:, s:e],
            w=w[:, s:e],
            initial_state=h0[i:i + 1],
            output_final_state=True,
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
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, torch.float32)

    with pytest.raises(ValueError, match=r"`chunk_size` must be 64 for GDN-2"):
        chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, chunk_size=chunk_size)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "safe_gate", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-scale{}-l2norm{}-gate{}-safe{}-{}".format(*p))
        for p in [
            (1, 64, 2, 32, 32, 1.0, False, False, False, torch.float32),
            (2, 256, 2, 64, 64, 0.5, True, False, False, torch.float32),
            (2, 100, 3, 64, 64, 1.0, True, False, False, torch.float16),   # non-multiple T, fp16
            (1, 64, 1, 128, 128, 1.0, True, False, False, torch.bfloat16),
            (1, 64, 1, 256, 256, 1.0, True, False, False, torch.bfloat16),
            (1, 65, 2, 48, 32, 1.0, True, False, False, torch.float16),
            (1, 65, 2, 32, 48, 1.0, True, False, False, torch.float16),
            (2, 256, 2, 64, 64, 1.0, True, True, False, torch.float32),    # gate-in-kernel
            (1, 128, 2, 64, 64, 1.0, True, True, True, torch.float32),     # gate-in-kernel + safe_gate
        ]
    ] + [
        pytest.param(
            1, 32768, 1, 32, 32, 1.0, True, False, False, torch.float16,
            id='32k',
            marks=pytest.mark.skipif(
                not IS_NPU,
                reason='32K GDN-2 output/gradient reference check requires Ascend NPU',
            ),
        ),
    ],
)
def test_chunk(B, T, H, K, V, scale, use_qk_l2norm_in_kernel, use_gate_in_kernel, safe_gate, dtype):
    """Forward + gradient comparison: chunk kernel vs autograd through the naive reference."""
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(B, T, H, H, K, V, dtype, gate_in_kernel=use_gate_in_kernel)
    lower_bound = -5.0 if safe_gate else None
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

    leaves = [q, k, v, g, b, w, h0]
    if use_gate_in_kernel:
        leaves += [A_log, dt_bias]
    for t in leaves:
        t.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # q/k are normalized on the leaves so gradients flow to them either way.
    g_ref = _activate_g(g, A_log, dt_bias, safe_gate, lower_bound) if use_gate_in_kernel else g
    qn = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    kn = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    ref, ref_ht = naive_recurrent_gdn2(
        q=qn,
        k=kn,
        v=v,
        g=g_ref,
        b=b,
        w=w,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = {n: t.grad.clone() for n, t in
                 zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
    if use_gate_in_kernel:
        ref_grads["A_log"], ref_grads["dt_bias"] = A_log.grad.clone(), dt_bias.grad.clone()
    for t in leaves:
        t.grad = None

    tri, tri_ht = chunk_gdn2(
        q=q if use_qk_l2norm_in_kernel else qn,
        k=k if use_qk_l2norm_in_kernel else kn,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
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


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
def test_chunk_state_v_first():
    """state_v_first must give the same output and a transposed final state."""
    dtype = torch.float32
    B, T, H, K, V = 2, 128, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)
    h0_kv = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()

    o0, ht0 = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0_kv,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=False,
    )
    o1, ht1 = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    assert_close("o", o0, o1, 0.005)
    assert_close("ht", ht0, ht1.transpose(-1, -2), 0.005)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
@pytest.mark.parametrize(
    ("cu_seqlens", "H", "K", "V", "use_gate_in_kernel", "dtype", "state_v_first", "disable_recompute"),
    [
        pytest.param(*p, False, False, id="cu_seqlens{}-H{}-K{}-V{}-gate{}-{}".format(*p))
        for p in [
            ([0, 64, 128], 2, 64, 64, False, torch.float32),
            ([0, 15, 100, 256], 2, 64, 64, False, torch.float16),     # ragged, non-multiple, fp16
            ([0, 100, 300, 512], 2, 64, 64, True, torch.float16),     # gate-in-kernel + varlen
            ([0, 15, 80, 145], 2, 48, 32, False, torch.float16),
        ]
    ] + [
        pytest.param(
            [0, 15, 80, 145], 2, 48, 32, False, torch.float16, True, True,
            id='ragged-state-v-first-disable-recompute',
        ),
    ],
)
@pytest.mark.smoke
def test_chunk_varlen(cu_seqlens, H, K, V, use_gate_in_kernel, dtype, state_v_first, disable_recompute):
    """Packed varlen chunk run (fwd + grads) must equal per-sequence reference."""
    cu = torch.LongTensor(cu_seqlens).to(device)
    cu_cpu = cu.cpu()
    T, N = cu[-1].item(), len(cu_seqlens) - 1
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(1, T, H, H, K, V, dtype, gate_in_kernel=use_gate_in_kernel)
    h0 = torch.randn(N, H, K, V, dtype=torch.float32, device=device)

    leaves = [q, k, v, g, b, w, h0] + ([A_log, dt_bias] if use_gate_in_kernel else [])
    for t in leaves:
        t.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    tri, tri_ht = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        initial_state=h0.transpose(-1, -2).contiguous() if state_v_first else h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        disable_recompute=disable_recompute,
        state_v_first=state_v_first,
    )
    if state_v_first:
        tri_ht = tri_ht.transpose(-1, -2)
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
            v=v[:, s:e],
            g=g_ref,
            b=b[:, s:e],
            w=w[:, s:e],
            initial_state=h0[i:i + 1],
            output_final_state=True,
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


@pytest.mark.skipif(not IS_NPU, reason='Ascend verifier checks require NPU')
@pytest.mark.parametrize(
    ('name', 'tensor_names'),
    [
        pytest.param('chunk_gdn2_fwd_intra', ('q', 'k', 'v', 'gk', 'b', 'w_gate'), id='fwd'),
        pytest.param(
            'chunk_gdn2_bwd_wy_dqkg_fused',
            ('q', 'k', 'v', 'v_new', 'g', 'b', 'w_gate', 'A', 'h', 'do', 'dh', 'dv'),
            id='bwd',
        ),
    ],
)
@pytest.mark.parametrize(
    ('case', 'dtype', 'reason'),
    [
        pytest.param('accept', torch.float16, None, id='fp16'),
        pytest.param('accept', torch.bfloat16, None, id='bf16'),
        pytest.param('accept', torch.float32, None, id='fp32'),
        pytest.param('chunk_size', torch.float32, 'chunk_size=64', id='chunk-size'),
        pytest.param('q_device', torch.float32, 'NPU tensors', id='mixed-device'),
        pytest.param('q_dtype', torch.float32, 'unsupported dtype', id='integer-input'),
        pytest.param('cu_seqlens', torch.float32, 'NPU tensors', id='cpu-cu-seqlens'),
        pytest.param('chunk_indices', torch.float32, 'NPU tensors', id='cpu-chunk-indices'),
    ],
)
def test_chunk_npu_verifier(name, tensor_names, case, dtype, reason):
    """Check leaf metadata acceptance and rejection without launching kernels."""
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    tensor = torch.zeros(1, dtype=dtype, device=device)
    kwargs = dict.fromkeys(tensor_names, tensor)
    if name == 'chunk_gdn2_fwd_intra':
        kwargs['lower_bound'] = -2.0
    if case == 'chunk_size':
        kwargs['chunk_size'] = 32
    elif case == 'q_device':
        kwargs['q'] = tensor.cpu()
    elif case == 'q_dtype':
        kwargs['q'] = tensor.to(torch.int32)
    elif case in ('cu_seqlens', 'chunk_indices'):
        kwargs[case] = torch.tensor([0, 1], dtype=torch.int64)
    verifier = getattr(TritonAscendGDN2Backend(), f'{name}_verifier')
    accepted, actual_reason = verifier(**kwargs, scale=1.0)
    assert accepted == (reason is None)
    if reason is None:
        assert actual_reason is None
    else:
        assert reason in actual_reason


@pytest.mark.skipif(not IS_NPU, reason='Ascend verifier checks require NPU')
def test_chunk_npu_fwd_verifier_rejects_oversized_k():
    """Reject K values that exceed the grouped kernel's single-slab UB bound."""
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    x = torch.zeros(1, 1, 1, 257, dtype=torch.float16, device=device)
    accepted, reason = TritonAscendGDN2Backend().chunk_gdn2_fwd_intra_verifier(
        q=x,
        k=x,
        v=x,
        gk=x,
        b=x,
        w_gate=x,
        scale=1.0,
    )
    assert not accepted
    assert reason == 'GDN-2 Ascend intra requires next_power_of_2(K) <= 256 for UB capacity, got K=257 (BK=512)'


@pytest.mark.skipif(not IS_NPU, reason='Ascend verifier checks require NPU')
@pytest.mark.parametrize(
    ('case', 'dtype', 'expected', 'reason'),
    [
        pytest.param('accept', torch.float16, True, None, id='accept-fp16'),
        pytest.param('accept', torch.bfloat16, True, None, id='accept-bf16'),
        pytest.param('accept', torch.float32, True, None, id='accept-fp32'),
        pytest.param('mixed_device', torch.float16, False, 'NPU tensors', id='mixed-device'),
        pytest.param('unsupported_dtype', torch.int32, False, 'unsupported dtype', id='integer-input'),
        pytest.param(
            'unsupported_optional_dtype',
            torch.float16,
            False,
            'unsupported dtype',
            id='integer-a-log',
        ),
    ],
)
def test_fused_recurrent_npu_verifier(case, dtype, expected, reason):
    """Verifier decisions must reflect the backend's routing and dtype requirements."""
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    K = 32
    B, T, H, HV, V = 1, 1, 1, 1, 32
    q = torch.zeros(B, T, H, K, dtype=dtype, device=device)
    k = torch.zeros_like(q)
    g = torch.zeros(B, T, HV, K, dtype=dtype, device=device)
    b = torch.zeros_like(g)
    v = torch.zeros(B, T, HV, V, dtype=dtype, device=device)
    w = torch.zeros_like(v)
    verifier_kwargs = {}
    if case == 'mixed_device':
        q = q.cpu()
    elif case == 'unsupported_optional_dtype':
        verifier_kwargs['A_log'] = torch.zeros(HV, dtype=torch.int32, device=device)
    accepted, actual_reason = TritonAscendGDN2Backend().fused_recurrent_gdn2_fwd_verifier(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        **verifier_kwargs,
    )
    assert accepted is expected
    if reason is not None:
        assert reason in actual_reason


@pytest.mark.skipif(not IS_NPU, reason='Ascend dispatch and launch splitting require NPU')
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
def test_chunk_npu_launch_splits(varlen, monkeypatch):
    """Split launches must retain native-reference numerics and dispatch both leaf stages."""
    from fla.ops.gdn2.backends.triton_ascend import chunk_bwd, chunk_intra

    # exercise multiple launches and packed sequence boundaries with a small reference workload.
    monkeypatch.setattr(chunk_intra, '_LAUNCH_BLOCK_BUDGET', 4)
    fwd = Mock(wraps=chunk_intra.chunk_gdn2_fwd_intra_npu)
    bwd = Mock(wraps=chunk_bwd.chunk_gdn2_bwd_wy_dqkg_fused_npu)
    monkeypatch.setattr(chunk_intra, 'chunk_gdn2_fwd_intra_npu', fwd)
    monkeypatch.setattr(chunk_bwd, 'chunk_gdn2_bwd_wy_dqkg_fused_npu', bwd)
    if varlen:
        test_chunk_varlen(
            cu_seqlens=[0, 15, 100, 257],
            H=2,
            K=48,
            V=32,
            use_gate_in_kernel=False,
            dtype=torch.float16,
            state_v_first=False,
            disable_recompute=False,
        )
    else:
        test_chunk(
            B=2,
            T=257,
            H=2,
            K=48,
            V=32,
            scale=1.0,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            safe_gate=False,
            dtype=torch.float16,
        )
    fwd.assert_called_once()
    bwd.assert_called_once()


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_chunk_matches_fused_recurrent(dtype):
    """The two production kernels must agree with each other."""
    B, T, H, K, V = 2, 256, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)
    common = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)

    o_chunk, ht_chunk = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **common)
    o_rec, ht_rec = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **common)
    assert_close("o", o_rec, o_chunk, 0.006)
    assert_close("ht", ht_rec, ht_chunk, 0.006)


@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
@torch.inference_mode()
def test_chunk_return_intermediate_states():
    """return_intermediate_states yields per-chunk pre-states h; the output must
    still match the normal path."""
    dtype = torch.float32
    B, T, H, K, V = 2, 192, 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)

    o_ref, ht_ref = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o, ht, h = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        return_intermediate_states=True,
    )
    NT = (T + 63) // 64
    assert h.shape == (B, NT, H, K, V), f"unexpected intermediate-state shape {tuple(h.shape)}"
    assert_close("o", o_ref, o, 0.005)
    assert_close("ht", ht_ref, ht, 0.005)


# =============================================================================
# layer — GatedDeltaNet2 (GVA + short conv) end to end
# =============================================================================
@pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")
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
    every parameter, covering the GVA (num_v_heads > num_heads) head expansion
    and the short-conv path that the op-level tests do not reach."""
    from fla.layers import GatedDeltaNet2

    torch.manual_seed(0)
    hidden_size, head_dim, B, T = 128, 32, 2, 128
    layer = GatedDeltaNet2(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
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


def test_chunk_bwd_autotune_key_covers_head_dims():
    """The fused WY/dqkg backward must autotune each (K, V) geometry separately.

    The kernel sweeps BK/BV tile configs whose relative performance depends on
    the head dimensions, so K and V have to be part of the autotune key. Were
    they missing, the first (K, V) geometry to reach the kernel in a process
    would tune it for every other geometry sharing BT: the second geometry
    cache-hits on the first one's winner and runs a config that was never
    measured for its shape.

    Checked against the declared key rather than by driving two backwards,
    which would pay a cold autotune sweep per geometry on every CI run.
    """
    from fla.ops.gdn2.chunk_bwd import chunk_gdn2_bwd_kernel_wy_dqkg_fused

    tuner = _unwrap_autotuner(chunk_gdn2_bwd_kernel_wy_dqkg_fused)
    assert list(tuner.keys) == ['BT', 'K', 'V', 'STATE_V_FIRST'], (
        f"unexpected autotune key {list(tuner.keys)}; it must carry K and V, "
        "the head dimensions that bound the swept BK/BV tiles"
    )


@pytest.mark.parametrize('lower_bound', [None, -2.0, -5.0], ids=['default', 'shift2', 'shift5'])
def test_chunk_npu_fwd_lower_bound(lower_bound):
    """The Ascend verifier and entry accept the intra anchor override."""
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend
    from fla.ops.gdn2.backends.triton_ascend.chunk_intra import chunk_gdn2_fwd_intra_npu

    tensor = Mock(shape=(1, 22, 2, 64), dtype=torch.bfloat16)
    tensor.device.type = 'npu'
    kwargs = dict.fromkeys(('q', 'k', 'v', 'gk', 'b', 'w_gate'), tensor)
    kwargs.update(scale=1.0, safe_gate=True, lower_bound=lower_bound)
    assert TritonAscendGDN2Backend().verify('chunk_gdn2_fwd_intra', **kwargs) == (True, None)
    inspect.signature(chunk_gdn2_fwd_intra_npu).bind(**kwargs)


@pytest.mark.skipif(IS_NPU, reason="Ascend uses a separate pairwise-gate forward kernel")
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('use_gate_in_kernel', [False, True], ids=['precomputed', 'fused'])
@pytest.mark.parametrize(
    ("T", "P", "H", "K", "V"),
    [
        pytest.param(*test, id='T{}-P{}-H{}-K{}-V{}'.format(*test))
        for test in [
            (16, 6, 2, 64, 64),
            (22, 21, 2, 64, 64),
            (40, 33, 2, 64, 32),
            (70, 69, 2, 64, 64),
        ]
    ],
)
def test_chunk_causality(T: int, P: int, H: int, K: int, V: int, use_gate_in_kernel: bool, varlen: bool):
    """Future gates must leave prefix outputs and prefix-loss gradients unchanged."""
    torch.manual_seed(42)
    bos = 17 if varlen else 0
    shape = (1, bos + T, H, K)
    q = F.normalize(torch.randn(shape), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape), dim=-1).bfloat16()
    v = (0.5 * torch.randn(1, bos + T, H, V)).bfloat16()
    b = torch.rand(shape).bfloat16()
    w = torch.rand(1, bos + T, H, V).bfloat16()
    g = (-0.7 * (0.75 + 0.25 * torch.rand(shape))).bfloat16().to(device)
    A_log = torch.log(torch.empty(H).uniform_(1.0, 16.0))
    dt_bias = 0.1 * torch.rand(H * K) - 0.05
    q, k, v, b, w, A_log, dt_bias = (
        x.to(device).requires_grad_(True) for x in (q, k, v, b, w, A_log, dt_bias)
    )
    cu_seqlens = torch.tensor([0, bos, bos + T], dtype=torch.int64, device=device) if varlen else None
    do = torch.randn_like(v[:, bos:bos + P])
    edited = g.clone()
    if use_gate_in_kernel:
        edited[:, bos + P:] = (
            math.log(0.99 / 0.01) / A_log.detach().exp().view(1, 1, H, 1) - dt_bias.detach().view(1, 1, H, K)
        ).to(g.dtype)
    else:
        edited[:, bos + P:] = -4.95
    assert not torch.equal(g[:, bos + P:], edited[:, bos + P:])

    outputs, grads = [], []
    names = ('dq', 'dk', 'dv', 'dg', 'db', 'dw') + (('dA', 'dbias') if use_gate_in_kernel else ())
    for gates in (g, edited):
        gates = gates.requires_grad_(True)
        o, _ = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=gates,
            b=b,
            w=w,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=1.0,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=True,
            lower_bound=-5.0 if use_gate_in_kernel else None,
            cu_seqlens=cu_seqlens,
        )
        prefix = o[:, bos:bos + P]
        inputs = (q, k, v, gates, b, w) + ((A_log, dt_bias) if use_gate_in_kernel else ())
        outputs.append(prefix.detach())
        grads.append(torch.autograd.grad(outputs=(prefix.float() * do.float()).sum(), inputs=inputs))

    for name, ref, tri in zip(('o',) + names, (outputs[0],) + grads[0], (outputs[1],) + grads[1]):
        assert torch.isfinite(ref).all() and torch.isfinite(tri).all(), name
        assert torch.equal(ref, tri), f'future gates changed {name} from a prefix-only loss'


@pytest.mark.skipif(IS_NPU, reason="Ascend uses a separate pairwise-gate forward kernel")
@pytest.mark.parametrize(
    ('lower_bound', 'dtype', 'use_gate_in_kernel', 'cu_seqlens', 'gate_pattern'),
    [
        pytest.param(
            lower_bound,
            dtype,
            use_gate_in_kernel,
            cu_seqlens,
            gate_pattern,
            id=f'lower_bound{lower_bound}-{dtype}-use_gate{use_gate_in_kernel}-cu_seqlens{cu_seqlens}-{gate_pattern}',
        )
        for cu_seqlens in [None, [0, 1, 17, 38]]
        for use_gate_in_kernel in [False, True]
        for dtype in [torch.bfloat16, torch.float16]
        for lower_bound in [-5.0, -2.0, -0.01]
        for gate_pattern in (
            ['random', 'lower_bound', 'zero', 'mixed'] if not use_gate_in_kernel and lower_bound == -5.0 else ['random']
        )
    ],
)
def test_chunk_lower_bound(lower_bound, dtype, use_gate_in_kernel, cu_seqlens, gate_pattern):
    """Bounded gates preserve reference outputs, states, gradients, and precomputed-gate argument parity."""
    torch.manual_seed(42)
    B, T, H, K, V = 1, 38, 2, 64, 32
    shape = (B, T, H, K)
    q, k = (F.normalize(torch.randn(shape), dim=-1).to(dtype) for _ in range(2))
    v = (0.5 * torch.randn(B, T, H, V)).to(dtype)
    b = torch.rand(shape).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    g = (10 * torch.randn(shape)).to(dtype)
    if not use_gate_in_kernel:
        g = lower_bound * g.float().sigmoid()
        if gate_pattern == 'lower_bound':
            g.fill_(lower_bound)
        elif gate_pattern == 'zero':
            g.zero_()
        elif gate_pattern == 'mixed':
            g[..., ::2] = lower_bound
            g[..., 1::2] = 0.
    A_log = torch.zeros(H)
    dt_bias = 0.1 * torch.randn(H * K)
    offsets = cu_seqlens if cu_seqlens is not None else [0, T]
    h0 = torch.randn(len(offsets) - 1, H, K, V)
    q, k, v, g, b, w, h0, A_log, dt_bias = (
        x.to(device).requires_grad_(True) for x in (q, k, v, g, b, w, h0, A_log, dt_bias)
    )
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=device) if cu_seqlens is not None else None
    inputs = (q, k, v, g, b, w, h0)
    names = ('dq', 'dk', 'dv', 'dg', 'db', 'dw', 'dh0')
    tolerances = (0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01)
    if use_gate_in_kernel:
        inputs += (A_log, dt_bias)
        names += ('dA', 'dbias')
        tolerances += (0.02, 0.02)
    do, dht = torch.randn_like(v), torch.randn_like(h0)
    g_ref = _activate_g(g, A_log, dt_bias, True, lower_bound) if use_gate_in_kernel else g
    ref, ref_ht = [], []
    for i, (bos, eos) in enumerate(zip(offsets[:-1], offsets[1:])):
        o_i, ht_i = naive_recurrent_gdn2(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=g_ref[:, bos:eos],
            b=b[:, bos:eos],
            w=w[:, bos:eos],
            initial_state=h0[i:i + 1],
            output_final_state=True,
        )
        ref.append(o_i)
        ref_ht.append(ht_i)
    ref, ref_ht = torch.cat(ref, dim=1), torch.cat(ref_ht, dim=0)
    ref_grads = torch.autograd.grad(outputs=(ref * do).sum() + (ref_ht * dht).sum(), inputs=inputs)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=True,
        cu_seqlens=cu_seqlens,
    )
    if use_gate_in_kernel:
        kwargs.update(lower_bound=lower_bound, A_log=A_log, dt_bias=dt_bias)
    tri, tri_ht = chunk_gdn2(**kwargs)
    tri_grads = torch.autograd.grad(outputs=(tri * do).sum() + (tri_ht * dht).sum(), inputs=inputs)
    for name, ref_value, tri_value, tolerance in zip(
        ('o', 'ht') + names, (ref, ref_ht) + ref_grads, (tri, tri_ht) + tri_grads, (0.005, 0.005) + tolerances,
    ):
        assert torch.isfinite(ref_value).all() and torch.isfinite(tri_value).all(), name
        assert_close(name, ref_value, tri_value, tolerance)
    if not use_gate_in_kernel:
        bound_o, bound_ht = chunk_gdn2(**kwargs, lower_bound=lower_bound)
        bound_grads = torch.autograd.grad(outputs=(bound_o * do).sum() + (bound_ht * dht).sum(), inputs=inputs)
        for name, default, explicit in zip(
            ('o', 'ht') + names, (tri, tri_ht) + tri_grads, (bound_o, bound_ht) + bound_grads,
        ):
            assert torch.equal(default, explicit), f'precomputed-gate lower_bound changed {name}'


@pytest.mark.skipif(IS_NPU, reason="Ascend uses a separate pairwise-gate forward kernel")
@pytest.mark.parametrize('use_gate_in_kernel', [False, True], ids=['precomputed', 'fused'])
def test_chunk_intra_default_shift(use_gate_in_kernel, monkeypatch):
    """Default intra anchors match an explicit shift strength of two in forward and backward."""
    from fla.ops.gdn2 import chunk_bwd, chunk_fwd

    torch.manual_seed(42)
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(
        1, 70, 2, 2, 64, 32, torch.bfloat16, gate_in_kernel=use_gate_in_kernel,
    )
    q, k = (F.normalize(x.float(), dim=-1).to(x.dtype) for x in (q, k))
    inputs = (q, k, v, g, b, w) + ((A_log, dt_bias) if use_gate_in_kernel else ())
    for x in inputs:
        x.requires_grad_(True)
    do = torch.randn_like(v)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=True,
        lower_bound=-5.0 if use_gate_in_kernel else None,
    )
    default, _ = chunk_gdn2(**kwargs)
    default_grads = torch.autograd.grad(outputs=(default * do).sum(), inputs=inputs)
    monkeypatch.setattr(chunk_fwd, 'chunk_gdn2_fwd_intra', partial(chunk_fwd.chunk_gdn2_fwd_intra, lower_bound=-2.0))
    monkeypatch.setattr(chunk_bwd, 'chunk_gdn2_bwd_intra', partial(chunk_bwd.chunk_gdn2_bwd_intra, lower_bound=-2.0))
    explicit, _ = chunk_gdn2(**kwargs)
    explicit_grads = torch.autograd.grad(outputs=(explicit * do).sum(), inputs=inputs)
    for ref, tri in zip((default,) + default_grads, (explicit,) + explicit_grads):
        assert torch.isfinite(ref).all() and torch.isfinite(tri).all()
        assert torch.equal(ref, tri)
