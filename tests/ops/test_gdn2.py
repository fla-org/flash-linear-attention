# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Correctness tests for the GDN-2 (Gated DeltaNet 2) ops.
#
# Native-size correctness uses the pure-PyTorch ``naive_recurrent_gdn2``
# reference: a direct transcription of
#     S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
# with no chunking, no WY trick, no fused gates. Any kernel that disagrees with
# it is wrong. With ``FLA_GDN2_LONG_SEQUENCE=1``, every case runs at T=32768;
# whole-vs-streaming and packed-vs-per-sequence equivalence replace the Python
# recurrence so the reference itself does not build a 32768-step autograd graph.
#
# GDN-2 reuses KDA's gate activation verbatim, so the gate-in-kernel reference
# uses ``naive_kda_gate`` / ``naive_kda_lowerbound_gate``.

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, assert_close, device

_requires_accelerator = pytest.mark.skipif(
    not (IS_NVIDIA or IS_AMD or IS_NPU),
    reason="CUDA/ROCm or Ascend NPU required",
)

_LONG_SEQUENCE = os.environ.get("FLA_GDN2_LONG_SEQUENCE") == "1"
_LONG_T = 32768
_LONG_SEGMENT_T = 4096
_LONG_LOSS_SCALE = 1.0 / 256
_LONG_DENSE_TOLERANCES = {
    "o": 0.005,
    "ht": 0.005,
    "dq": 0.01,
    "dk": 0.01,
    "dv": 0.01,
    "dg": 0.02,
    "db": 0.02,
    "dw": 0.02,
    "dh0": 0.01,
    "dA_log": 0.02,
    "dt_bias": 0.02,
}
_LONG_VARLEN_TOLERANCES = {
    **_LONG_DENSE_TOLERANCES,
    "o": 0.006,
    "ht": 0.006,
    "dq": 0.012,
    "dk": 0.012,
    "dv": 0.012,
    "dh0": 0.012,
}


def _test_length(native_length):
    return _LONG_T if _LONG_SEQUENCE else native_length


def _test_cu_seqlens(native_cu_seqlens):
    if not _LONG_SEQUENCE:
        return native_cu_seqlens

    total = native_cu_seqlens[-1]
    scaled = [0]
    for i, boundary in enumerate(native_cu_seqlens[1:-1], 1):
        value = round(boundary * _LONG_T / total)
        if value % 64 == 0:
            value += i
        scaled.append(value)
    return [*scaled, _LONG_T]


def _segmented_fused_recurrent(q, k, v, g, b, w, initial_state=None, **kwargs):
    outputs = []
    state = initial_state
    for start in range(0, q.shape[1], _LONG_SEGMENT_T):
        end = min(start + _LONG_SEGMENT_T, q.shape[1])
        output, state = fused_recurrent_gdn2(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            b=b[:, start:end],
            w=w[:, start:end],
            initial_state=state,
            output_final_state=True,
            **kwargs,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def _run_long_chunk(
    base,
    do,
    dht,
    *,
    segmented=False,
    cu_seqlens=None,
    separate_varlen=False,
    normalize_qk=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
):
    tensors = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in base.items() if tensor is not None}
    q = F.normalize(tensors["q"].float(), p=2, dim=-1).to(tensors["q"].dtype) if normalize_qk else tensors["q"]
    k = F.normalize(tensors["k"].float(), p=2, dim=-1).to(tensors["k"].dtype) if normalize_qk else tensors["k"]

    def call(start, end, state, packed_cu_seqlens=None):
        return chunk_gdn2(
            q=q[:, start:end],
            k=k[:, start:end],
            v=tensors["v"][:, start:end],
            g=tensors["g"][:, start:end],
            b=tensors["b"][:, start:end],
            w=tensors["w"][:, start:end],
            A_log=tensors.get("A_log"),
            dt_bias=tensors.get("dt_bias"),
            scale=scale,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=packed_cu_seqlens,
            cu_seqlens_cpu=packed_cu_seqlens.cpu() if packed_cu_seqlens is not None else None,
        )

    if separate_varlen:
        outputs, states = [], []
        cu_seqlens_cpu = cu_seqlens.cpu()
        for i in range(len(cu_seqlens_cpu) - 1):
            start, end = cu_seqlens_cpu[i].item(), cu_seqlens_cpu[i + 1].item()
            output, state = call(start, end, tensors["h0"][i: i + 1])
            outputs.append(output)
            states.append(state)
        output, state = torch.cat(outputs, dim=1), torch.cat(states, dim=0)
    elif segmented:
        outputs, state = [], tensors["h0"]
        for start in range(0, q.shape[1], _LONG_SEGMENT_T):
            end = min(start + _LONG_SEGMENT_T, q.shape[1])
            output_i, state = call(start, end, state)
            outputs.append(output_i)
        output = torch.cat(outputs, dim=1)
    else:
        output, state = call(0, q.shape[1], tensors["h0"], cu_seqlens)

    loss = ((output.float() * do.float()).sum() + (state.float() * dht.float()).sum()) * _LONG_LOSS_SCALE
    loss.backward()

    result = {"o": output.detach(), "ht": state.detach()}
    for name, tensor in tensors.items():
        assert tensor.grad is not None, f"{name}.grad is None"
        grad_name = "dt_bias" if name == "dt_bias" else f"d{name}"
        result[grad_name] = tensor.grad.detach()
    return result


def _assert_long_results(expected, actual, tolerances, warning_names=()):
    assert expected.keys() == actual.keys()
    for name in expected:
        assert_close(
            name,
            expected[name],
            actual[name],
            tolerances[name],
            warning=name in warning_names,
        )


def _assert_long_chunk_equivalent(
    base,
    do,
    dht,
    *,
    segmented=False,
    separate_varlen=False,
    tolerances=None,
    warning_names=(),
    **kwargs,
):
    if tolerances is None:
        tolerances = _LONG_DENSE_TOLERANCES
    reference = _run_long_chunk(
        base,
        do,
        dht,
        segmented=segmented,
        separate_varlen=separate_varlen,
        **kwargs,
    )
    whole = _run_long_chunk(base, do, dht, **kwargs)
    _assert_long_results(reference, whole, tolerances, warning_names)


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
@_requires_accelerator
@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "scale", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-HV{}-K{}-V{}-scale{}-l2norm{}-{}".format(*p))
        for p in [
            (1, _test_length(64), 2, 2, 32, 32, 1.0, False, torch.float32),
            (2, _test_length(128), 2, 2, 64, 64, 0.5, False, torch.float32),
            (2, _test_length(100), 3, 3, 64, 64, 1.0, True, torch.float32),
            (1, _test_length(130), 2, 2, 64, 128, 1.0, True, torch.float16),
            (2, _test_length(128), 2, 4, 64, 64, 1.0, False, torch.float32),
            (2, _test_length(100), 2, 4, 64, 128, 1.0, True, torch.float16),
            (1, _test_length(4), 1, 1, 48, 16, 1.0, False, torch.float32),
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
    if _LONG_SEQUENCE:
        ref, ref_ht = _segmented_fused_recurrent(
            q=q if use_qk_l2norm_in_kernel else qn,
            k=k if use_qk_l2norm_in_kernel else kn,
            v=v,
            g=g,
            b=b,
            w=w,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    else:
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


@_requires_accelerator
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "has_a_log", "has_dt_bias", "safe_gate"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-a_log{}-dt_bias{}-safe_gate{}".format(*p))
        for p in [
            (2, _test_length(100), 2, 64, 64, True, True, False),
            (2, _test_length(100), 2, 64, 64, True, False, False),
            (1, _test_length(128), 2, 64, 64, True, True, True),
            (2, _test_length(100), 2, 64, 64, False, False, True),
            (1, _test_length(128), 2, 64, 64, False, True, True),
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

    if _LONG_SEQUENCE:
        ref, ref_ht = _segmented_fused_recurrent(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
        )
    else:
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


@_requires_accelerator
def test_fused_recurrent_state_v_first():
    """state_v_first stores the state transposed to [V, K]; output must match."""
    dtype = torch.float32
    B, T, H, K, V = 2, _test_length(64), 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)

    o0, ht0 = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, output_final_state=True, state_v_first=False)
    o1, ht1 = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, output_final_state=True, state_v_first=True)
    assert_close("o", o0, o1, 0.005)
    assert_close("ht", ht0, ht1.transpose(-1, -2), 0.005)


@_requires_accelerator
def test_fused_recurrent_initial_state():
    dtype = torch.float32
    B, T, H, K, V = 2, _test_length(64), 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    if _LONG_SEQUENCE:
        ref, ref_ht = _segmented_fused_recurrent(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=h0,
            use_qk_l2norm_in_kernel=True,
        )
    else:
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


@_requires_accelerator
@pytest.mark.parametrize(
    ("cu_seqlens", "H", "K", "V"),
    [
        pytest.param(*p, id="cu_seqlens{}-H{}-K{}-V{}".format(*p))
        for p in [
            (_test_cu_seqlens([0, 64, 128]), 2, 64, 64),
            (_test_cu_seqlens([0, 15, 100, 256]), 2, 64, 64),
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
        if _LONG_SEQUENCE:
            o_i, ht_i = fused_recurrent_gdn2(
                q=q[:, s:e],
                k=k[:, s:e],
                v=v[:, s:e],
                g=g[:, s:e],
                b=b[:, s:e],
                w=w[:, s:e],
                initial_state=h0[i: i + 1],
                use_qk_l2norm_in_kernel=True,
                output_final_state=True,
            )
        else:
            o_i, ht_i = naive_recurrent_gdn2(
                q=F.normalize(q[:, s:e].float(), p=2, dim=-1).to(dtype),
                k=F.normalize(k[:, s:e].float(), p=2, dim=-1).to(dtype),
                v=v[:, s:e],
                g=g[:, s:e],
                b=b[:, s:e],
                w=w[:, s:e],
                initial_state=h0[i: i + 1],
                output_final_state=True,
            )
        refs.append(o_i)
        ref_hts.append(ht_i)
    assert_close("o", torch.cat(refs, 1), tri, 0.005)
    assert_close("ht", torch.cat(ref_hts, 0), tri_ht, 0.005)


# =============================================================================
# chunk — forward + numeric backward
# =============================================================================
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_chunk_invalid_chunk_size(chunk_size):
    B, T, H, K, V = 1, _test_length(64), 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, torch.float32)

    with pytest.raises(ValueError, match=r"`chunk_size` must be 64 for GDN-2"):
        chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, chunk_size=chunk_size)


@_requires_accelerator
@pytest.mark.parametrize(
    ("B", "T", "H", "K", "V", "scale", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "safe_gate", "dtype"),
    [
        pytest.param(*p, id="B{}-T{}-H{}-K{}-V{}-scale{}-l2norm{}-gate{}-safe{}-{}".format(*p))
        for p in [
            (1, _test_length(64), 2, 32, 32, 1.0, False, False, False, torch.float32),
            (2, _test_length(256), 2, 64, 64, 0.5, True, False, False, torch.float32),
            (2, _test_length(100), 3, 64, 64, 1.0, True, False, False, torch.float16),
            (2, _test_length(256), 2, 64, 64, 1.0, True, True, False, torch.float32),
            (1, _test_length(128), 2, 64, 64, 1.0, True, True, True, torch.float32),
        ]
    ],
)
def test_chunk(B, T, H, K, V, scale, use_qk_l2norm_in_kernel, use_gate_in_kernel, safe_gate, dtype):
    """Full forward + gradient comparison of the chunkwise kernel vs autograd
    through the naive recurrent reference, or vs streaming at 32K."""
    q, k, v, g, b, w, A_log, dt_bias = _rand_inputs(B, T, H, H, K, V, dtype, gate_in_kernel=use_gate_in_kernel)
    lower_bound = -5.0 if safe_gate else None
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

    if _LONG_SEQUENCE:
        base = {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w, "h0": h0}
        if use_gate_in_kernel:
            base.update(A_log=A_log, dt_bias=dt_bias)
        do = torch.randn_like(v)
        dht = torch.randn_like(h0)
        kwargs = dict(
            scale=scale,
            normalize_qk=not use_qk_l2norm_in_kernel,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        _assert_long_chunk_equivalent(
            base,
            do,
            dht,
            segmented=True,
            warning_names=("dA_log",),
            **kwargs,
        )
        return

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
    ref_grads = {n: t.grad.clone() for n, t in zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
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
    tri_grads = {n: t.grad.clone() for n, t in zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
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


@pytest.mark.skipif(not IS_NPU, reason="Ascend NPU required")
def test_chunk_npu_forward_backward(monkeypatch):
    """Ascend chunk path must agree with the recurrent reference on the NPU baseline."""
    B, T, H, K, V = 1, _test_length(64), 1, 32, 32
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, torch.float16)
    q = F.normalize(q.float(), p=2, dim=-1).to(torch.float16).detach().requires_grad_()
    k = F.normalize(k.float(), p=2, dim=-1).to(torch.float16).detach().requires_grad_()
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    leaves = (q, k, v, g, b, w, h0)
    for tensor in leaves:
        tensor.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    from fla.ops.gdn2.backends import gdn2_registry

    backend = gdn2_registry.get_active()
    assert backend is not None and backend.backend_type == "triton_ascend"
    accepted, reason = backend.chunk_gdn2_fwd_intra_verifier(q, k, v, g, b, w, 1.0)
    assert accepted and reason is None
    accepted, reason = backend.chunk_gdn2_fwd_intra_verifier(q, k, v, g, b, w, 1.0, chunk_size=32)
    assert not accepted and reason == "GDN-2 Ascend intra requires chunk_size=64, got 32"
    routed = {"fwd": 0, "bwd": 0}
    original_fwd = backend.chunk_gdn2_fwd_intra
    original_bwd = backend.chunk_gdn2_bwd_wy_dqkg_fused

    def routed_fwd(*args, **kwargs):
        routed["fwd"] += 1
        return original_fwd(*args, **kwargs)

    def routed_bwd(*args, **kwargs):
        routed["bwd"] += 1
        return original_bwd(*args, **kwargs)

    monkeypatch.setattr(backend, "chunk_gdn2_fwd_intra", routed_fwd)
    monkeypatch.setattr(backend, "chunk_gdn2_bwd_wy_dqkg_fused", routed_bwd)

    if _LONG_SEQUENCE:
        base = {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w, "h0": h0}
        _assert_long_chunk_equivalent(base, do, dht, segmented=True)
        assert routed["fwd"] > 0 and routed["bwd"] > 0
        return

    ref, ref_ht = naive_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = [tensor.grad.detach().clone() for tensor in leaves]
    for tensor in leaves:
        tensor.grad = None

    actual, actual_ht = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
    )
    ((actual * do).sum() + (actual_ht * dht).sum()).backward()
    actual_grads = [tensor.grad for tensor in leaves]

    assert_close("o", ref, actual, 0.005)
    assert_close("ht", ref_ht, actual_ht, 0.005)
    for name, expected, observed, tol in zip(
        ("dq", "dk", "dv", "dg", "db", "dw", "dh0"),
        ref_grads,
        actual_grads,
        (0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01),
    ):
        assert_close(name, expected, observed, tol)
    assert routed["fwd"] > 0 and routed["bwd"] > 0


@pytest.mark.skipif(not IS_NPU, reason="Ascend NPU required")
def test_chunk_npu_varlen_forward_backward():
    """Ascend grouped intra path must preserve ragged sequence boundaries."""
    cu = torch.tensor(_test_cu_seqlens([0, 7, 31, 65]), dtype=torch.long, device=device)
    cu_cpu = cu.cpu()
    T, N, H, K, V = cu_cpu[-1].item(), len(cu) - 1, 2, 64, 48
    q, k, v, g, b, w, _, _ = _rand_inputs(1, T, H, H, K, V, torch.float16)
    q = F.normalize(q.float(), p=2, dim=-1).to(torch.float16).detach().requires_grad_()
    k = F.normalize(k.float(), p=2, dim=-1).to(torch.float16).detach().requires_grad_()
    h0 = torch.randn(N, H, K, V, dtype=torch.float32, device=device)
    leaves = (q, k, v, g, b, w, h0)
    for tensor in leaves:
        tensor.requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    if _LONG_SEQUENCE:
        base = {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w, "h0": h0}
        _assert_long_chunk_equivalent(
            base,
            do,
            dht,
            cu_seqlens=cu,
            separate_varlen=True,
            tolerances=_LONG_VARLEN_TOLERANCES,
        )
        return

    refs, ref_states = [], []
    for i in range(N):
        start, end = cu_cpu[i].item(), cu_cpu[i + 1].item()
        ref, ref_state = naive_recurrent_gdn2(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            b=b[:, start:end],
            w=w[:, start:end],
            initial_state=h0[i: i + 1],
            output_final_state=True,
        )
        refs.append(ref)
        ref_states.append(ref_state)
    expected = torch.cat(refs, dim=1)
    expected_state = torch.cat(ref_states, dim=0)
    ((expected * do).sum() + (expected_state * dht).sum()).backward(retain_graph=True)
    expected_grads = [tensor.grad.detach().clone() for tensor in leaves]
    for tensor in leaves:
        tensor.grad = None

    actual, actual_state = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    ((actual * do).sum() + (actual_state * dht).sum()).backward()

    assert_close("o", expected, actual, 0.006)
    assert_close("ht", expected_state, actual_state, 0.006)
    for name, expected_grad, actual_grad, tol in zip(
        ("dq", "dk", "dv", "dg", "db", "dw", "dh0"),
        expected_grads,
        (tensor.grad for tensor in leaves),
        (0.012, 0.012, 0.012, 0.02, 0.02, 0.02, 0.012),
    ):
        assert_close(name, expected_grad, actual_grad, tol)


@_requires_accelerator
def test_chunk_state_v_first():
    """state_v_first must give the same output and a transposed final state."""
    dtype = torch.float32
    B, T, H, K, V = 2, _test_length(128), 2, 64, 64
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


@_requires_accelerator
@pytest.mark.parametrize(
    ("cu_seqlens", "H", "K", "V", "use_gate_in_kernel", "dtype"),
    [
        pytest.param(*p, id="cu_seqlens{}-H{}-K{}-V{}-gate{}-{}".format(*p))
        for p in [
            (_test_cu_seqlens([0, 64, 128]), 2, 64, 64, False, torch.float32),
            (_test_cu_seqlens([0, 15, 100, 256]), 2, 64, 64, False, torch.float16),
            (_test_cu_seqlens([0, 100, 300, 512]), 2, 64, 64, True, torch.float16),
        ]
    ],
)
def test_chunk_varlen(cu_seqlens, H, K, V, use_gate_in_kernel, dtype):
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

    if _LONG_SEQUENCE:
        base = {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w, "h0": h0}
        if use_gate_in_kernel:
            base.update(A_log=A_log, dt_bias=dt_bias)
        kwargs = dict(
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=use_gate_in_kernel,
        )
        _assert_long_chunk_equivalent(
            base,
            do,
            dht,
            separate_varlen=True,
            tolerances=_LONG_VARLEN_TOLERANCES,
            **kwargs,
        )
        return

    tri, tri_ht = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_grads = {n: t.grad.clone() for n, t in zip(("q", "k", "v", "g", "b", "w", "h0"), (q, k, v, g, b, w, h0))}
    for t in leaves:
        t.grad = None

    refs, ref_hts = [], []
    for i in range(N):
        s, e = cu[i].item(), cu[i + 1].item()
        g_ref = _activate_g(g[:, s:e], A_log, dt_bias, False, None) if use_gate_in_kernel else g[:, s:e]
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


@_requires_accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_chunk_matches_fused_recurrent(dtype):
    """The two production kernels must agree with each other (and the naive)."""
    B, T, H, K, V = 2, _test_length(256), 2, 64, 64
    q, k, v, g, b, w, _, _ = _rand_inputs(B, T, H, H, K, V, dtype)
    common = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)

    o_chunk, ht_chunk = chunk_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **common)
    o_rec, ht_rec = fused_recurrent_gdn2(q=q, k=k, v=v, g=g, b=b, w=w, **common)
    assert_close("o", o_rec, o_chunk, 0.006)
    assert_close("ht", ht_rec, ht_chunk, 0.006)


@_requires_accelerator
@torch.inference_mode()
def test_chunk_return_intermediate_states():
    """return_intermediate_states yields per-chunk pre-states h; the output must
    still match the normal path."""
    dtype = torch.float32
    B, T, H, K, V = 2, _test_length(192), 2, 64, 64
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
@_requires_accelerator
@pytest.mark.parametrize(
    ("num_heads", "num_v_heads", "use_short_conv"),
    [
        pytest.param(*p, id="H{}-HV{}-conv{}".format(*p))
        for p in [
            (2, 2, False),
            (2, 4, False),  # GVA: num_v_heads > num_heads
            (2, 2, True),  # short conv path
        ]
    ],
)
def test_layer(num_heads, num_v_heads, use_short_conv):
    """The GatedDeltaNet2 layer must run fwd/bwd and produce finite grads for
    every parameter, covering the GVA (num_v_heads > num_heads) head expansion
    and the short-conv path that the op-level tests do not reach."""
    from fla.layers import GatedDeltaNet2

    torch.manual_seed(0)
    hidden_size, head_dim, B, T = 128, 32, 2, _test_length(128)
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
