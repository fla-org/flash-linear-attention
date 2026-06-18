# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.parallax.decode import parallax_decode, parallax_decode_one_step
from fla.ops.parallax.naive import naive_parallax
from fla.ops.parallax.parallel import parallel_parallax
from fla.utils import assert_close, check_shared_mem, device


def _ref_varlen(q, r, k, v, cu_seqlens, window_size=None):
    out = q.new_empty(q.shape)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        out[:, bos:eos] = naive_parallax(
            q=q[:, bos:eos].float(),
            r=r[:, bos:eos].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
            window_size=window_size,
        ).to(q.dtype)
    return out


# bf16 carries fewer mantissa bits than fp16, and the `r` correction amplifies
# rounding, so it gets a looser ratio (bf16 backward grads run ~1e-2 relative).
TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-scale{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1.0),
            (3, 111, 2, 2, 100, 1.0),
            (3, 1024, 2, 8, 60, 0.1),
            (3, 1024, 2, 8, 128, 0.1),
            (4, 2048, 2, 8, 64, 0.1),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    ref = naive_parallax(q=q.float(), r=r.float(), k=k.float(), v=v.float(), scale=scale)
    ref = ref.to(dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_parallax(q=q, r=r, k=k, v=v, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 16),
            (3, 111, 2, 2, 100, 32),
            (3, 1024, 2, 8, 128, 64),
            (2, 2048, 2, 8, 64, 256),
            (2, 1024, 2, 2, 64, 200),    # W > tile_size and W % tile_size != 0 (safe-zone boundary)
        ]
    ],
)
def test_parallel_swa(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    W: int,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    ref = naive_parallax(q=q.float(), r=r.float(), k=k.float(), v=v.float(), window_size=W)
    ref = ref.to(dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_parallax(q=q, r=r, k=k, v=v, window_size=W)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-cu{}".format(*test))
        for test in [
            (2, 2, 64, [0, 15]),
            (2, 8, 64, [0, 256, 500, 1000]),
            (2, 2, 100, [0, 15, 100, 300, 1200, 2000]),
        ]
    ],
)
def test_parallel_varlen(H: int, HQ: int, D: int, cu_seqlens: list[int], dtype: torch.dtype):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = _ref_varlen(q, r, k, v, cu_seqlens)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_parallax(q=q, r=r, k=k, v=v, cu_seqlens=cu)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'W', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-W{}-cu{}".format(*test))
        for test in [
            (2, 2, 64, 16, [0, 111]),
            (2, 8, 100, 32, [0, 256, 500, 1000]),
        ]
    ],
)
def test_parallel_swa_varlen(H: int, HQ: int, D: int, W: int, cu_seqlens: list[int], dtype: torch.dtype):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = _ref_varlen(q, r, k, v, cu_seqlens, window_size=W)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_parallax(q=q, r=r, k=k, v=v, window_size=W, cu_seqlens=cu)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)


def _decode_ref(q, r, k, v, scale, window_size=None):
    """End-aligned causal reference: query i sits at absolute position Skv-Sq+i."""
    B, Sq, HQ, D = q.shape
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    kv_off = Skv - Sq
    q = q.float().reshape(B, Sq, H, G, D)
    r = r.float().reshape(B, Sq, H, G, D)
    k = k.float()
    v = v.float()
    s1 = torch.einsum('bqhgd,bkhd->bhgqk', q, k) * scale         # [B,H,G,Sq,Skv]
    s2 = torch.einsum('bqhgd,bkhd->bhgqk', r, k)
    i = torch.arange(Sq, device=q.device)[:, None]
    j = torch.arange(Skv, device=q.device)[None, :]
    absq = kv_off + i
    mask = j <= absq
    if window_size is not None:
        mask = mask & (j >= absq - window_size + 1)
    s1 = s1.masked_fill(~mask[None, None, None], float('-inf'))
    m = s1.amax(dim=-1, keepdim=True)
    p1 = (s1 - m).exp()
    d1 = p1.sum(dim=-1)
    p2 = p1 * s2
    d2 = p2.sum(dim=-1)
    o1 = torch.einsum('bhgqk,bkhd->bqhgd', p1, v)
    o2 = torch.einsum('bhgqk,bkhd->bqhgd', p2, v)
    c_norm = (d2 / d1).permute(0, 3, 1, 2)
    inv_d1 = (1.0 / d1).permute(0, 3, 1, 2)
    out = o1 * inv_d1[..., None] * (1.0 + c_norm[..., None]) - o2 * inv_d1[..., None]
    return out.reshape(B, Sq, HQ, D).to(q.dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'Sq', 'Skv', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-Sq{}-Skv{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (2, 1, 1, 2, 2, 64, None),       # single token, empty-ish cache
            (2, 1, 137, 2, 2, 64, None),     # single decode step over a cache
            (2, 1, 500, 2, 8, 128, None),    # GQA decode, D128
            (2, 1, 300, 2, 2, 100, 64),      # windowed decode, non-pow2 D
            (2, 64, 64, 2, 2, 64, None),     # full prefill == training causal
            (3, 200, 200, 2, 8, 64, 32),     # windowed prefill (GQA)
        ]
    ],
)
def test_decode(B: int, Sq: int, Skv: int, H: int, HQ: int, D: int, W, dtype: torch.dtype):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, Sq, HQ, D), dtype=dtype, device=device)
    r = torch.randn((B, Sq, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, Skv, H, D), dtype=dtype, device=device)
    v = torch.randn((B, Skv, H, D), dtype=dtype, device=device)

    ref = _decode_ref(q, r, k, v, scale=D ** -0.5, window_size=W)
    tri = parallax_decode(q, r, k, v, window_size=W)
    assert_close(" o", ref, tri, tol)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (2, 256, 2, 2, 64, None),
            (2, 256, 2, 8, 64, None),     # GQA
            (2, 200, 2, 2, 128, None),    # D128
            (2, 256, 2, 4, 64, 64),       # sliding window
        ]
    ],
)
def test_decode_matches_parallel(B: int, T: int, H: int, HQ: int, D: int, W, dtype: torch.dtype):
    """The decode kernel must reproduce the (reference-verified) training kernel
    for the corresponding positions: decode(last m queries | full KV) == parallel[last m]."""
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    o_full = parallel_parallax(q, r, k, v, window_size=W)
    for m in (1, T // 2, T):
        o_dec = parallax_decode(q[:, T - m:], r[:, T - m:], k, v, window_size=W)
        assert_close(f"o(m={m})", o_full[:, T - m:], o_dec, tol)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (2, 200, 2, 2, 64, None),
            (2, 200, 2, 8, 64, None),     # GQA
            (2, 200, 2, 2, 128, None),    # D128
            (2, 200, 2, 4, 64, 64),       # sliding window
        ]
    ],
)
def test_decode_one_step(B: int, T: int, H: int, HQ: int, D: int, W, dtype: torch.dtype):
    """Single-query decode kernel: must match the training kernel at the last position
    and the prefill-shaped decode kernel."""
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)

    o_train = parallel_parallax(q, r, k, v, window_size=W)[:, -1:]
    o_one = parallax_decode_one_step(q[:, -1:], r[:, -1:], k, v, window_size=W)
    o_tile = parallax_decode(q[:, -1:], r[:, -1:], k, v, window_size=W)
    assert_close("one_step vs train", o_train, o_one, tol)
    assert_close("one_step vs tile ", o_tile, o_one, tol)

    # left-padding: one_step(cache_start=p) must equal training on the unpadded suffix
    p = 23
    cs = torch.full((B,), p, device=device, dtype=torch.int32)
    o_one_p = parallax_decode_one_step(q[:, -1:], r[:, -1:], k, v, window_size=W, cache_start=cs)
    o_ref_p = parallel_parallax(q[:, p:], r[:, p:], k[:, p:], v[:, p:], window_size=W)[:, -1:]
    assert_close("one_step cache_start", o_ref_p, o_one_p, tol)
