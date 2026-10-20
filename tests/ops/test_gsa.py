# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from fla.ops.gsa.naive import naive_recurrent_gsa
from fla.utils import assert_close, check_shared_mem, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'V', 'M', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-V{}-M{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 64, 32, 1, torch.float),
            (1, 1, 2, 32, 32, 32, 8, torch.float16),
            (1, 65, 2, 60, 60, 96, 8, torch.float),
            (1, 65, 2, 60, 60, 32, 8, torch.bfloat16),
            (2, 1024, 4, 60, 60, 64, 1, torch.float),
            (2, 1024, 8, 128, 128, 64, 0.1, torch.float),
            (2, 1024, 8, 128, 128, 32, 1, torch.float),
            (2, 1024, 8, 128, 128, 64, 1, torch.float),
            (2, 1024, 8, 128, 128, 64, 10, torch.float),
            (4, 2048, 8, 64, 64, 64, 1, torch.float),
            (2, 1024, 8, 128, 128, 64, 0.1, torch.float16),
            (2, 1024, 8, 128, 128, 64, 10, torch.float16),
            (2, 65, 2, 32, 48, 32, None, torch.float32),
            (2, 65, 2, 32, 48, 32, None, torch.float16),
            (2, 65, 2, 32, 48, 32, None, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    V: int,
    M: int,
    gate_logit_normalizer: float | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((B, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = None
    if gate_logit_normalizer is not None:
        g = (F.logsigmoid(torch.randn((B, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(B, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(B, H, M, V, device=device).requires_grad_()
    do = torch.randn_like(v)
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)

    ref_s, ref_g = s, g
    if g is None:
        z = s.float().logcumsumexp(1)
        ref_s = (s.float() - z).exp().to(dtype)
        ref_g = torch.cat((z[:, :1], z[:, :-1]), dim=1) - z
    ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=ref_s,
        g=ref_g,
        initial_state=(hk0, hv0),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    if g is not None:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dhk0, hk0.grad = hk0.grad.clone(), None
    ref_dhv0, hv0.grad = hv0.grad.clone(), None

    tri, (tri_hkt, tri_hvt) = fused_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        initial_state=(hk0, hv0),
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    if g is not None:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.005)
    if g is not None:
        assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'V', 'M', 'cu_seqlens', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-V{}-M{}-cu_seqlens{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (4, 64, 64, 64, [0, 15], 1, torch.float),
            (2, 60, 60, 32, [0, 1, 17, 65], 1, torch.bfloat16),
            (4, 64, 64, 64, [0, 256, 500, 1000], 1, torch.float),
            (4, 100, 100, 64, [0, 15, 100, 300, 1200, 2000], 1, torch.float),
            (4, 64, 64, 64, [0, 1, 100, 300, 1200, 2048], 1, torch.float16),
            (4, 128, 128, 64, [0, 200, 512, 1200, 2048], 1, torch.float16),
            (2, 32, 48, 32, [0, 1, 17, 65], None, torch.float32),
            (2, 32, 48, 32, [0, 1, 17, 65], None, torch.float16),
            (2, 32, 48, 32, [0, 1, 17, 65], None, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent_varlen(
    H: int,
    D: int,
    V: int,
    M: int,
    cu_seqlens: list[int],
    gate_logit_normalizer: float | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((1, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = None
    if gate_logit_normalizer is not None:
        g = (F.logsigmoid(torch.randn((1, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(N, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(N, H, M, V, device=device).requires_grad_()
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)

    do = torch.randn_like(v)
    refs, ref_hkts, ref_hvts = [], [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        ref_s = s[:, bos:eos]
        ref_g = g[:, bos:eos] if g is not None else None
        if ref_g is None:
            z = ref_s.float().logcumsumexp(1)
            ref_s = (ref_s.float() - z).exp().to(dtype)
            ref_g = torch.cat((z[:, :1], z[:, :-1]), dim=1) - z
        ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            s=ref_s,
            g=ref_g,
            initial_state=(hk0[i:i+1], hv0[i:i+1]),
            output_final_state=True,
        )
        refs.append(ref)
        ref_hkts.append(ref_hkt)
        ref_hvts.append(ref_hvt)
    ref = torch.cat(refs, dim=1)
    ref_hkt = torch.cat(ref_hkts, dim=0)
    ref_hvt = torch.cat(ref_hvts, dim=0)
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    if g is not None:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dhk0, hk0.grad = hk0.grad.clone(), None
    ref_dhv0, hv0.grad = hv0.grad.clone(), None

    tri, (tri_hkt, tri_hvt) = fused_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        initial_state=(hk0, hv0),
        output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    if g is not None:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.005)
    if g is not None:
        assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'V', 'M', 'gate_logit_normalizer', 'dtype', 'checkpoint_level', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-V{}-M{}-gate{}-{}-checkpoint{}-chunk{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 64, 32, 1, torch.float16, 2, 64),
            (2, 1024, 4, 4, 60, 60, 64, 1, torch.float16, 2, 64),
            (2, 1024, 4, 4, 256, 256, 64, 1, torch.float16, 2, 64),
            (2, 1024, 4, 4, 128, 128, 64, 0.1, torch.float, 2, 64),
            (2, 1024, 4, 4, 128, 128, 128, 1, torch.float16, 2, 64),
            (2, 1024, 4, 4, 128, 128, 64, 10, torch.float16, 2, 64),
            (2, 65, 2, 2, 32, 48, 32, None, torch.float32, 2, 64),
            (2, 65, 2, 2, 32, 48, 32, None, torch.float16, 2, 64),
            (2, 65, 2, 2, 32, 48, 32, None, torch.bfloat16, 2, 64),
            (1, 65, 2, 4, 32, 48, 32, 1, torch.float16, 0, 64),
            (1, 65, 2, 4, 32, 48, 32, 1, torch.float16, 1, 64),
            (1, 65, 2, 4, 32, 48, 32, 1, torch.float16, 2, 64),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 0, 16),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 1, 16),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 2, 16),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 0, 32),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 1, 32),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 2, 32),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 0, 64),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 1, 64),
            (1, 64, 2, 2, 32, 32, 32, 1, torch.float32, 2, 64),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    V: int,
    M: int,
    gate_logit_normalizer: float | None,
    dtype: torch.dtype,
    checkpoint_level: int,
    chunk_size: int,
):
    if (D > 64 or M > 64) and check_shared_mem('hopper') is False:
        pytest.skip(reason='Current CI do not support this config')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((B, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = None
    if gate_logit_normalizer is not None:
        g = (F.logsigmoid(torch.randn((B, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(B, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(B, H, M, V, device=device).requires_grad_()
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)

    do = torch.randn(*q.shape[:3], V, dtype=dtype, device=device)
    ref_s, ref_g = s, g
    if g is None:
        z = s.float().logcumsumexp(1)
        ref_s = (s.float() - z).exp().to(dtype)
        ref_g = torch.cat((z[:, :1], z[:, :-1]), dim=1) - z
    ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=ref_s,
        g=ref_g,
        scale=D**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    if g is not None:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dhk0, hk0.grad = hk0.grad.clone(), None
    ref_dhv0, hv0.grad = hv0.grad.clone(), None

    tri, (tri_hkt, tri_hvt) = chunk_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=D**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
        checkpoint_level=checkpoint_level,
        chunk_size=chunk_size,
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    if g is not None:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.005 if g is None or dtype == torch.float32 else 0.008)
    if g is not None:
        assert_close('dg', ref_dg, tri_dg, 0.005 if dtype == torch.float32 else 0.008)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'V', 'M', 'cu_seqlens', 'gate_logit_normalizer', 'dtype', 'checkpoint_level'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-V{}-M{}-cu_seqlens{}-gate_logit_normalizer{}-{}-checkpoint_level{}".format(*test))
        for test in [
            (4, 4, 64, 64, 64, [0, 15], 1, torch.float16, 2),
            (4, 4, 64, 64, 64, [0, 256, 500, 1000], 1, torch.float16, 2),
            (4, 4, 100, 100, 64, [0, 15, 100, 300, 1200, 2000], 1, torch.float16, 2),
            (2, 2, 32, 48, 32, [0, 1, 17, 65], None, torch.float32, 2),
            (2, 2, 32, 48, 32, [0, 1, 17, 65], None, torch.float16, 2),
            (2, 2, 32, 48, 32, [0, 1, 17, 65], None, torch.bfloat16, 2),
            (2, 4, 32, 48, 32, [0, 1, 17, 65], 1, torch.float16, 0),
            (2, 4, 32, 48, 32, [0, 1, 17, 65], 1, torch.float16, 1),
            (2, 4, 32, 48, 32, [0, 1, 17, 65], 1, torch.float16, 2),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
@pytest.mark.smoke
def test_chunk_varlen(
    H: int,
    HQ: int,
    D: int,
    V: int,
    M: int,
    cu_seqlens: list[int],
    gate_logit_normalizer: float | None,
    dtype: torch.dtype,
    checkpoint_level: int,
):
    if (D > 64 or M > 64) and check_shared_mem('hopper') is False:
        pytest.skip(reason='Current CI do not support this config')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((1, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = None
    if gate_logit_normalizer is not None:
        g = (F.logsigmoid(torch.randn((1, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(N, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(N, H, M, V, device=device).requires_grad_()
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)

    do = torch.randn(*q.shape[:3], V, dtype=dtype, device=device)

    refs, ref_hkts, ref_hvts = [], [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        ref_s = s[:, bos:eos]
        ref_g = g[:, bos:eos] if g is not None else None
        if ref_g is None:
            z = ref_s.float().logcumsumexp(1)
            ref_s = (ref_s.float() - z).exp().to(dtype)
            ref_g = torch.cat((z[:, :1], z[:, :-1]), dim=1) - z
        ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            s=ref_s,
            g=ref_g,
            initial_state=(hk0[i:i+1], hv0[i:i+1]),
            output_final_state=True,
        )
        refs.append(ref)
        ref_hkts.append(ref_hkt)
        ref_hvts.append(ref_hvt)
    ref = torch.cat(refs, dim=1)
    ref_hkt = torch.cat(ref_hkts, dim=0)
    ref_hvt = torch.cat(ref_hvts, dim=0)
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    if g is not None:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dhk0, hk0.grad = hk0.grad.clone(), None
    ref_dhv0, hv0.grad = hv0.grad.clone(), None

    tri, (tri_hkt, tri_hvt) = chunk_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=D**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
        checkpoint_level=checkpoint_level,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    if g is not None:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005 if g is None else 0.004)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.008 if HQ != H else 0.005)
    if g is not None:
        assert_close('dg', ref_dg, tri_dg, 0.008 if HQ != H else 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'M', 'dtype', 'has_initial_state', 'output_final_state'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-M{}-{}-initial{}-final{}".format(*test))
        for test in [
            (2, 63, 1, 2, 64, 32, torch.float, True, True),
            (2, 17, 2, 4, 60, 48, torch.float, True, True),
            (2, 200, 2, 8, 64, 64, torch.float, True, True),
            (2, 256, 4, 16, 128, 64, torch.float, True, True),
            (2, 1, 2, 2, 16, 8, torch.float, False, False),
            (2, 1, 2, 2, 16, 8, torch.float, False, True),
            (2, 1, 2, 2, 16, 8, torch.float, True, False),
            (2, 1, 2, 2, 16, 8, torch.float, True, True),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
@torch.no_grad()
def test_fused_recurrent_inference(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    M: int,
    dtype: torch.dtype,
    has_initial_state: bool,
    output_final_state: bool,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    s = torch.randn((B, T, H, M), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, H, M), dtype=dtype, device=device))
    h0 = None
    if has_initial_state:
        hk0 = torch.randn(B, H, D, M, dtype=dtype, device=device)
        hv0 = torch.randn(B, H, M, D, dtype=dtype, device=device)
        h0 = (hk0, hv0)
        hk0_copy, hv0_copy = hk0.clone(), hv0.clone()

    ref, ref_state = naive_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        initial_state=h0,
        output_final_state=output_final_state,
    )
    ht = h0
    for i in range(T):
        tri, ht = fused_recurrent_gsa(
            q=q[:, i:i+1],
            k=k[:, i:i+1],
            v=v[:, i:i+1],
            s=s[:, i:i+1],
            g=g[:, i:i+1],
            initial_state=ht,
            output_final_state=output_final_state,
        )
        assert_close('o', ref[:, i:i+1], tri, 0.005)

    if output_final_state:
        assert_close('hkt', ref_state[0], ht[0], 0.005)
        assert_close('hvt', ref_state[1], ht[1], 0.005)
    else:
        assert ht == [None, None]
    if has_initial_state:
        torch.testing.assert_close(hk0, hk0_copy, rtol=0, atol=0)
        torch.testing.assert_close(hv0, hv0_copy, rtol=0, atol=0)
        if output_final_state:
            assert ht[0].data_ptr() != hk0.data_ptr()
            assert ht[1].data_ptr() != hv0.data_ptr()
