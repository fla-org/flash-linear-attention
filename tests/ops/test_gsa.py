
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from fla.ops.gsa.naive import naive_recurrent_gsa
from fla.utils import assert_close, check_shared_mem, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'M', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-M{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 32, 1, torch.float),
            (2, 1024, 4, 60, 64, 1, torch.float),
            (2, 1024, 8, 128, 64, 0.1, torch.float),
            (2, 1024, 8, 128, 32, 1, torch.float),
            (2, 1024, 8, 128, 64, 1, torch.float),
            (2, 1024, 8, 128, 64, 10, torch.float),
            (4, 2048, 8, 64, 64, 1, torch.float),
            (2, 1024, 8, 128, 64, 0.1, torch.float16),
            (2, 1024, 8, 128, 64, 10, torch.float16),
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
    M: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((B, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = (F.logsigmoid(torch.randn((B, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(B, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(B, H, M, D, device=device).requires_grad_()
    do = torch.randn_like(v)
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)

    ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(q, k, v, s, g, initial_state=(hk0, hv0), output_final_state=True)
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
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
    tri_dg, s.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'M', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-M{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 64, [0, 15], torch.float),
            (4, 64, 64, [0, 256, 500, 1000], torch.float),
            (4, 100, 64, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 64, 64, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (4, 128, 64, [0, 200, 512, 1200, 2048], torch.float16),
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
    M: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((1, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H, M), dtype=dtype, device=device)).requires_grad_()
    hk0 = torch.randn(N, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(N, H, M, D, device=device).requires_grad_()
    dhkt = torch.randn(N, H, D, M, device=device).requires_grad_()
    dhvt = torch.randn(N, H, M, D, device=device).requires_grad_()

    do = torch.randn_like(v)
    refs, ref_hkts, ref_hfts = [], [], []
    for i in range(N):
        ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
            q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            s[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=(hk0[i:i+1], hv0[i:i+1]),
            output_final_state=True,
        )
        refs.append(ref)
        ref_hkts.append(ref_hkt)
        ref_hfts.append(ref_hvt)
    ref = torch.cat(refs, 1)
    ref_hkt = torch.cat(ref_hkts, 0)
    ref_hvt = torch.cat(ref_hfts, 0)
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
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
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    tri_dg, s.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'M', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-M{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 32, 1, torch.float16),
            (2, 1024, 4, 60, 64, 1, torch.float16),
            (2, 1024, 4, 256, 64, 1, torch.float16),
            (2, 1024, 4, 128, 64, 0.1, torch.float),
            (2, 1024, 4, 128, 128, 1, torch.float16),
            (2, 1024, 4, 128, 64, 10, torch.float16),
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
    D: int,
    M: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    if (D > 64 or M > 64) and check_shared_mem('hopper') is False:
        pytest.skip(reason='Current CI do not support this config')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((B, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = (F.logsigmoid(torch.randn((B, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(B, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(B, H, M, D, device=device).requires_grad_()
    dhkt = torch.randn(B, H, D, M, device=device).requires_grad_()
    dhvt = torch.randn(B, H, M, D, device=device).requires_grad_()

    do = torch.randn_like(v)
    ref, (ref_hkt, ref_hvt) = fused_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=D**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True)
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
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
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    tri_dg, s.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.008)
    assert_close('dg', ref_dg, tri_dg, 0.008)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'M', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-M{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 64, [0, 15], torch.float16),
            (4, 64, 64, [0, 256, 500, 1000], torch.float16),
            (4, 100, 64, [0, 15, 100, 300, 1200, 2000], torch.float16),
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
def test_chunk_varlen(
    H: int,
    D: int,
    M: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if (D > 64 or M > 64) and check_shared_mem('hopper') is False:
        pytest.skip(reason='Current CI do not support this config')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((1, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H, M), dtype=dtype, device=device)).requires_grad_()
    hk0 = torch.randn(N, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(N, H, M, D, device=device).requires_grad_()
    dhkt = torch.randn(N, H, D, M, device=device).requires_grad_()
    dhvt = torch.randn(N, H, M, D, device=device).requires_grad_()

    do = torch.randn_like(v)

    ref, (ref_hkt, ref_hvt) = fused_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        scale=D**-0.5,
        initial_state=(hk0, hv0),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
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
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_ds, s.grad = s.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dhk0, hk0.grad = hk0.grad.clone(), None
    tri_dhv0, hv0.grad = hv0.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('hkt', ref_hkt, tri_hkt, 0.005)
    assert_close('hvt', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('ds', ref_ds, tri_ds, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'HQ', 'H', 'D', 'M', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-HQ{}-H{}-D{}-M{}-{}".format(*test))
        for test in [
            (2, 63, 2, 1, 64, 32, torch.float),
            (2, 200, 8, 2, 64, 64, torch.float),
            (2, 256, 16, 4, 128, 64, torch.float),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_inference(
    B: int,
    T: int,
    HQ: int,
    H: int,
    D: int,
    M: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    s = torch.randn((B, T, H, M), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, H, M), dtype=dtype, device=device))
    h0 = (torch.randn(B, H, D, M, dtype=dtype, device=device),
          torch.randn(B, H, M, D, dtype=dtype, device=device))

    ref, _ = naive_recurrent_gsa(q, k, v, s, g, initial_state=h0)
    tri = torch.empty_like(ref)
    for i in range(T):
        o, ht = fused_recurrent_gsa(
            q[:, i:i+1],
            k[:, i:i+1],
            v[:, i:i+1],
            s[:, i:i+1],
            g[:, i:i+1],
            initial_state=h0,
            output_final_state=True,
        )
        tri[:, i] = o.squeeze(1)
        assert_close(f'o{i}', ref[:, i], tri[:, i], 0.005)
        h0 = ht
