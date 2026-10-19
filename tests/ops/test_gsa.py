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
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((1, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = None
    if gate_logit_normalizer is not None:
        g = (F.logsigmoid(torch.randn((1, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(N, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(N, H, M, V, device=device).requires_grad_()
    dhkt = torch.randn(N, H, D, M, device=device).requires_grad_()
    dhvt = torch.randn(N, H, M, V, device=device).requires_grad_()

    do = torch.randn_like(v)
    refs, ref_hkts, ref_hfts = [], [], []
    for i in range(N):
        ref_s = s[:, cu_seqlens[i]:cu_seqlens[i+1]]
        ref_g = g[:, cu_seqlens[i]:cu_seqlens[i+1]] if g is not None else None
        if ref_g is None:
            z = ref_s.float().logcumsumexp(1)
            ref_s = (ref_s.float() - z).exp().to(dtype)
            ref_g = torch.cat((z[:, :1], z[:, :-1]), dim=1) - z
        ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
            q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            ref_s,
            ref_g,
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
        cu_seqlens=cu_seqlens,
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
    ('B', 'T', 'H', 'HQ', 'D', 'V', 'M', 'gate_logit_normalizer', 'dtype', 'checkpoint_level'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-V{}-M{}-gate_logit_normalizer{}-{}-checkpoint_level{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 64, 32, 1, torch.float16, 2),
            (2, 1024, 4, 4, 60, 60, 64, 1, torch.float16, 2),
            (2, 1024, 4, 4, 256, 256, 64, 1, torch.float16, 2),
            (2, 1024, 4, 4, 128, 128, 64, 0.1, torch.float, 2),
            (2, 1024, 4, 4, 128, 128, 128, 1, torch.float16, 2),
            (2, 1024, 4, 4, 128, 128, 64, 10, torch.float16, 2),
            (2, 65, 2, 2, 32, 48, 32, None, torch.float32, 2),
            (2, 65, 2, 2, 32, 48, 32, None, torch.float16, 2),
            (2, 65, 2, 2, 32, 48, 32, None, torch.bfloat16, 2),
            (1, 65, 2, 4, 32, 48, 32, 1, torch.float16, 0),
            (1, 65, 2, 4, 32, 48, 32, 1, torch.float16, 1),
            (1, 65, 2, 4, 32, 48, 32, 1, torch.float16, 2),
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
    dhkt = torch.randn(B, H, D, M, device=device).requires_grad_()
    dhvt = torch.randn(B, H, M, V, device=device).requires_grad_()

    do = torch.randn(*q.shape[:3], V, dtype=dtype, device=device)
    reference = naive_recurrent_gsa if HQ != H else fused_recurrent_gsa
    ref, (ref_hkt, ref_hvt) = reference(
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
    assert_close('ds', ref_ds, tri_ds, 0.005 if g is None else 0.008)
    if g is not None:
        assert_close('dg', ref_dg, tri_dg, 0.008)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'M', 'dtype', 'chunk_size', 'checkpoint_level'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-M{}-{}-chunk{}-checkpoint{}".format(*test))
        for chunk_size in [16, 32, 64]
        for checkpoint_level in [0, 1, 2]
        for test in [
            (1, 64, 2, 32, 32, torch.float32, chunk_size, checkpoint_level),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk_with_chunk_size(
    B: int,
    T: int,
    H: int,
    D: int,
    M: int,
    dtype: torch.dtype,
    chunk_size: int,
    checkpoint_level: int,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    s = torch.randn(B, T, H, M, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, M, dtype=torch.float32, device=device))
    hk0 = torch.randn(B, H, D, M, dtype=torch.float32, device=device)
    hv0 = torch.randn(B, H, M, D, dtype=torch.float32, device=device)
    do = torch.randn_like(v)
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)

    def run_ref():
        q_, k_, v_, s_, g_, hk0_, hv0_ = (
            x.detach().clone().requires_grad_(True)
            for x in (q, k, v, s, g, hk0, hv0)
        )
        o, (hkt, hvt) = fused_recurrent_gsa(
            q=q_,
            k=k_,
            v=v_,
            s=s_,
            g=g_,
            scale=D**-0.5,
            initial_state=(hk0_, hv0_),
            output_final_state=True,
        )
        ((o * do).sum() + (hkt * dhkt).sum() + (hvt * dhvt).sum()).backward()
        return o, hkt, hvt, q_.grad, k_.grad, v_.grad, s_.grad, g_.grad, hk0_.grad, hv0_.grad

    def run_tri(chunk_size: int):
        q_, k_, v_, s_, g_, hk0_, hv0_ = (
            x.detach().clone().requires_grad_(True)
            for x in (q, k, v, s, g, hk0, hv0)
        )
        o, (hkt, hvt) = chunk_gsa(
            q=q_,
            k=k_,
            v=v_,
            s=s_,
            g=g_,
            scale=D**-0.5,
            initial_state=(hk0_, hv0_),
            output_final_state=True,
            chunk_size=chunk_size,
            checkpoint_level=checkpoint_level,
        )
        ((o * do).sum() + (hkt * dhkt).sum() + (hvt * dhvt).sum()).backward()
        return o, hkt, hvt, q_.grad, k_.grad, v_.grad, s_.grad, g_.grad, hk0_.grad, hv0_.grad

    ref_o, ref_hkt, ref_hvt, ref_dq, ref_dk, ref_dv, ref_ds, ref_dg, ref_dhk0, ref_dhv0 = run_ref()
    tri_o, tri_hkt, tri_hvt, tri_dq, tri_dk, tri_dv, tri_ds, tri_dg, tri_dhk0, tri_dhv0 = run_tri(chunk_size)

    assert_close(f'o@{chunk_size}', ref_o, tri_o, 0.005)
    assert_close(f'hkt@{chunk_size}', ref_hkt, tri_hkt, 0.005)
    assert_close(f'hvt@{chunk_size}', ref_hvt, tri_hvt, 0.005)
    assert_close(f'dq@{chunk_size}', ref_dq, tri_dq, 0.005)
    assert_close(f'dk@{chunk_size}', ref_dk, tri_dk, 0.005)
    assert_close(f'dv@{chunk_size}', ref_dv, tri_dv, 0.005)
    assert_close(f'ds@{chunk_size}', ref_ds, tri_ds, 0.005)
    assert_close(f'dg@{chunk_size}', ref_dg, tri_dg, 0.005)
    assert_close(f'dhk0@{chunk_size}', ref_dhk0, tri_dhk0, 0.005)
    assert_close(f'dhv0@{chunk_size}', ref_dhv0, tri_dhv0, 0.005)


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
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    s = torch.randn((1, T, H, M), dtype=dtype, device=device).requires_grad_()
    g = None
    if gate_logit_normalizer is not None:
        g = (F.logsigmoid(torch.randn((1, T, H, M), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    hk0 = torch.randn(N, H, D, M, device=device).requires_grad_()
    hv0 = torch.randn(N, H, M, V, device=device).requires_grad_()
    dhkt = torch.randn(N, H, D, M, device=device).requires_grad_()
    dhvt = torch.randn(N, H, M, V, device=device).requires_grad_()

    do = torch.randn(*q.shape[:3], V, dtype=dtype, device=device)

    if HQ != H:
        refs, ref_hkts, ref_hvts = [], [], []
        for i in range(N):
            ref, (ref_hkt, ref_hvt) = naive_recurrent_gsa(
                q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
                k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
                v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
                s=s[:, cu_seqlens[i]:cu_seqlens[i+1]],
                g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
                initial_state=(hk0[i:i+1], hv0[i:i+1]),
                output_final_state=True,
            )
            refs.append(ref)
            ref_hkts.append(ref_hkt)
            ref_hvts.append(ref_hvt)
        ref = torch.cat(refs, 1)
        ref_hkt, ref_hvt = torch.cat(ref_hkts, 0), torch.cat(ref_hvts, 0)
    else:
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
        cu_seqlens=cu_seqlens,
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
    ('B', 'T', 'HQ', 'H', 'D', 'M', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-HQ{}-H{}-D{}-M{}-{}".format(*test))
        for test in [
            (2, 63, 2, 1, 64, 32, torch.float),
            (2, 17, 4, 2, 60, 48, torch.float),
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


@pytest.mark.parametrize(
    ('has_initial_state', 'output_final_state'),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_inference_state_contract(has_initial_state: bool, output_final_state: bool):
    torch.manual_seed(42)
    B, T, H, D, M = 2, 1, 2, 16, 8
    q = torch.randn(B, T, H, D, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    s = torch.randn(B, T, H, M, device=device)
    g = F.logsigmoid(torch.randn_like(s))
    initial_state = None
    initial_state_before = None
    if has_initial_state:
        initial_state = (
            torch.randn(B, H, D, M, device=device),
            torch.randn(B, H, M, D, device=device),
        )
        initial_state_before = tuple(state.clone() for state in initial_state)

    ref, ref_state = naive_recurrent_gsa(
        q=q,
        k=k,
        v=v,
        s=s,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    with torch.no_grad():
        tri, tri_state = fused_recurrent_gsa(
            q=q,
            k=k,
            v=v,
            s=s,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    assert_close('o', ref, tri, 0.005)
    if output_final_state:
        assert_close('hkt', ref_state[0], tri_state[0], 0.005)
        assert_close('hvt', ref_state[1], tri_state[1], 0.005)
    else:
        assert tri_state == [None, None]

    if has_initial_state:
        torch.testing.assert_close(initial_state[0], initial_state_before[0], rtol=0, atol=0)
        torch.testing.assert_close(initial_state[1], initial_state_before[1], rtol=0, atol=0)
        if output_final_state:
            assert tri_state[0].data_ptr() != initial_state[0].data_ptr()
            assert tri_state[1].data_ptr() != initial_state[1].data_ptr()
