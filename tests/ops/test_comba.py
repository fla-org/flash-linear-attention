# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.comba import chunk_comba, fused_recurrent_comba
from fla.ops.comba.naive import naive_chunk_comba
from fla.ops.comba.utils import chunk_comba_cumsum_scalar_fwd
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


def cumsum_comba_local_fwd_reference(s, reverse=False, chunk_size=128):
    o_0 = torch.zeros_like(s)
    o_1 = torch.zeros_like(s)
    T = s.size(1)
    fn = torch.cumsum
    for i in range(0, T, chunk_size):
        s_chunk = s[:, i:i+chunk_size]
        o_1[:, i:i+chunk_size] = fn(s_chunk.float(), dim=1).to(o_1)
        o_0[:, i:i+chunk_size] = o_1[:, i:i+chunk_size] - s_chunk

    return o_0, o_1


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'chunk_size', 'dtype'),
    [
        pytest.param(*test, id='B{}-T{}-H{}-chunk_size{}-{}'.format(*test))
        for test in [
            (32, 200, 4, 64, torch.float),
            (32, 1000, 4, 64, torch.float),
            (32, 2048, 8, 128, torch.float),
        ]
    ],
)
def test_cumsum_local_scalar_fwd(
    B: int,
    T: int,
    H: int,
    chunk_size: int,
    dtype: torch.dtype,
):
    s = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    ref_0, ref_1 = cumsum_comba_local_fwd_reference(s, chunk_size=chunk_size)
    tri_0, tri_1 = chunk_comba_cumsum_scalar_fwd(s, chunk_size=chunk_size)
    assert_close("local cumsum scalar", ref_0, tri_0, 0.001 if dtype == torch.float else 0.003)
    assert_close("local cumsum scalar", ref_1, tri_1, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float),
            (2, 1024, 4, 60, 1, 1, torch.float),
            (2, 1024, 8, 128, 1, 0.1, torch.float),
            (2, 1024, 8, 128, 0.1, 1, torch.float),
            (2, 1024, 8, 128, 1, 10, torch.float),
            (4, 2048, 8, 64, 0.1, 1, torch.float),
            (2, 1024, 8, 128, 1, 0.1, torch.float16),
            (2, 1024, 8, 128, 1, 10, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    p = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, p, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, p, beta, g, h0))
    ref, ref_ht = naive_chunk_comba(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        p=p.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_comba(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        p=p.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1024, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (2, 1024, 4, 100, 1, 0.1, 0, False, torch.float16),
            (2, 1024, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1024, 4, 128, 0.1, 1, 0.5, False, torch.float16),
            (2, 1024, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, True, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    p = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, p, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, p, beta, g, h0))

    tri, tri_ht = chunk_comba(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        p=F.normalize(p.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else p.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dp, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, p.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = p.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = naive_chunk_comba(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        p=F.normalize(p.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, ref_dk, ref_dv, ref_dp, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, p.grad, beta.grad, g.grad, h0.grad

    assert_close("  o", ref, tri, 0.005)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close(" dq", ref_dq, tri_dq, 0.005)
    assert_close(" dk", ref_dk, tri_dk, 0.008)
    assert_close(" dv", ref_dv, tri_dv, 0.005)
    assert_close(" dp", ref_dp, tri_dp, 0.008)
    assert_close(" dg", ref_dg, tri_dg, 0.02)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.005)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 0, [0, 15], torch.float16),
            (4, 64, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    p = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, p, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, p, beta, g, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_comba(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        p=p.clone(),
        beta=beta.clone(),
        g=g.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_chunk_comba(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            p=p[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
