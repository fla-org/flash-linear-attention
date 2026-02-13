# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 4, 60, 1, 1, torch.float),
            (2, 1000, 2, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 0.1, 1, torch.float),
            (4, 1024, 3, 3, 128, 1, 10, torch.float),
            (4, 2048, 4, 4, 64, 0.1, 1, torch.float),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 1, 10, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    g = g / gate_logit_normalizer
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=True,
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
            (2, 500, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.float16),
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
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

    tri, tri_ht = chunk_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
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
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 15], torch.float16),
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
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=torch.float).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_gated_delta_rule(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
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


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 8192], torch.float16),
            (4, 60, 0, [0, 15], torch.float16),
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
@torch.inference_mode()
def test_chunk_varlen_prefill(
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
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype).to(device)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype).to(device)
    v = torch.randn((1, T, H, D), dtype=dtype).to(device)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype)).to(device)
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    h0 = torch.randn((N, H, D, D), dtype=dtype).to(device)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_gated_delta_rule(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
