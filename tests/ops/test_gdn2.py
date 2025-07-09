# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2
from fla.ops.gdn2.naive import naive_chunk_gdn2, naive_recurrent_gdn2
from fla.utils import assert_close, device, is_intel_alchemist


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test)
        )
        for test in [
            (1, 64, 1, 64, 1, 1, torch.float),
            (2, 512, 3, 60, 1, 1, torch.float),
            (4, 1024, 4, 128, 0.1, 1, torch.float),
        ]
    ]
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = naive_chunk_gdn2(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-use_qk_l2norm_in_kernel{}-{}".format(*test)
        )
        for test in [
            (1, 64, 1, 64, 1, 1, False, torch.float),
            (2, 512, 3, 60, 1, 1, False, torch.float),
            (3, 1000, 4, 100, 0.1, 1, True, torch.float),
            (4, 1024, 4, 128, 0.1, 1, False, torch.float),
        ]
    ]
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = fused_recurrent_gdn2(
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
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-use_qk_l2norm_in_kernel{}-{}".format(*test)
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, torch.float16),
            (2, 500, 3, 60, 1, 1, 0, False, torch.float16),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, torch.float16),
            (3, 1024, 4, 100, 1, 0.1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, False, torch.float16),
            (4, 1024, 4, 128, 0.1, 1, 0, True, torch.float16),
            (2, 1500, 4, 128, 0.1, 10, 0, False, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, 0, False, torch.float16)
        ]
    ]
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
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_gdn2(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_db, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    tri, tri_ht = chunk_gdn2(
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
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_db, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_db, tri_db, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
