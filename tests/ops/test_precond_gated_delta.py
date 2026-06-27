# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Tests for preconditioned gated delta rule kernel

import os

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from fla.ops.precond_gated_delta_rule import (
    chunk_precond_gated_delta_rule,
    fused_recurrent_precond_gated_delta_rule,
)
from fla.ops.precond_gated_delta_rule.naive import naive_recurrent_precond_gated_delta_rule
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HV{}-D{}-gln{}-{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1, torch.float),
            (2, 500, 4, 4, 60, 1, torch.float),
            (2, 1000, 2, 8, 128, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 1, torch.float),
            (4, 1024, 3, 3, 128, 10, torch.float),
            (4, 2048, 4, 4, 64, 1, torch.float),
            (2, 1024, 4, 4, 128, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 10, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = 1.5

    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta_atk = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    beta = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    g_atk = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32)) / gate_logit_normalizer
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32)) / gate_logit_normalizer
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    A0 = torch.zeros(B, HV, D, dtype=torch.float32)

    log_atk_scale = torch.rand(HV, dtype=torch.float32) - 1.0

    q, k, v, beta_atk, beta, g_atk, g, h0, A0, log_atk_scale = map(
        lambda t: t.to(device).requires_grad_(False),
        (q, k, v, beta_atk, beta, g_atk, g, h0, A0, log_atk_scale)
    )

    # Reference implementation: manually L2 normalize + repeat q/k to HV heads
    ref, ref_ht, _ = naive_recurrent_precond_gated_delta_rule(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        v=v.clone(),
        g_atk=g_atk.clone(),
        g=g.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        x=x,
        log_atk_scale=log_atk_scale.clone(),
        initial_state=h0.clone(),
        initial_A_state=A0.clone(),
        output_final_state=True,
    )

    # Kernel implementation: L2 norm done inside kernel
    tri, tri_ht, _ = fused_recurrent_precond_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g_atk=g_atk.clone(),
        g=g.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        initial_A_state=A0.clone(),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        x=x,
        log_atk_scale=log_atk_scale.clone(),
    )

    assert_close('o', ref, tri, 0.002)
    assert_close('ht', ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gln{}-mask_p{}-qk_l2norm{}-{}".format(*test),
        )
        for test in [
            (2, 75, 4, 64, 1, 0.01, 0, False, torch.float16),
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
    x = 1.5

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta_atk = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    beta = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    g_atk = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g_atk = g_atk / gate_logit_normalizer
    g_atk = g_atk * (torch.rand_like(g_atk) > mask_p)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)

    log_atk_scale = torch.rand(H, dtype=torch.float32) - 1.0

    q, k, v, beta_atk, beta, g_atk, g, h0 = map(
        lambda t: t.to(device).requires_grad_(True),
        (q, k, v, beta_atk, beta, g_atk, g, h0)
    )
    log_atk_scale = log_atk_scale.to(device).requires_grad_(True)

    tri, tri_ht_h, _ = chunk_precond_gated_delta_rule(
        q=q.clone() if use_qk_l2norm_in_kernel else F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone() if use_qk_l2norm_in_kernel else F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g_atk=g_atk.clone(),
        g=g.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        log_atk_scale=log_atk_scale.clone(),
        x=x,
    )

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht_h * dht).sum()).backward(retain_graph=True)

    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    tri_dbeta_atk, tri_dbeta = beta_atk.grad, beta.grad
    tri_dg_atk, tri_dg = g_atk.grad, g.grad
    tri_dh0 = h0.grad
    tri_d_log_atk_scale = log_atk_scale.grad

    q.grad = k.grad = v.grad = beta_atk.grad = beta.grad = None
    g_atk.grad = g.grad = h0.grad = None
    log_atk_scale.grad = None

    ref, ref_ht, _ = naive_recurrent_precond_gated_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g_atk=g_atk.clone(),
        g=g.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        scale=scale,
        x=x,
        log_atk_scale=log_atk_scale.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)

    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    ref_dbeta_atk, ref_dbeta = beta_atk.grad, beta.grad
    ref_dg_atk, ref_dg = g_atk.grad, g.grad
    ref_dh0 = h0.grad
    ref_d_log_atk_scale = log_atk_scale.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht_h, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dbeta_atk', ref_dbeta_atk, tri_dbeta_atk, 0.02)
    assert_close('dg_atk', ref_dg_atk, tri_dg_atk, 0.02)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('d_log_atk_scale', ref_d_log_atk_scale, tri_d_log_atk_scale, 0.02)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens'),
    [
        pytest.param(*test, id=f"H{test[0]}-D{test[1]}-mask{test[2]}-seqs{len(test[3])-1}")
        for test in [
            (4, 60, 0, [0, 15]),
            (4, 64, 0, [0, 256, 500, 1000]),
            (4, 64, 0.5, [0, 256, 500, 1000]),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000]),
        ]
    ],
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    dtype = torch.float16
    x = 1.5

    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.rand((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.rand((1, T, H, D), dtype=dtype)
    beta_atk = torch.rand(1, T, H, dtype=dtype).sigmoid()
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    g_atk = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32)) * 0.5
    g_atk = g_atk * (torch.rand_like(g_atk) > mask_p)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32)) * 0.5
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.zeros((N, H, D, D), dtype=torch.float32)

    log_atk_scale = torch.rand(H, dtype=torch.float32) - 1.0

    q, k, v, beta_atk, beta, g_atk, g, h0 = map(
        lambda t: t.to(device).requires_grad_(True),
        (q, k, v, beta_atk, beta, g_atk, g, h0)
    )
    log_atk_scale = log_atk_scale.to(device).requires_grad_(True)

    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    # Kernel forward
    tri, tri_ht_h, _ = chunk_precond_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g_atk=g_atk.clone(),
        g=g.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        log_atk_scale=log_atk_scale.clone(),
        x=x,
    )

    # Backward
    ((tri * do).sum() + (tri_ht_h * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    tri_dbeta_atk, tri_dbeta = beta_atk.grad, beta.grad
    tri_dg_atk, tri_dg = g_atk.grad, g.grad
    tri_dh0 = h0.grad
    tri_d_log_atk_scale = log_atk_scale.grad

    # Clear gradients
    q.grad = k.grad = v.grad = beta_atk.grad = beta.grad = None
    g_atk.grad = g.grad = h0.grad = None
    log_atk_scale.grad = None

    # Reference: process each segment separately
    ref_list = []
    ref_ht_list = []
    for i in range(N):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        ref_i, ref_ht_i, _ = naive_recurrent_precond_gated_delta_rule(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g_atk=g_atk[:, start:end],
            g=g[:, start:end],
            beta_atk=beta_atk[:, start:end],
            beta=beta[:, start:end],
            x=x,
            log_atk_scale=log_atk_scale.clone(),
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_list.append(ref_i)
        ref_ht_list.append(ref_ht_i)

    ref = torch.cat(ref_list, 1)
    ref_ht = torch.cat(ref_ht_list, 0)

    # Reference backward
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    ref_dbeta_atk, ref_dbeta = beta_atk.grad, beta.grad
    ref_dg_atk, ref_dg = g_atk.grad, g.grad
    ref_dh0 = h0.grad
    ref_d_log_atk_scale = log_atk_scale.grad

    # Assertions
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht_h, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('dbeta_atk', ref_dbeta_atk, tri_dbeta_atk, 0.015)
    assert_close('dg_atk', ref_dg_atk, tri_dg_atk, 0.015)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    assert_close('d_log_atk_scale', ref_d_log_atk_scale, tri_d_log_atk_scale, 0.02)


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [pytest.param(*test, id=f"H{test[0]}-D{test[1]}-mask{test[2]}-seqs{len(test[3]) - 1}-{test[4]}")
     for test in [
         (4, 60, 0, [0, 8192], torch.float16),
         (4, 60, 0, [0, 15], torch.float16),
         (4, 64, 0, [0, 256, 500, 1000], torch.float16),
         (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
         (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
    ]],
)
@torch.inference_mode()
def test_chunk_varlen_prefill(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    x = 1.5

    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype).to(device)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype).to(device)
    v = torch.randn((1, T, H, D), dtype=dtype).to(device)
    beta_atk = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid().to(device)
    g_atk = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32)).to(device) * 0.5
    g_atk = g_atk * (torch.rand_like(g_atk) > mask_p)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32)).to(device) * 0.5
    g = g * (torch.rand_like(g) > mask_p)
    h0 = torch.randn((N, H, D, D), dtype=torch.float32).to(device)
    log_atk_scale = (torch.rand(H, dtype=torch.float32) - 1.0).to(device)

    tri, tri_ht, _ = chunk_precond_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g_atk=g_atk.clone(),
        g=g.clone(),
        beta_atk=beta_atk.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        log_atk_scale=log_atk_scale.clone(),
        x=x,
    )

    ref_list = []
    ref_ht_list = []
    for i in range(N):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        ref_i, ref_ht_i, _ = naive_recurrent_precond_gated_delta_rule(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g_atk=g_atk[:, start:end],
            g=g[:, start:end],
            beta_atk=beta_atk[:, start:end],
            beta=beta[:, start:end],
            x=x,
            log_atk_scale=log_atk_scale.clone(),
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_list.append(ref_i)
        ref_ht_list.append(ref_ht_i)

    ref = torch.cat(ref_list, 1)
    ref_ht = torch.cat(ref_ht_list, 0)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gln{}-{}".format(*test))
     for test in [
         (1, 63, 1, 64, 1, 1, torch.float16),
         (2, 500, 3, 60, 1, 1, torch.float16),
         (3, 1024, 4, 128, 0.1, 1, torch.float16),
         (4, 2048, 8, 64, 0.1, 1, torch.float16),
    ]],
)
def test_chunk_transpose_state(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = 1.5

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta_atk = torch.rand(B, T, H, dtype=dtype).sigmoid()
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g_atk = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    log_atk_scale = torch.rand(H, dtype=torch.float32) - 1.0
    # Non-zero initial state to exercise the transpose load path
    h0_kv = torch.randn(B, H, D, D, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()

    q, k, v, beta_atk, beta, g_atk, g, log_atk_scale, h0_kv, h0_vk = map(
        lambda t: t.to(device).requires_grad_(True),
        (q, k, v, beta_atk, beta, g_atk, g, log_atk_scale, h0_kv, h0_vk)
    )

    do = torch.randn_like(v)
    dht_vk = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    dht_kv = dht_vk.transpose(-1, -2).contiguous()

    # transposed layout
    tri, tri_ht, _ = chunk_precond_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g_atk=g_atk.clone(), g=g.clone(),
        beta_atk=beta_atk.clone(), beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        log_atk_scale=log_atk_scale.clone(),
        x=x,
        transpose_state_layout=True,
    )
    ((tri * do).sum() + (tri_ht * dht_vk).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    tri_dbeta_atk, tri_dbeta = beta_atk.grad, beta.grad
    tri_dg_atk, tri_dg = g_atk.grad, g.grad
    tri_d_log_atk_scale = log_atk_scale.grad
    tri_dh0 = h0_vk.grad
    q.grad = k.grad = v.grad = beta_atk.grad = beta.grad = None
    g_atk.grad = g.grad = log_atk_scale.grad = h0_vk.grad = None

    # standard layout (reference)
    ref, ref_ht, _ = chunk_precond_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g_atk=g_atk.clone(), g=g.clone(),
        beta_atk=beta_atk.clone(), beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        log_atk_scale=log_atk_scale.clone(),
        x=x,
        transpose_state_layout=False,
    )
    ((ref * do).sum() + (ref_ht * dht_kv).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad
    ref_dbeta_atk, ref_dbeta = beta_atk.grad, beta.grad
    ref_dg_atk, ref_dg = g_atk.grad, g.grad
    ref_d_log_atk_scale = log_atk_scale.grad
    ref_dh0 = h0_kv.grad

    # Tolerance is 1e-3 rather than 1e-4 (upstream GDN uses 1e-4). The transposed and
    # non-transposed code paths exercise different tile orderings in the chunk backward,
    # and fp16 accumulation in the ATK-extended reductions amplifies tiny per-tile rounding
    # differences that upstream GDN's simpler backward avoids. Observed worst ratios under
    # 1e-4 are ~6.6e-4 (dh0) and ~2e-4 (dg); 1e-3 gives headroom.
    assert_close('o', ref, tri, 1e-3)
    assert_close('ht', ref_ht, tri_ht.transpose(-1, -2), 1e-3)
    assert_close('dq', ref_dq, tri_dq, 1e-3)
    assert_close('dk', ref_dk, tri_dk, 1e-3)
    assert_close('dv', ref_dv, tri_dv, 1e-3)
    assert_close('dbeta_atk', ref_dbeta_atk, tri_dbeta_atk, 1e-3)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 1e-3)
    assert_close('dg_atk', ref_dg_atk, tri_dg_atk, 1e-3)
    assert_close('dg', ref_dg, tri_dg, 1e-3)
    assert_close('d_log_atk_scale', ref_d_log_atk_scale, tri_d_log_atk_scale, 1e-3)
    assert_close('dh0', ref_dh0, tri_dh0.transpose(-1, -2), 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'gate_logit_normalizer', 'dtype'),
    [pytest.param(*test, id="B{}-T{}-H{}-D{}-gln{}-{}".format(*test))
     for test in [
         (1, 63, 1, 64, 1, torch.float32),
         (2, 500, 4, 60, 1, torch.float32),
         (2, 1000, 2, 128, 0.1, torch.float32),
         (3, 1024, 2, 128, 1, torch.float32),
    ]],
)
def test_fused_recurrent_transpose_state(
    B: int,
    T: int,
    H: int,
    D: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = 1.5

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta_atk = torch.rand(B, T, H, dtype=dtype).sigmoid()
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    g_atk = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    log_atk_scale = torch.rand(H, dtype=torch.float32) - 1.0
    h0_kv = torch.randn(B, H, D, D, dtype=torch.float32)
    h0_vk = h0_kv.transpose(-1, -2).contiguous()
    A0 = torch.zeros(B, H, D, dtype=torch.float32)

    q, k, v, beta_atk, beta, g_atk, g, log_atk_scale, h0_kv, h0_vk, A0 = map(
        lambda t: t.to(device).requires_grad_(False),
        (q, k, v, beta_atk, beta, g_atk, g, log_atk_scale, h0_kv, h0_vk, A0)
    )

    ref, ref_ht, _ = fused_recurrent_precond_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g_atk=g_atk.clone(), g=g.clone(),
        beta_atk=beta_atk.clone(), beta=beta.clone(),
        initial_state=h0_kv.clone(),
        initial_A_state=A0.clone(),
        output_final_state=True,
        x=x,
        log_atk_scale=log_atk_scale.clone(),
        transpose_state_layout=False,
    )

    tri, tri_ht, _ = fused_recurrent_precond_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(),
        g_atk=g_atk.clone(), g=g.clone(),
        beta_atk=beta_atk.clone(), beta=beta.clone(),
        initial_state=h0_vk.clone(),
        initial_A_state=A0.clone(),
        output_final_state=True,
        x=x,
        log_atk_scale=log_atk_scale.clone(),
        transpose_state_layout=True,
    )

    assert_close('o', ref, tri, 1e-4)
    assert_close('ht', ref_ht, tri_ht.transpose(-1, -2), 1e-4)
