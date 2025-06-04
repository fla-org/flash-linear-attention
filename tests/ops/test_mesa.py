# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

from fla.ops.mesa_net import chunk_mesa_net, naive_mesa_net_exact
from fla.utils import COMPILER_MODE, assert_close, device, device_platform, is_intel_alchemist

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [512]
    test_d_list = [128]
else:
    test_b_list = [2]
    test_t_list = [15, 63, 300, 1000]
    test_d_list = [128]
test_h_list = [3]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('gate_range', [[0.8, 0.99], [0.01, 0.1]])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure'
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    gate_range: Tuple[float, float],
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.rand(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    lower_gate, upper_gate = gate_range
    g = torch.rand(B, T, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    lamb = torch.rand(H, D, dtype=dtype).sigmoid() * 0.75 + 0.25
    q, k, v, beta, g, lamb = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, lamb))
    do = torch.rand_like(v)

    k_init_rand = torch.nn.functional.normalize(torch.rand(B, H, D, device=device, dtype=dtype), dim=-1, p=2)
    h_kk_init = (k_init_rand.unsqueeze(-1) * k_init_rand.unsqueeze(-2)).detach().clone().float().requires_grad_(True)
    h_kv_init = torch.rand(B, H, D, D, dtype=torch.float32, device=device).requires_grad_(True)
    d_h_kk_final = torch.rand_like(h_kk_init)
    d_h_kv_final = torch.rand_like(h_kv_init)

    tri, tri_kk_final, tri_kv_final = chunk_mesa_net(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        lamb=lamb.clone(),
        max_CG_iteration=D,
        h_kk_init=h_kk_init.clone(),
        h_kv_init=h_kv_init.clone(),
        output_final_state=True,
    )

    ((tri * do).sum() + (tri_kk_final * d_h_kk_final).sum() + (tri_kv_final * d_h_kv_final).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dlamb = q.grad, k.grad, v.grad, beta.grad, g.grad, lamb.grad
    tri_dh_kk_init, tri_dh_kv_init = h_kk_init.grad, h_kv_init.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = lamb.grad = h_kk_init.grad = h_kv_init.grad = None

    ref, ref_hkk_final, ref_hkv_final = naive_mesa_net_exact(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        lamb=lamb.clone(),
        h_kk_init=h_kk_init.clone(),
        h_kv_init=h_kv_init.clone(),
    )

    ((ref * do).sum() +
     (ref_hkk_final * d_h_kk_final).sum() + (ref_hkv_final * d_h_kv_final).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dlamb = q.grad, k.grad, v.grad, beta.grad, g.grad, lamb.grad
    ref_dh_kk_init, ref_dh_kv_init = h_kk_init.grad, h_kv_init.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = lamb.grad = h_kk_init.grad = h_kv_init.grad = None

    assert_close('  o', ref, tri, 0.006)
    assert_close(' h_kk_final', ref_hkk_final, tri_kk_final, 0.008)
    assert_close(' h_kv_final', ref_hkv_final, tri_kv_final, 0.008)
    assert_close(' dq', ref_dq, tri_dq, 0.008)
    assert_close(' dk', ref_dk, tri_dk, 0.008)
    assert_close(' dv', ref_dv, tri_dv, 0.008)
    assert_close(' db', ref_dbeta, tri_dbeta, 0.008)
    assert_close(' dg', ref_dg, tri_dg, 0.008)
    assert_close(' dlamb', ref_dlamb, tri_dlamb, 0.015)
    assert_close(' dh_kk_init', ref_dh_kk_init, tri_dh_kk_init, 0.008)
    assert_close(' dh_kv_init', ref_dh_kv_init, tri_dh_kv_init, 0.008)


@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('gate_range', [[0.8, 0.99], [0.01, 0.1]])
@pytest.mark.parametrize('cu_seqlens', [[0, 14, 121, 421, 500], [0, 32, 222, 333, 444, 555, 666, 777, 888, 999, 1000]])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_chunk_varlen(
    H: int,
    D: int,
    gate_range: Tuple[float, float],
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    lower_gate, upper_gate = gate_range
    g = torch.rand(1, T, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    lamb = torch.rand(H, D, dtype=dtype).sigmoid() * 0.75 + 0.25

    k_init_rand = torch.nn.functional.normalize(torch.rand(N, H, D, device=device, dtype=dtype), dim=-1, p=2)
    h_kk_init = (k_init_rand.unsqueeze(-1) * k_init_rand.unsqueeze(-2)).detach().clone().float().requires_grad_(True)
    h_kv_init = torch.rand(N, H, D, D, dtype=torch.float32, device=device).requires_grad_(True)

    q, k, v, beta, g, lamb, h_kk_init, h_kv_init = map(lambda x: x.to(
        device).requires_grad_(), (q, k, v, beta, g, lamb, h_kk_init, h_kv_init))
    do = torch.randn_like(v)
    d_h_kk_final = torch.rand_like(h_kk_init)
    d_h_kv_final = torch.rand_like(h_kv_init)

    tri, tri_h_kk_final, tri_h_kv_final = chunk_mesa_net(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        lamb=lamb.clone(),
        h_kk_init=h_kk_init.clone(),
        h_kv_init=h_kv_init.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ((tri * do).sum() +
     (tri_h_kk_final * d_h_kk_final).sum() + (tri_h_kv_final * d_h_kv_final).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dlamb, tri_dh_kk_init, tri_dh_kv_init = \
        q.grad, k.grad, v.grad, beta.grad, g.grad, lamb.grad, h_kk_init.grad, h_kv_init.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = lamb.grad = h_kk_init.grad = h_kv_init.grad = None

    ref = []
    ref_h_kk_t = []
    ref_h_kv_t = []
    for i in range(N):
        ref_i, ref_h_kk_i, ref_h_kv_i = naive_mesa_net_exact(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            lamb=lamb,
            h_kk_init=h_kk_init[i],
            h_kv_init=h_kv_init[i]
        )
        ref.append(ref_i)
        ref_h_kk_t.append(ref_h_kk_i)
        ref_h_kv_t.append(ref_h_kv_i)
    ref = torch.cat(ref, 1)
    ref_h_kk_t = torch.cat(ref_h_kk_t, 0)
    ref_h_kv_t = torch.cat(ref_h_kv_t, 0)

    ((ref * do).sum() + (ref_h_kk_t * d_h_kk_final).sum() + (ref_h_kv_t * d_h_kv_final).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dlamb, ref_dh_kk_init, ref_dh_kv_init = \
        q.grad, k.grad, v.grad, beta.grad, g.grad, lamb.grad, h_kk_init.grad, h_kv_init.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = lamb.grad = h_kk_init.grad = h_kv_init.grad = None

    assert_close('  o', ref, tri, 0.005)
    assert_close(' h_kk_final', ref_h_kk_t, tri_h_kk_final, 0.005)
    assert_close(' h_kv_final', ref_h_kv_t, tri_h_kv_final, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.007)
    assert_close(' dk', ref_dk, tri_dk, 0.008)
    assert_close(' dv', ref_dv, tri_dv, 0.007)
    assert_close(' db', ref_dbeta, tri_dbeta, 0.015)
    assert_close(' dlamb', ref_dlamb, tri_dlamb, 0.015)
    assert_close(' dg', ref_dg, tri_dg, 0.015)
    assert_close('dh_kk_0', ref_dh_kk_init, tri_dh_kk_init, 0.007)
    assert_close('dh_kv_0', ref_dh_kv_init, tri_dh_kv_init, 0.007)
