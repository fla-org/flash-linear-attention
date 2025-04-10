# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn
from fla.ops.hgrn.naive import naive_recurrent_hgrn
from fla.ops.utils.testing import COMPILER_MODE, assert_close
from fla.utils import device

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_t_varlen_list = test_t_list
    test_d_list = [500, 1024]
    test_gate_list = [1.0]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [500, 1024]
    test_gate_list = [1, 0.1, 10]
test_h_list = [2]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_fused_recurrent(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn((B, T, D), dtype=dtype, device=device)
    g = torch.randn((B, T, D), dtype=dtype, device=device)
    h0 = torch.randn_like(x[:, 0])
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g, h0 = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g, h0))

    do = torch.randn_like(x)
    dht = torch.randn_like(h0)
    ref, ref_ht = naive_recurrent_hgrn(x, g, h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('  o', ref, tri, 0.005)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dx', ref_dx, tri_dx, 0.005)
    assert_close(' dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize('N', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_fused_recurrent_varlen(
    N: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn((1, T, D), dtype=dtype, device=device)
    g = torch.randn((1, T, D), dtype=dtype, device=device)
    h0 = torch.randn(N, D, dtype=dtype, device=device)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g, h0 = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g, h0))

    do = torch.randn_like(x)
    dht = torch.randn_like(h0)
    refs, ref_hts = [], []
    for i in range(N):
        ref, ref_ht = naive_recurrent_hgrn(
            x[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            h0[i:i+1],
            output_final_state=True
        )
        refs.append(ref)
        ref_hts.append(ref_ht)
    ref = torch.cat(refs, 1)
    ref_ht = torch.cat(ref_hts, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_hgrn(x, g, h0, output_final_state=True, cu_seqlens=cu_seqlens)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('  o', ref, tri, 0.005)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dx', ref_dx, tri_dx, 0.005)
    assert_close(' dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_chunk(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn((B, T, D), dtype=dtype, device=device)
    g = torch.randn((B, T, D), dtype=dtype, device=device)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))

    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, 0])
    ref, _ = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = chunk_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close(' o', ref, tri, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
