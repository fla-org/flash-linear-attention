# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6
from fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from fla.ops.utils.testing import COMPILER_MODE, assert_close
from fla.utils import device, device_platform

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
    test_gate_list = [1.0]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [64, 32, 100, 256]
    test_gate_list = [1, 0.1, 10]
test_h_list = [2]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('gate_logit_normalizer', test_gate_list)
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    gate_logit_normalizer: float,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    w = F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device=device)) / gate_logit_normalizer

    u = torch.randn(H, D, dtype=dtype, device=device).requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=dtype, device=device).requires_grad_()
    w = w.requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ref, _ = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=False,
    )

    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    # triton implementation
    tri, tri_ht = chunk_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((tri * do).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('  o', ref, tri, 0.004)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dw', ref_dw, tri_dw, 0.005)
    assert_close(' du', ref_du, tri_du, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize("N", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
def test_chunk_varlen(
    N: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    w = F.logsigmoid(torch.randn((1, T, H, D), dtype=dtype, device=device)).requires_grad_(True)
    u = torch.randn(H, D, dtype=dtype, device=device).requires_grad_(True)
    h0 = torch.randn((N, H, D, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ref, _ = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None
    assert_close('  o', ref, tri, 0.004)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dw', ref_dw, tri_dw, 0.005)
    assert_close(' du', ref_du, tri_du, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)
