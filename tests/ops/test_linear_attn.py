# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.linear_attn import (chunk_linear_attn, fused_chunk_linear_attn,
                                 fused_recurrent_linear_attn)
from fla.ops.linear_attn.naive import naive_chunk_linear_attn


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    ref = naive_chunk_linear_attn(q, k, v, normalize=False)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, _ = fused_recurrent_linear_attn(q, k, v, normalize=False)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref, ref_ht = fused_recurrent_linear_attn(q, k, v, initial_state=h0, output_final_state=True, normalize=False)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, tri_ht = chunk_linear_attn(q, k, v, initial_state=h0, output_final_state=True, normalize=False)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_ht.allclose(tri_ht, 0, atol), f"ht diff: {torch.abs(ref_ht - tri_ht).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    atol = 1e-3
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    h0 = torch.zeros((B, H, D, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)
    ref, ref_ht = fused_recurrent_linear_attn(q, k, v, initial_state=h0, output_final_state=True, normalize=False)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, tri_ht = fused_chunk_linear_attn(q, k, v, initial_state=h0, output_final_state=True, normalize=False)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert ref.allclose(tri, 0, atol), f" o diff: {torch.abs(ref - tri).max()}"
    assert ref_ht.allclose(tri_ht, 0, atol), f"ht diff: {torch.abs(ref_ht - tri_ht).max()}"
    assert ref_dq.allclose(tri_dq, 0, atol), f"dq diff: {torch.abs(ref_dq - tri_dq).max()}"
    assert ref_dk.allclose(tri_dk, 0, atol), f"dk diff: {torch.abs(ref_dk - tri_dk).max()}"
    assert ref_dv.allclose(tri_dv, 0, atol), f"dv diff: {torch.abs(ref_dv - tri_dv).max()}"
