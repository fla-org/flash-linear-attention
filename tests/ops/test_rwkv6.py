# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6
from fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("gate_logit_normalizer", [0.05, 1, 20])
@pytest.mark.parametrize("T", [130, 146, 162, 178, 300, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [300, 100])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    gate_logit_normalizer: float,
    head_first: bool
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    if head_first:
        q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
        k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
        v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
        w = F.logsigmoid(torch.randn((B, H, T, D), dtype=dtype, device='cuda')) / gate_logit_normalizer
    else:
        q = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
        k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
        v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
        w = F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device='cuda')) / gate_logit_normalizer

    u = torch.randn(H, D, dtype=dtype, device='cuda').requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=dtype, device='cuda').requires_grad_()
    w = w.requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_rwkv6(q.clone(),
                                        k.clone(),
                                        v.clone(),
                                        w.clone(),
                                        u.clone(),
                                        initial_state=h0.clone(),
                                        output_final_state=True,
                                        head_first=head_first)
    ref, _ = fused_recurrent_rwkv6(q.clone(),
                                   k.clone(),
                                   v.clone(),
                                   w.clone(),
                                   u.clone(),
                                   initial_state=h0.clone(),
                                   output_final_state=False,
                                   head_first=head_first)

    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    # triton implementation
    tri, tri_ht = chunk_rwkv6(q.clone(),
                              k.clone(),
                              v.clone(),
                              w.clone(),
                              u.clone(),
                              initial_state=h0.clone(),
                              output_final_state=True,
                              head_first=head_first)
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


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [64, 128, 200, 250, 256, 300, 400, 512, 1000, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [300, 100])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
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
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).cuda().sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    w = F.logsigmoid(torch.randn((1, T, H, D), dtype=dtype, device='cuda')).requires_grad_(True)
    u = torch.randn(H, D, dtype=dtype, device='cuda').requires_grad_(True)
    h0 = torch.randn((N, H, D, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=offsets,
        head_first=False
    )
    ref, _ = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=False,
        cu_seqlens=offsets,
        head_first=False
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
        cu_seqlens=offsets,
        head_first=False
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
