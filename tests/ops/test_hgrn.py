# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn
from fla.ops.hgrn.naive import naive_recurrent_hgrn


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
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [500, 1024])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn((B, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, T, D), dtype=dtype, device='cuda')
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

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dx", ref_dx, tri_dx, 0.005)
    assert_close("dg", ref_dg, tri_dg, 0.005)
    assert_close("dh0", ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [500, 1024])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent_varlen(
    N: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).cuda().sort()[0]

    x = torch.randn((1, T, D), dtype=dtype, device='cuda')
    g = torch.randn((1, T, D), dtype=dtype, device='cuda')
    h0 = torch.randn(N, D, dtype=dtype, device='cuda')
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g, h0 = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g, h0))

    do = torch.randn_like(x)
    dht = torch.randn_like(h0)
    refs, ref_hts = [], []
    for i in range(N):
        ref, ref_ht = naive_recurrent_hgrn(
            x[:, offsets[i]:offsets[i+1]],
            g[:, offsets[i]:offsets[i+1]],
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

    tri, tri_ht = fused_recurrent_hgrn(x, g, h0, output_final_state=True, cu_seqlens=offsets)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dx", ref_dx, tri_dx, 0.005)
    assert_close("dg", ref_dg, tri_dg, 0.005)
    assert_close("dh0", ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [500, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
def test_chunk(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn((B, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, T, D), dtype=dtype, device='cuda')
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

    assert_close(" o", ref, tri, 0.005)
    assert_close("dx", ref_dx, tri_dx, 0.005)
    assert_close("dg", ref_dg, tri_dg, 0.005)
