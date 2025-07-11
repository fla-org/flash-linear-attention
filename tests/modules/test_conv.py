# -*- coding: utf-8 -*-

import pytest
import torch
from einops import rearrange

from fla.modules.convolution import causal_conv1d, causal_conv1d_update
from fla.utils import assert_close, device

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@pytest.mark.parametrize('B', [4])
@pytest.mark.parametrize('T', [1, 500, 1024])
@pytest.mark.parametrize('D', [128, 200, 1024])
@pytest.mark.parametrize('W', [3, 4])
@pytest.mark.parametrize('activation', [None, 'swish'])
@pytest.mark.parametrize('has_bias', [False, True])
@pytest.mark.parametrize('has_residual', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
def test_conv(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(B, T, D).to(device, dtype)

    ref = causal_conv1d_fn(
        x=rearrange(x, "b t d -> b d t"),
        weight=weight,
        bias=bias,
        activation=activation,
    )
    ref = rearrange(ref, "b d t -> b t d")
    if has_residual:
        ref += residual
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri = causal_conv1d(x, weight, bias, residual=residual, activation=activation)
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None

    assert_close(" y", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    if has_bias:
        assert_close("db", ref_db, tri_db, 1e-3)
    if has_residual:
        assert_close("dr", ref_dr, tri_dr, 1e-3)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [500, 1024])
@pytest.mark.parametrize('D', [128, 200, 1024])
@pytest.mark.parametrize("W", [3, 4])
@pytest.mark.parametrize("activation", [None, 'swish'])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
def test_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(1, T, D).to(device, dtype)

    ref = torch.cat([
        rearrange(
            causal_conv1d_fn(
                x=rearrange(x[:, bos:eos].contiguous(), "b t d -> b d t"),
                weight=weight,
                bias=bias,
                activation=activation,
            ),
            "b t d -> b d t"
        ) + (residual[:, bos:eos] if has_residual else torch.zeros_like(x[:, bos:eos]))
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], 1)
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri = causal_conv1d(x, weight, bias, residual=residual, activation=activation, cu_seqlens=cu_seqlens)
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None

    assert_close(" y", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    if has_bias:
        assert_close("db", ref_db, tri_db, 1e-3)
    if has_residual:
        assert_close("dr", ref_dr, tri_dr, 1e-3)


@pytest.mark.parametrize('B', [4])
@pytest.mark.parametrize('T', [1, 500, 1024])
@pytest.mark.parametrize('D', [128, 200, 1024])
@pytest.mark.parametrize('W', [3, 4])
@pytest.mark.parametrize('activation', [None, 'swish'])
@pytest.mark.parametrize('has_bias', [False, True])
@pytest.mark.parametrize('has_residual', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.skipif(
    causal_conv1d_fn is None,
    reason="causal_conv1d is not installed"
)
def test_conv_decoding(
        B: int,
        T: int,
        D: int,
        W: int,
        activation: str,
        has_bias: bool,
        has_residual: bool,
        dtype: torch.dtype
):
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype)
    weight = torch.randn(D, W).to(device, dtype) * 0
    bias = torch.randn(D).to(device, dtype) if has_bias else None
    residual = x.clone() if has_residual else None

    ref = causal_conv1d_fn(
        x=rearrange(x, "b t d -> b d t"),
        weight=weight,
        bias=bias,
        activation=activation,
    )
    ref = rearrange(ref, "b d t -> b t d")
    if has_residual:
        ref += residual
    ref_cache = x.new_zeros(B, D, W)
    ref_cache[:, :, -min(W, T):].copy_(rearrange(x[..., -min(W, T):, :], 'n w d -> n d w'))

    tri = torch.zeros_like(x)
    tri_cache = x.new_zeros(B, D, W)
    for i in range(T):
        y, tri_cache = causal_conv1d_update(
            x=x[:, i:i+1, :],
            cache=tri_cache,
            residual=residual[:, i:i+1, :] if has_residual else None,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        tri[:, i:i+1, :] = y

    assert_close("    y", ref, tri, 1e-3)
    assert_close("cache", ref_cache, tri_cache, 1e-3)
