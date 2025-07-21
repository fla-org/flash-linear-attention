# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.modules.convolution import ShortConvolution, causal_conv1d, causal_conv1d_update
from fla.utils import assert_close, device

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def causal_conv1d_ref_torch(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref_torch(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


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


@pytest.mark.parametrize('B', [2])
@pytest.mark.parametrize('T', [10, 256])
@pytest.mark.parametrize('D', [128])
@pytest.mark.parametrize('W', [3, 4])
@pytest.mark.parametrize('activation', [None, 'swish'])
@pytest.mark.parametrize('has_bias', [False, True])
@pytest.mark.parametrize('has_residual', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('backend', ['triton', 'cuda'])
def test_short_conv_with_cache_prefill_fwd(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype)
    residual = torch.randn(B, T, D).to(device, dtype) if has_residual else None

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=has_bias,
        activation=activation,
        backend=backend,
        device=device,
        dtype=dtype
    )

    cache = torch.randn(B, D, W - 1).to(device, dtype)

    ref = causal_conv1d_ref_torch(
        x=x.transpose(1, 2),                    # (B, D, T)
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        initial_states=cache,                   # (B, D, W-1)
        activation=activation,
    ).transpose(1, 2)                           # (B, T, D)
    if has_residual:
        ref += residual

    zero_padding = torch.zeros(B, D, 1).to(device, dtype)
    tri_cache = torch.cat([zero_padding, cache], dim=-1)  # (B, D, W)
    with torch.no_grad():
        y, _ = conv(x, residual=residual, cache=tri_cache)
    assert_close("y", ref, y, 1e-3)


@pytest.mark.parametrize('B', [2])
@pytest.mark.parametrize('D', [8])
@pytest.mark.parametrize('W', [3, 4])
@pytest.mark.parametrize('activation', [None, 'swish'])
@pytest.mark.parametrize('has_bias', [False, True])
@pytest.mark.parametrize('has_residual', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('backend', ['triton', 'cuda'])
def test_short_conv_decoding_with_cache(
    B: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    torch.manual_seed(42)

    x = torch.randn(B, 1, D).to(device, dtype)        # (B, 1, D)
    residual = x.clone() if has_residual else None

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=has_bias,
        activation=activation,
        backend=backend,
        device=device,
        dtype=dtype
    )

    state = torch.randn(B, D, W).to(device, dtype)

    # reference
    ref = causal_conv1d_update_ref_torch(
        x.squeeze(1),                           # (B, D)
        conv_state=state.clone(),
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        activation=activation,
    ).unsqueeze(1)                             # (B, 1, D)
    if has_residual:
        ref += residual

    # ShortConvolution step
    with torch.no_grad():
        y, _ = conv.step(x, residual, state.clone())

    assert_close("y", ref, y, 1e-3)
