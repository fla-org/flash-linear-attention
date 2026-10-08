# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.convolution import ShortConvolution, causal_conv1d
from fla.modules.l2norm import l2_norm
from fla.ops.convolution.fused_short_conv import fused_short_conv
from fla.utils import assert_close, device


def _check_fused_short_conv(
    B,
    T,
    H,
    head_dim,
    W,
    activation,
    has_bias,
    has_residual,
    has_initial_state,
    output_final_state,
    strided,
    dtype,
    cu_seqlens=None,
    norm_eps=1e-6,
    use_norm=True,
):
    torch.manual_seed(42)
    D = H * head_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if strided:
        x = torch.randn(B, T, 2 * D, device=device, dtype=dtype)[..., ::2].detach().requires_grad_()
        dy = torch.randn(B, T, 2 * D, device=device, dtype=dtype)[..., ::2]
        assert not x.is_contiguous()
        assert not dy.is_contiguous()
    else:
        x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
        dy = torch.randn_like(x)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True) if has_bias else None
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True) if has_residual else None
    initial_state = torch.randn(N, D, W, device=device, dtype=dtype, requires_grad=True) if has_initial_state else None

    kwargs = dict(
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        output_final_state=output_final_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    ref, ref_state = causal_conv1d(x=x.contiguous(), backend='triton', **kwargs)
    if use_norm:
        ref = l2_norm(x=ref.view(B, T, H, head_dim), eps=norm_eps).view(B, T, D)
    actual, actual_state = fused_short_conv(
        x=x,
        use_norm=use_norm,
        norm_eps=norm_eps,
        head_dim=head_dim,
        **kwargs,
    )
    assert actual.shape == ref.shape
    assert actual.dtype == ref.dtype == dtype
    assert_close('y', ref, actual, 1e-3)
    assert torch.isfinite(actual).all()

    ref_outputs, actual_outputs, cotangents = [ref], [actual], [dy]
    if output_final_state:
        assert actual_state.shape == (N, D, W)
        assert_close('state', ref_state, actual_state, 1e-3)
        assert torch.isfinite(actual_state).all()
        ref_outputs.append(ref_state)
        actual_outputs.append(actual_state)
        cotangents.append(torch.randn_like(ref_state))
    else:
        assert ref_state is None and actual_state is None

    named_inputs = [('x', x), ('weight', weight), ('bias', bias), ('residual', residual), ('h0', initial_state)]
    named_inputs = [(name, tensor) for name, tensor in named_inputs if tensor is not None]
    inputs = [tensor for _, tensor in named_inputs]
    ref_grads = torch.autograd.grad(ref_outputs, inputs, grad_outputs=cotangents)
    actual_grads = torch.autograd.grad(actual_outputs, inputs, grad_outputs=cotangents)
    for (name, _), ref_grad, actual_grad in zip(named_inputs, ref_grads, actual_grads, strict=True):
        assert torch.isfinite(actual_grad).all(), name
        assert_close(f'd{name}', ref_grad, actual_grad, 1e-3)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'head_dim', 'W', 'activation', 'has_bias', 'has_residual', 'has_initial_state',
     'output_final_state', 'strided'),
    [
        pytest.param(2, 63, 3, 80, 4, 'silu', True, True, True, True, False, id='tail63-head80-all-grads'),
        pytest.param(2, 65, 2, 96, 3, 'swish', True, False, True, False, True, id='tail65-head96-strided'),
        pytest.param(2, 2, 2, 64, 4, None, False, True, True, True, False, id='short-with-state'),
        pytest.param(1, 1, 3, 80, 4, 'silu', True, False, False, True, False, id='single-no-state'),
        pytest.param(1, 128, 2, 128, 4, 'silu', False, False, False, False, False, id='aligned-no-state'),
        pytest.param(2, 65, 3, 96, 1, None, True, True, False, True, True, id='width1-residual-strided'),
        pytest.param(1, 17, 2, 256, 4, 'silu', True, True, True, True, False, id='head256-boundary'),
        pytest.param(1, 17, 2, 320, 4, 'silu', True, True, True, True, False, id='head320-fallback'),
    ],
)
def test_fused_short_conv(
    B, T, H, head_dim, W, activation, has_bias, has_residual, has_initial_state, output_final_state, strided, dtype,
):
    _check_fused_short_conv(
        B=B,
        T=T,
        H=H,
        head_dim=head_dim,
        W=W,
        activation=activation,
        has_bias=has_bias,
        has_residual=has_residual,
        has_initial_state=has_initial_state,
        output_final_state=output_final_state,
        strided=strided,
        dtype=dtype,
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('lengths', 'head_dim', 'has_initial_state', 'output_final_state', 'strided'),
    [
        pytest.param([0, 1, 2, 0, 63, 65, 0], 80, True, True, True, id='empty-short-tail-state-strided'),
        pytest.param([2, 0, 1], 96, False, True, False, id='short-no-state'),
        pytest.param([65, 3, 63], 96, True, False, False, id='state-no-final'),
    ],
)
def test_fused_short_conv_varlen(lengths, head_dim, has_initial_state, output_final_state, strided, dtype):
    cu_seqlens = torch.tensor([0, *lengths], device=device, dtype=torch.long).cumsum(0)
    _check_fused_short_conv(
        B=1,
        T=sum(lengths),
        H=3,
        head_dim=head_dim,
        W=4,
        activation='silu',
        has_bias=True,
        has_residual=True,
        has_initial_state=has_initial_state,
        output_final_state=output_final_state,
        strided=strided,
        dtype=dtype,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize('use_norm', [False, True])
def test_fused_short_conv_norm_options(use_norm):
    _check_fused_short_conv(
        B=2,
        T=65,
        H=3,
        head_dim=80,
        W=4,
        activation='silu',
        has_bias=True,
        has_residual=True,
        has_initial_state=True,
        output_final_state=True,
        strided=True,
        dtype=torch.float32,
        norm_eps=0.1,
        use_norm=use_norm,
    )


def test_fused_short_conv_default():
    torch.manual_seed(42)
    x = torch.randn(2, 63, 160, device=device)
    weight = torch.randn(160, 4, device=device)
    ref, ref_state = causal_conv1d(x=x, weight=weight, activation='silu', output_final_state=True, backend='triton')
    actual, actual_state = fused_short_conv(x=x, weight=weight, activation='silu', output_final_state=True)
    assert_close('y', ref, actual, 0.0)
    assert_close('state', ref_state, actual_state, 0.0)


@pytest.mark.parametrize('head_dim', [None, 0, -1, 63, 80.0])
def test_fused_short_conv_invalid_head_dim(head_dim):
    x = torch.zeros(1, 2, 160, device=device)
    weight = torch.zeros(160, 4, device=device)
    with pytest.raises(ValueError):
        fused_short_conv(x=x, weight=weight, use_norm=True, head_dim=head_dim)


@pytest.mark.parametrize('norm_eps', [0.0, -1e-6])
def test_fused_short_conv_invalid_norm_eps(norm_eps):
    x = torch.zeros(1, 2, 160, device=device)
    weight = torch.zeros(160, 4, device=device)
    with pytest.raises(ValueError):
        fused_short_conv(x=x, weight=weight, use_norm=True, head_dim=80, norm_eps=norm_eps)


@pytest.mark.parametrize('shape', [(2, 160), (1, 2, 2, 80)])
def test_fused_short_conv_invalid_rank(shape):
    x = torch.zeros(shape, device=device)
    weight = torch.zeros(160, 4, device=device)
    with pytest.raises(ValueError):
        fused_short_conv(x=x, weight=weight, use_norm=True, head_dim=80)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('head_dim', [80, 96])
@torch.no_grad()
def test_short_convolution_l2_prefill_decode(head_dim, dtype):
    torch.manual_seed(42)
    B, H, W = 2, 3, 4
    D = H * head_dim
    conv = ShortConvolution(hidden_size=D, kernel_size=W, bias=True, norm='l2', backend='triton').to(device, dtype)
    reference = ShortConvolution(hidden_size=D, kernel_size=W, bias=True, backend='triton').to(device, dtype)
    reference.load_state_dict(conv.state_dict())
    initial_state = torch.randn(B, D, W, device=device, dtype=dtype)
    actual_cache, ref_cache = initial_state.clone(), initial_state.clone()

    for T in [2, 1, 1, 65, 1]:
        x = torch.randn(B, T, D, device=device, dtype=dtype)
        residual = torch.randn_like(x)
        ref, ref_cache = reference(x=x, residual=residual, cache=ref_cache, output_final_state=True)
        ref = l2_norm(x=ref.view(B, T, H, head_dim)).view_as(ref)
        actual, actual_cache = conv(
            x=x,
            residual=residual,
            cache=actual_cache,
            output_final_state=True,
            head_dim=head_dim,
        )
        assert_close('y', ref, actual, 1e-3)
        assert_close('cache', ref_cache, actual_cache, 1e-3)


@pytest.mark.parametrize('T', [1, 65])
@torch.no_grad()
def test_short_convolution_disable_l2(T):
    torch.manual_seed(42)
    conv = ShortConvolution(hidden_size=160, kernel_size=4, norm='l2', backend='triton').to(device)
    reference = ShortConvolution(hidden_size=160, kernel_size=4, backend='triton').to(device)
    reference.load_state_dict(conv.state_dict())
    x = torch.randn(2, T, 160, device=device)
    ref, ref_state = reference(x=x, output_final_state=True)
    actual, actual_state = conv(x=x, use_norm=False, output_final_state=True)
    assert_close('y', ref, actual, 0.0)
    assert_close('state', ref_state, actual_state, 0.0)


@torch.no_grad()
def test_short_convolution_l2_varlen_empty_sequence():
    torch.manual_seed(42)
    H, head_dim, W = 2, 80, 4
    D = H * head_dim
    conv = ShortConvolution(hidden_size=D, kernel_size=W, norm='l2', backend='triton').to(device)
    x = torch.randn(1, 3, D, device=device)
    cu_seqlens = torch.tensor([0, 0, 2, 3], device=device)
    initial_state = torch.randn(3, D, W, device=device)
    ref, ref_state = causal_conv1d(
        x=x,
        weight=conv.weight.squeeze(1),
        initial_state=initial_state,
        output_final_state=True,
        activation=conv.activation,
        backend='triton',
        cu_seqlens=cu_seqlens,
    )
    ref = l2_norm(x=ref.view(1, 3, H, head_dim)).view_as(ref)
    actual, actual_state = conv(
        x=x,
        cache=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_dim=head_dim,
    )
    assert_close('y', ref, actual, 1e-3)
    assert_close('state', ref_state, actual_state, 1e-3)


@pytest.mark.parametrize('norm_eps', [0.0, -1e-6])
def test_short_convolution_invalid_norm_eps(norm_eps):
    with pytest.raises(ValueError, match='norm_eps'):
        ShortConvolution(hidden_size=160, kernel_size=4, norm='l2', norm_eps=norm_eps)


@pytest.mark.parametrize('head_dim', [None, 0, -1, 63, 80.0])
@pytest.mark.parametrize('T', [1, 2])
def test_short_convolution_invalid_head_dim(head_dim, T):
    conv = ShortConvolution(hidden_size=160, kernel_size=4, norm='l2')
    x = torch.zeros(1, T, 160)
    with pytest.raises(ValueError, match='head_dim'):
        conv(x=x, head_dim=head_dim)
