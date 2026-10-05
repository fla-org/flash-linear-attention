# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.backends.conv_gluon import ConvGluonBackend
from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.utils import IS_NVIDIA, assert_close, device


@pytest.mark.skipif(not IS_NVIDIA, reason='Gluon convolution requires NVIDIA')
@pytest.mark.parametrize(
    ('dtype', 'weight_dtype'),
    [(torch.float32, torch.float32), (torch.float16, torch.float32), (torch.bfloat16, torch.float32),
     (torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)],
)
@pytest.mark.parametrize('activation', [None, 'silu'])
@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'packed', 'state', 'strided'),
    [
        (2, 1, 33, 4, False, False, False),
        (2, 63, 65, 3, False, False, True),
        (1, 129, 127, 2, False, False, False),
        (1, 257, 256, 4, True, False, True),
        (1, 129, 65, 4, True, True, False),
        (2, 3, 65, 4, False, True, True),
    ],
    ids=['one-token', 'channel-tail', 'time-tail', 'packed-qkv', 'packed-state', 'short-state'],
)
def test_causal_conv1d_gluon(monkeypatch, B, T, D, W, packed, state, strided, activation, dtype, weight_dtype):
    torch.manual_seed(42)
    x = torch.randn(B, T, D * (3 if strided else 1), device=device, dtype=dtype)
    x = x[..., D:2 * D] if strided else x
    x.requires_grad_(True)
    weight = torch.randn(D, W, device=device, dtype=weight_dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=weight_dtype, requires_grad=True)
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    cu = torch.tensor([0, 0, 1, 3, T], device=device) if packed else None
    N = 4 if packed else B
    h0 = torch.randn(N, D, W, device=device, dtype=dtype, requires_grad=True) if state else None
    inputs = (x, weight, bias, residual) + ((h0,) if state else ())
    dy = torch.randn(B, T, D * 2, device=device, dtype=dtype)[..., ::2]
    dht = torch.randn_like(h0) if state else None
    results = []
    for enabled in ['0', '1']:
        monkeypatch.setenv('FLA_CONV_GLUON', enabled)
        y, ht = causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=h0,
            output_final_state=state,
            activation=activation,
            cu_seqlens=cu,
        )
        grads = torch.autograd.grad((y, ht) if state else y, inputs, (dy, dht) if state else dy)
        results.append((y, ht, grads))
    ref, out = results
    assert_close('y', ref[0], out[0], 1e-3)
    if state:
        assert_close('ht', ref[1], out[1], 1e-3)
    for name, expected, actual in zip(('dx', 'dw', 'db', 'dr', 'dh0'), ref[2], out[2]):
        assert_close(name, expected, actual, 1e-3)


@pytest.mark.parametrize('case', ['rank', 'channels', 'width', 'weight', 'packed-batch', 'chunk', 'state', 'dtype', 'distributed'])
def test_conv_gluon_verifier(monkeypatch, case):
    backend = ConvGluonBackend()
    x = torch.empty(2, 64, 32)
    weight = torch.empty(32, 4)
    kwargs = {}
    if case == 'dtype':
        x = x.double()
    elif case == 'distributed':
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    elif case == 'rank':
        x = x.unsqueeze(-1)
    elif case == 'channels':
        x = x[..., ::2]
    elif case == 'width':
        weight = torch.empty(32, 5)
    elif case == 'weight':
        weight = None
    elif case == 'packed-batch':
        kwargs['cu_seqlens'] = torch.tensor([0, 128])
    elif case == 'chunk':
        kwargs['BT'] = 32
    if case == 'state':
        accepted, reason = backend.causal_conv1d_bwd_verifier(
            x=x,
            dy=x,
            dht=None,
            weight=weight,
            initial_state=torch.empty(2, 32, 4),
        )
    else:
        accepted, reason = backend.causal_conv1d_fwd_verifier(x=x, weight=weight, **kwargs)
    assert not accepted and reason


@pytest.mark.skipif(not IS_NVIDIA, reason='Gluon convolution requires NVIDIA')
def test_conv_gluon_dispatch(monkeypatch):
    from fla.modules.conv import gluon

    torch.manual_seed(42)
    calls = []
    fwd, bwd = gluon.causal_conv1d_fwd, gluon.causal_conv1d_bwd

    def forward(*args, **kwargs):
        calls.append('fwd')
        return fwd(*args, **kwargs)

    def backward(*args, **kwargs):
        calls.append('bwd')
        return bwd(*args, **kwargs)

    monkeypatch.setattr(gluon, 'causal_conv1d_fwd', forward)
    monkeypatch.setattr(gluon, 'causal_conv1d_bwd', backward)
    monkeypatch.setenv('FLA_CONV_GLUON', '1')
    x = torch.randn(1, 65, 64, device=device, requires_grad=True)
    for W in [4, 5]:
        weight = torch.randn(64, W, device=device, requires_grad=True)
        y, _ = causal_conv1d(x, weight, activation='silu')
        y.sum().backward()
    assert calls == ['fwd', 'bwd']
