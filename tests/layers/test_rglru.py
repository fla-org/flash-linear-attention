# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.layers.rglru import RGLRU
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device


def rglru_ref(layer: RGLRU, x: torch.Tensor) -> torch.Tensor:
    """The recurrent block of a single sequence `x` of shape `[1, T, D]`, with the RG-LRU as a sequential loop."""
    T = x.shape[1]
    y = F.gelu(layer.linear_y(x), approximate='tanh')
    u = layer.linear_x(x).transpose(1, 2)
    u = F.conv1d(u, layer.conv1d.weight, layer.conv1d.bias, padding=layer.conv_size - 1, groups=layer.lru_width)
    u = u[..., :T].transpose(1, 2)
    log_a = -layer.c * torch.sigmoid(layer.a_gate(u)) * F.softplus(layer.a_param)
    gated = u * torch.sigmoid(layer.input_gate(u))
    h = torch.zeros_like(u[:, 0])
    hs = []
    for t in range(T):
        a = log_a[:, t].exp()
        h = gated[:, t] if t == 0 else a * h + torch.sqrt(1 - a ** 2) * gated[:, t]
        hs.append(h)
    return layer.linear_out(torch.stack(hs, 1) * y)


def make_layer(mode='chunk', **kwargs):
    torch.manual_seed(42)
    return RGLRU(mode=mode, hidden_size=64, lru_width=96, num_heads=4, layer_idx=0, **kwargs).to(device)


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
@pytest.mark.parametrize('T', [1, 37, 100])
def test_forward_backward(mode: str, T: int):
    layer = make_layer(mode).train()
    x = torch.randn(2, T, 64, device=device, requires_grad=True)
    do = torch.randn(2, T, 64, device=device)

    tri = layer(x)[0]
    (tri * do).sum().backward()
    tri_grads = [x.grad] + [p.grad for p in layer.parameters()]
    x.grad = None
    layer.zero_grad()

    ref = torch.cat([rglru_ref(layer, x[i:i+1]) for i in range(2)], 0)
    (ref * do).sum().backward()
    ref_grads = [x.grad] + [p.grad for p in layer.parameters()]

    assert_close('o', ref, tri, 1e-4)
    names = ['dx'] + [f'd{n}' for n, _ in layer.named_parameters()]
    for name, r, t in zip(names, ref_grads, tri_grads):
        # the decay gets no gradient from single-token sequences, which skip the recurrence
        if r is None:
            assert t is None or not t.any(), name
            continue
        assert_close(name, r, t, 1e-4)


@pytest.mark.parametrize('prefill', [1, 5, 30])
def test_decode(prefill: int):
    """Prefilling and then decoding token by token reproduces a single forward pass."""
    layer = make_layer().eval()
    B, T = 2, 40
    x = torch.randn(B, T, 64, device=device)
    with torch.no_grad():
        ref = layer(x)[0]
        cache = Cache()
        outs = [layer(x[:, :prefill], past_key_values=cache, use_cache=True)[0]]
        for t in range(prefill, T):
            outs.append(layer(x[:, t:t+1], past_key_values=cache, use_cache=True)[0])
    assert_close('o', ref, torch.cat(outs, 1), 1e-4)
    assert cache.get_seq_length(0) == T


def test_left_padding():
    layer = make_layer().eval()
    lens, prefill, T = [20, 9, 14], 20, 32
    x = torch.randn(len(lens), T, 64, device=device)
    mask = torch.ones(len(lens), T, dtype=torch.long, device=device)
    for i, n in enumerate(lens):
        mask[i, :prefill - n] = 0
    with torch.no_grad():
        cache = Cache()
        outs = [layer(x[:, :prefill], attention_mask=mask[:, :prefill], past_key_values=cache, use_cache=True)[0]]
        for t in range(prefill, T):
            outs.append(layer(x[:, t:t+1], attention_mask=mask[:, :t+1], past_key_values=cache, use_cache=True)[0])
        o = torch.cat(outs, 1)
        for i, n in enumerate(lens):
            assert_close(f'o{i}', rglru_ref(layer, x[i:i+1, prefill - n:]), o[i:i+1, prefill - n:], 1e-4)


def test_varlen():
    layer = make_layer().train()
    cu_seqlens = [0, 7, 40, 64]
    x = torch.randn(1, cu_seqlens[-1], 64, device=device, requires_grad=True)
    tri = layer(x, cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=device))[0]
    tri.sum().backward()
    tri_dx, x.grad = x.grad, None
    ref = torch.cat([rglru_ref(layer, x[:, bos:eos]) for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:])], 1)
    ref.sum().backward()
    assert_close('o', ref, tri, 1e-4)
    assert_close('dx', x.grad, tri_dx, 1e-4)


def test_init():
    layer = make_layer()
    a = torch.exp(-F.softplus(layer.a_param))
    assert a.min() >= 0.9 - 1e-6 and a.max() <= 0.999 + 1e-6
