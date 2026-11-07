# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.layers.lolcats import LoLCATsAttention
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device


def lolcats_ref(layer: LoLCATsAttention, x: torch.Tensor) -> torch.Tensor:
    """The quadratic hybrid attention of the LoLCATs reference implementation, with explicit masks."""
    T = x.shape[1]
    q, k, v = (rearrange(p(x), '... (h d) -> ... h d', d=layer.head_dim) for p in (layer.q_proj, layer.k_proj, layer.v_proj))
    q, k = layer.rotary(q, k)
    k, v = (t.repeat_interleave(layer.num_kv_groups, dim=2) for t in (k, v))
    fq, fk = layer.feature_map_q(q), layer.feature_map_k(k)
    q, k, v, fq, fk = (t.transpose(1, 2).float() for t in (q, k, v, fq, fk))

    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    linear = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-layer.window_size)
    window = causal & ~linear
    w = torch.sigmoid(layer.window_factors.float())[None, :, None, None]
    l = 1 - w if layer.affine_attention_factors else 1

    a_sm = (q @ k.transpose(-1, -2) * layer.head_dim ** -0.5).masked_fill(~window, -1e8)
    a_sm = w * (a_sm - a_sm.amax(-1, keepdim=True)).exp()
    a_ln = l * (fq @ fk.transpose(-1, -2)).masked_fill(~linear, 0)
    o = (a_sm + a_ln) @ v / (a_sm.sum(-1, keepdim=True) + a_ln.sum(-1, keepdim=True))
    return layer.o_proj(o.transpose(1, 2).flatten(-2, -1).to(x.dtype))


def make_layer(num_kv_heads=None, affine=False, window_size=16):
    torch.manual_seed(42)
    layer = LoLCATsAttention(
        hidden_size=64, num_heads=4, num_kv_heads=num_kv_heads, feature_dim=8, window_size=window_size,
        affine_attention_factors=affine, init_window_factor=-1., layer_idx=0,
    ).to(device)
    with torch.no_grad():
        layer.window_factors.normal_()
    return layer


@pytest.mark.parametrize('T', [1, 10, 16, 37, 100])
@pytest.mark.parametrize(('num_kv_heads', 'affine'), [(None, False), (2, True)])
def test_forward_backward(T, num_kv_heads, affine):
    layer = make_layer(num_kv_heads, affine)
    x = torch.randn(2, T, 64, device=device, requires_grad=True)
    do = torch.randn_like(x)

    tri = layer(x)[0]
    (tri * do).sum().backward()
    tri_grads = [x.grad] + [p.grad for p in layer.parameters()]
    x.grad = None
    layer.zero_grad()
    ref = lolcats_ref(layer, x)
    (ref * do).sum().backward()
    ref_grads = [x.grad] + [p.grad for p in layer.parameters()]

    assert_close('o', ref, tri, 1e-4)
    for name, r, t in zip(['dx'] + [f'd{n}' for n, _ in layer.named_parameters()], ref_grads, tri_grads):
        if r is None:
            assert t is None or not t.any(), name
        else:
            assert_close(name, r, t, 1e-4)


def test_softmax_within_window():
    """Within the window there are no linear attention terms, so the layer is softmax attention."""
    layer = make_layer(window_size=32)
    x = torch.randn(2, 32, 64, device=device)
    q, k, v = (rearrange(p(x), '... (h d) -> ... h d', d=layer.head_dim) for p in (layer.q_proj, layer.k_proj, layer.v_proj))
    q, k = layer.rotary(q, k)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    assert_close('o', layer.o_proj(o.transpose(1, 2).flatten(-2, -1)), layer(x)[0], 1e-5)


@pytest.mark.parametrize('prefill', [1, 5, 16, 30])
@pytest.mark.parametrize('num_kv_heads', [None, 2])
def test_decode(prefill, num_kv_heads):
    """Prefilling and then decoding token by token reproduces a single forward pass."""
    layer = make_layer(num_kv_heads).eval()
    x = torch.randn(2, 60, 64, device=device)
    with torch.no_grad():
        ref = layer(x)[0]
        cache = Cache()
        outs = [layer(x[:, :prefill], past_key_values=cache, use_cache=True)[0]]
        for t in range(prefill, x.shape[1]):
            outs.append(layer(x[:, t:t+1], past_key_values=cache, use_cache=True)[0])
    assert_close('o', ref, torch.cat(outs, 1), 1e-4)


def test_attention_transfer_loss():
    layer = make_layer(num_kv_heads=2, window_size=8)
    x = torch.randn(2, 40, 64, device=device)
    loss = layer.attention_transfer_loss(x)
    loss.backward()
    # only the feature maps and window factors are trained to mimic softmax attention
    assert loss > 0
    assert layer.feature_map_q.layer.grad.abs().sum() > 0 and layer.feature_map_k.layer.grad.abs().sum() > 0
    assert layer.window_factors.grad.abs().sum() > 0
    # the window covers the whole sequence, so the layer is exactly softmax attention
    assert make_layer(window_size=64).attention_transfer_loss(x) < 1e-10


def test_left_padding():
    """Left-padded prefill and decoding match every sequence run on its own."""
    layer = make_layer(num_kv_heads=2).eval()
    lens, prefill, T = [30, 9, 21], 30, 50
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
        assert torch.isfinite(o).all()
        for i, n in enumerate(lens):
            assert_close(f'o{i}', layer(x[i:i+1, prefill - n:])[0], o[i:i+1, prefill - n:], 1e-4)
