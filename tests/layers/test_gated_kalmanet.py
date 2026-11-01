# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.layers.gka import GatedKalmaNet
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(dict(num_heads=4), id="equal-heads"),
        pytest.param(dict(num_heads=2, num_v_heads=4), id="gva"),
        pytest.param(dict(num_heads=4, num_kv_heads=2), id="gqa"),
        pytest.param(dict(num_heads=4, expand_v=2), id="expand_v2"),
        pytest.param(dict(num_heads=2, num_v_heads=4, expand_v=0.5), id="gva-expand_v0.5"),
        pytest.param(dict(num_heads=4, use_forgetting_gate_kk=False), id="no-kk-decay"),
        pytest.param(
            dict(num_heads=4, use_forgetting_gate=False, use_alpha_connection=False, use_beta_gate=False,
                 use_v_conv=False, use_gate=False),
            id="all-flags-off",
        ),
    ],
)
def test_gka_layouts_forward_backward(kwargs: dict):
    """Every layout and flag combination must build the right projections and give finite gradients for every parameter."""
    torch.manual_seed(42)
    # head_dim=64 avoids chunk_gka's 16-bit backward guard (K < 64 or V < 32)
    head_dim, hidden_size = 64, 256
    layer = GatedKalmaNet(hidden_size=hidden_size, head_dim=head_dim, **kwargs).to(device=device, dtype=torch.bfloat16)

    num_heads = kwargs['num_heads']
    num_k_heads = kwargs.get('num_kv_heads', num_heads)
    num_v_heads = kwargs.get('num_kv_heads', kwargs.get('num_v_heads', num_heads))
    head_v_dim = int(head_dim * kwargs.get('expand_v', 1))
    num_expanded_heads = max(num_heads, num_v_heads)
    assert layer.q_proj.out_features == num_heads * head_dim
    assert layer.k_proj.out_features == num_k_heads * head_dim
    assert layer.v_proj.out_features == num_v_heads * head_v_dim
    if layer.use_forgetting_gate:
        assert layer.a_proj.out_features == num_expanded_heads
    assert layer.o_proj.in_features == num_expanded_heads * head_v_dim

    x = torch.randn(2, 100, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    o, _, _ = layer(x)
    assert o.shape == x.shape
    o.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def _repeat_heads(w: torch.Tensor, num_heads: int, groups: int) -> torch.Tensor:
    """Repeats the per-head rows of a projection (or depthwise conv) weight, as the layer repeats heads."""
    return w.view(num_heads, -1, *w.shape[1:]).repeat_interleave(groups, dim=0).reshape(-1, *w.shape[1:])


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(dict(num_heads=2, num_v_heads=4), id="gva"),
        pytest.param(dict(num_heads=4, num_kv_heads=2), id="gqa"),
    ],
)
def test_gka_grouped_heads_match_repeated_weights(kwargs: dict):
    """A grouped layer must equal an equal-heads layer whose q/k/v weights are repeated per head."""
    torch.manual_seed(42)
    head_dim, hidden_size = 64, 128
    grouped = GatedKalmaNet(hidden_size=hidden_size, head_dim=head_dim, **kwargs).to(device).eval()
    full = GatedKalmaNet(hidden_size=hidden_size, head_dim=head_dim, num_heads=4).to(device).eval()

    state = grouped.state_dict()
    for name, num in (('q', grouped.num_heads), ('k', grouped.num_k_heads), ('v', grouped.num_v_heads)):
        groups = 4 // num
        state[f'{name}_proj.weight'] = _repeat_heads(state[f'{name}_proj.weight'], num, groups)
        state[f'{name}_conv1d.weight'] = _repeat_heads(state[f'{name}_conv1d.weight'], num, groups)
    full.load_state_dict(state)

    # not bit-exact (different projection shapes round differently under TF32); a wrong head order gives O(1) errors
    x = torch.randn(2, 100, hidden_size, device=device)
    with torch.no_grad():
        assert_close('o', full(x)[0], grouped(x)[0], 0.02)


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        pytest.param(dict(num_heads=4, num_v_heads=8, num_kv_heads=2), 'cannot be set together', id="gva-and-gqa"),
        pytest.param(dict(num_heads=4, num_kv_heads=3), 'divisible by num_kv_heads', id="gqa-not-divisible"),
        pytest.param(dict(num_heads=4, num_v_heads=6), 'divisible by num_heads', id="gva-not-divisible"),
        pytest.param(dict(num_heads=4, expand_v=0.3), 'integer value head dim', id="expand_v"),
        pytest.param(dict(num_heads=4, mode='parallel'), 'Not supported mode', id="mode"),
        pytest.param(dict(num_heads=4, num_iter=0), 'must be at least 1', id="num_iter"),
    ],
)
def test_gka_invalid_config(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        GatedKalmaNet(hidden_size=128, head_dim=64, **kwargs)


def test_gka_chunk_matches_fused_recurrent():
    """The `chunk` and `fused_recurrent` layers must match on the same weights."""
    torch.manual_seed(42)
    kwargs = dict(hidden_size=128, head_dim=64, num_heads=2, num_v_heads=4)
    chunk = GatedKalmaNet(**kwargs, mode='chunk').to(device).eval()
    recurrent = GatedKalmaNet(**kwargs, mode='fused_recurrent').to(device).eval()
    recurrent.load_state_dict(chunk.state_dict())

    # longer than 64 tokens, so each layer runs its configured mode
    x = torch.randn(2, 100, 128, device=device)
    with torch.no_grad():
        assert_close('o', chunk(x)[0], recurrent(x)[0], 0.02)


def test_gka_decode_with_cache_matches_full_forward():
    """Prefill then token-by-token decoding with a `Cache` must match one full forward."""
    torch.manual_seed(42)
    layer = GatedKalmaNet(hidden_size=128, head_dim=64, num_heads=4, layer_idx=0).to(device).eval()
    x = torch.randn(2, 90, 128, device=device)

    with torch.no_grad():
        full = layer(x)[0]
        cache = Cache()
        outputs = [layer(x[:, :70], past_key_values=cache, use_cache=True)[0]]
        for t in range(70, 90):
            outputs.append(layer(x[:, t:t + 1], past_key_values=cache, use_cache=True)[0])

    assert_close('o', full, torch.cat(outputs, 1), 0.01)


def test_gka_eval_with_grad_uses_chunk():
    torch.manual_seed(42)
    layer = GatedKalmaNet(hidden_size=128, head_dim=64, num_heads=4, mode='fused_recurrent').to(device).eval()
    x = torch.randn(1, 32, 128, device=device, requires_grad=True)

    with mock.patch(
        'fla.layers.gka.fused_recurrent_gka',
        side_effect=AssertionError("eval with autograd must not use the inference-only fused recurrent kernel"),
    ) as fused_recurrent:
        layer(x)[0].sum().backward()

    fused_recurrent.assert_not_called()
    assert torch.isfinite(x.grad).all()
