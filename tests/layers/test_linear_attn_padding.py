# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.linear_attn import LinearAttention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('length', [65, 129])
@pytest.mark.parametrize('padding_length', [64, 128])
def test_linear_attn_left_padding(dtype, length, padding_length):
    # aligned padding preserves the reduction layout of the compact chunk reference
    torch.manual_seed(42)
    layer = LinearAttention(
        hidden_size=128,
        num_heads=2,
        feature_map='elu',
        do_feature_map_norm=True,
        output_norm='identity',
        mode='chunk',
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    x = torch.randn(1, length, 128, device=device, dtype=dtype)
    padding = torch.randn(1, padding_length, 128, device=device, dtype=dtype)
    padded = torch.cat([padding, x], dim=1).requires_grad_()
    mask = torch.ones(1, length + padding_length, device=device, dtype=torch.long)
    mask[:, :padding_length] = 0
    with torch.no_grad():
        expected = layer(x)[0]
        actual = layer(padded, attention_mask=mask)[0][:, padding_length:]
        _, _, expected_cache = layer(x, past_key_values=Cache(), use_cache=True)
        _, _, actual_cache = layer(padded, attention_mask=mask, past_key_values=Cache(), use_cache=True)
        next_x = torch.randn(1, 1, 128, device=device, dtype=dtype)
        expected_next = layer(next_x, past_key_values=expected_cache, use_cache=True)[0]
        next_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        actual_next = layer(next_x, attention_mask=next_mask, past_key_values=actual_cache, use_cache=True)[0]
    assert_close('padded inference', expected, actual, 1e-3)
    assert_close('padded cached decode', expected_next, actual_next, 1e-3)
    expected_x = x.clone().requires_grad_()
    expected = layer(expected_x)[0]
    actual = layer(padded, attention_mask=mask)[0][:, padding_length:]
    do = torch.randn_like(expected)
    parameters = tuple(layer.parameters())
    expected_dx, *expected_dp = torch.autograd.grad((expected * do).sum(), (expected_x, *parameters))
    actual_dx, *actual_dp = torch.autograd.grad((actual * do).sum(), (padded, *parameters))
    assert_close('padded training', expected, actual, 1e-3)
    assert_close('padded input gradient', expected_dx, actual_dx[:, padding_length:], 1e-3)
    assert torch.count_nonzero(actual_dx[:, :padding_length]) == 0
    for (name, _), expected_grad, actual_grad in zip(layer.named_parameters(), expected_dp, actual_dp):
        assert_close(f'padded {name} gradient', expected_grad, actual_grad, 1e-3)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode', ['chunk', 'fused_chunk', 'fused_recurrent'])
@pytest.mark.parametrize('feature_map', ['elu', 'elementwise_product'])
@pytest.mark.parametrize('norm_k', [False, True])
def test_linear_attn_padding_values_do_not_affect_valid_tokens(dtype, mode, feature_map, norm_k):
    torch.manual_seed(42)
    layer = LinearAttention(
        hidden_size=128,
        num_heads=2,
        feature_map=feature_map,
        norm_k=norm_k,
        do_feature_map_norm=True,
        output_norm='identity',
        mode=mode,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    x = torch.randn(2, 129, 128, device=device, dtype=dtype)
    mask = torch.ones(2, 129, device=device, dtype=torch.bool)
    mask[0, :7] = False
    mask[1, -31:] = False
    changed = x.clone()
    changed[~mask] = torch.randn_like(changed[~mask])
    with torch.no_grad():
        expected, _, expected_cache = layer(x, attention_mask=mask, past_key_values=Cache(), use_cache=True)
        actual, _, actual_cache = layer(changed, attention_mask=mask, past_key_values=Cache(), use_cache=True)
        for actual_state, expected_state in zip(actual_cache[0]['recurrent_state'], expected_cache[0]['recurrent_state']):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        next_x = torch.randn(2, 1, 128, device=device, dtype=dtype)
        next_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        expected_next = layer(next_x, attention_mask=next_mask, past_key_values=expected_cache, use_cache=True)[0]
        actual_next = layer(next_x, attention_mask=next_mask, past_key_values=actual_cache, use_cache=True)[0]
        torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)
    torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=0)
    x.requires_grad_()
    changed.requires_grad_()
    do = torch.randn_like(expected[mask])
    expected = layer(x, attention_mask=mask)[0][mask]
    actual = layer(changed, attention_mask=mask)[0][mask]
    parameters = tuple(layer.parameters())
    expected_dx, *expected_dp = torch.autograd.grad((expected * do).sum(), (x, *parameters))
    actual_dx, *actual_dp = torch.autograd.grad((actual * do).sum(), (changed, *parameters))
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
    assert torch.count_nonzero(actual_dx[~mask]) == 0
    for actual_grad, expected_grad in zip(actual_dp, expected_dp):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode', ['chunk', 'fused_chunk', 'fused_recurrent'])
def test_linear_attn_fully_padded(dtype, mode):
    torch.manual_seed(42)
    layer = LinearAttention(
        hidden_size=128,
        num_heads=2,
        feature_map='elu',
        do_feature_map_norm=True,
        output_norm='identity',
        mode=mode,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    x = torch.randn(1, 129, 128, device=device, dtype=dtype, requires_grad=True)
    mask = torch.zeros(1, 129, device=device, dtype=torch.long)
    with torch.no_grad():
        output, _, cache = layer(x, attention_mask=mask, past_key_values=Cache(), use_cache=True)
        assert torch.count_nonzero(output) == 0
        for state in cache[0]['recurrent_state']:
            assert torch.count_nonzero(state) == 0
        next_x = torch.randn(1, 1, 128, device=device, dtype=dtype)
        expected = layer(next_x)[0]
        next_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        actual = layer(next_x, attention_mask=next_mask, past_key_values=cache, use_cache=True)[0]
    assert_close('empty-prefix cached decode', expected, actual, 1e-3)
    output = layer(x, attention_mask=mask)[0]
    gradients = torch.autograd.grad(output.float().sum(), (x, *layer.parameters()))
    for gradient in gradients:
        assert torch.count_nonzero(gradient) == 0
