# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.layers import CyclicFlowAttention
from fla.models.cache_utils import Cache
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='CyFA kernels currently require NVIDIA GPUs')


@pytest.mark.parametrize('use_short_conv', [False, True])
def test_layer(use_short_conv):
    torch.manual_seed(42)
    layer = CyclicFlowAttention(
        hidden_size=256, num_heads=4, head_dim=64, num_slots=128, use_short_conv=use_short_conv, layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(2, 256, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)[0]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
def test_layer_eval_with_grad_uses_chunk(mode):
    torch.manual_seed(42)
    layer = CyclicFlowAttention(
        hidden_size=256, num_heads=4, head_dim=64, num_slots=128, mode=mode, layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    with mock.patch(
        'fla.layers.cyfa.fused_recurrent_cyfa',
        side_effect=AssertionError("eval with autograd must use chunk mode"),
    ) as fused_recurrent:
        output = layer(x)[0]
        output.sum().backward()
    fused_recurrent.assert_not_called()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@torch.no_grad()
def test_cache_layer_idx():
    layer = CyclicFlowAttention(hidden_size=64, num_heads=1, head_dim=64, num_slots=32).to(device).eval()
    with pytest.raises(ValueError, match='layer_idx'):
        layer(torch.randn(1, 1, 64, device=device), past_key_values=Cache(), use_cache=True)


@torch.no_grad()
def test_cached_multi_token_attention_mask():
    torch.manual_seed(42)
    layer = CyclicFlowAttention(
        hidden_size=64, num_heads=1, head_dim=64, num_slots=32, use_short_conv=False, layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    prefix = torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16)
    caches = [Cache(), Cache()]
    for cache in caches:
        layer(prefix, past_key_values=cache, use_cache=True)

    x = torch.randn(2, 2, 64, device=device, dtype=torch.bfloat16)
    attention_mask = torch.tensor([[0, 1], [1, 1]], device=device)
    actual = layer(x, attention_mask=attention_mask, past_key_values=caches[0], use_cache=True)[0]
    packed = torch.cat((x[0, 1:], x[1]), dim=0).unsqueeze(0)
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    expected = layer(packed, past_key_values=caches[1], use_cache=True, cu_seqlens=cu_seqlens)[0]
    assert_close('o', expected.squeeze(0).float(), actual[attention_mask.bool()].float(), 1e-6)


@pytest.mark.parametrize('use_short_conv', [False, True])
@torch.no_grad()
def test_cached_single_token_attention_mask(use_short_conv):
    torch.manual_seed(42)
    layer = CyclicFlowAttention(
        hidden_size=64, num_heads=1, head_dim=64, num_slots=32, use_short_conv=use_short_conv, layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    prefix = torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16)
    caches = [Cache(), Cache(), Cache()]
    for cache in caches:
        layer(prefix, past_key_values=cache, use_cache=True)

    x = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    attention_mask[0, -1] = False
    actual = layer(x, attention_mask=attention_mask, past_key_values=caches[0], use_cache=True)[0]
    cu_seqlens = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    expected = layer(x[1:], past_key_values=caches[1], use_cache=True, cu_seqlens=cu_seqlens)[0]
    assert_close('o', expected.squeeze(0).float(), actual[1].float(), 1e-6)
    assert torch.count_nonzero(actual[0]) == 0

    for name in ('recurrent_state', 'conv_state'):
        actual_state, packed_state, skipped_state = (cache[0][name] for cache in caches)
        if actual_state is None:
            continue
        for actual, packed, skipped in zip(actual_state, packed_state, skipped_state):
            assert_close(f'{name} masked', skipped[0], actual[0], 1e-6)
            assert_close(f'{name} valid', packed[1], actual[1], 1e-6)
    assert not torch.equal(caches[0][0]['recurrent_state'][2][1], caches[2][0]['recurrent_state'][2][1])


@torch.no_grad()
def test_cached_single_token_full_attention_mask():
    torch.manual_seed(42)
    layer = CyclicFlowAttention(
        hidden_size=64, num_heads=1, head_dim=64, num_slots=32, use_short_conv=False, layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    prefix = torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16)
    caches = [Cache(), Cache()]
    for cache in caches:
        layer(prefix, past_key_values=cache, use_cache=True)

    x = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    with mock.patch(
        'fla.layers.cyfa.get_unpad_data',
        side_effect=AssertionError('full single-token masks must stay dense'),
    ) as unpad:
        actual = layer(x, attention_mask=attention_mask, past_key_values=caches[0], use_cache=True)[0]
    expected = layer(x, past_key_values=caches[1], use_cache=True)[0]
    unpad.assert_not_called()
    assert_close('o', expected.float(), actual.float(), 1e-6)
    for actual_state, expected_state in zip(caches[0][0]['recurrent_state'], caches[1][0]['recurrent_state']):
        assert_close('state', expected_state.float(), actual_state.float(), 1e-6)


@pytest.mark.parametrize('mask', ['mixed', 'all_zero'])
def test_cached_single_token_attention_mask_with_grad(mask):
    torch.manual_seed(42)
    layer = CyclicFlowAttention(
        hidden_size=64, num_heads=1, head_dim=64, num_slots=32, use_short_conv=False, layer_idx=0,
    ).to(device=device, dtype=torch.bfloat16).eval()
    cache = Cache()
    with torch.no_grad():
        layer(torch.randn(2, 3, 64, device=device, dtype=torch.bfloat16), past_key_values=cache, use_cache=True)

    x = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    attention_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    if mask == 'mixed':
        attention_mask[0, -1] = False
    else:
        attention_mask.zero_()
    with (
        mock.patch('fla.layers.cyfa.get_unpad_data', side_effect=AssertionError('grad must stay dense')) as unpad,
        mock.patch('fla.layers.cyfa.fused_recurrent_cyfa', side_effect=AssertionError('grad must use chunk')) as fused,
    ):
        output = layer(x, attention_mask=attention_mask, past_key_values=cache, use_cache=True)[0]
        output.sum().backward()
    unpad.assert_not_called()
    fused.assert_not_called()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
