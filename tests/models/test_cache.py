# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pickle

import pytest
import torch

from fla.models.cache_utils import Cache, FLACache, FLALayer, LegacyFLACache
from fla.utils import device_name


@pytest.fixture(params=[FLACache, LegacyFLACache], ids=['layered', 'legacy'])
def cache(request):
    return request.param()


def make_attention_state(length: int, offset: int = 0):
    # token positions make window eviction visible in the cached values
    positions = torch.arange(offset, offset + length, dtype=torch.float32, device=device_name).view(1, length, 1, 1)
    keys = positions.expand(2, length, 2, 3).contiguous()
    return keys, keys + 1000


@pytest.mark.parametrize('layer_idx', [0, 5, None], ids=['first', 'missing', 'default'])
def test_get_seq_length(layer_idx):
    cache = FLACache()
    assert cache.get_seq_length(layer_idx=layer_idx) == 0

    cache.update(attn_state=make_attention_state(length=10))
    assert cache.get_seq_length(layer_idx=layer_idx) == (0 if layer_idx == 5 else 10)


@pytest.mark.parametrize('num_layers', [4, 8], ids=['4-layers', '8-layers'])
def test_update_layers(num_layers):
    """Layer counters are independent: https://github.com/fla-org/flash-linear-attention/issues/747."""
    cache = FLACache()
    lengths = [layer_idx * 10 for layer_idx in range(num_layers)]
    for layer_idx, length in enumerate(lengths):
        assert cache.get_seq_length(layer_idx=layer_idx) == 0
        cache.update(attn_state=make_attention_state(length=length), layer_idx=layer_idx)
        assert cache.get_seq_length(layer_idx=layer_idx) == length
        assert cache.get_seq_length(layer_idx=layer_idx + 1) == 0

    cache.update(attn_state=make_attention_state(length=5), layer_idx=0)
    assert [cache.get_seq_length(layer_idx=i) for i in range(num_layers)] == [5, *lengths[1:]]


@pytest.mark.parametrize(
    ('num_layers', 'prefill_length', 'update_length', 'num_steps'),
    [
        pytest.param(4, 0, 1, 5, id='decode-from-empty'),
        pytest.param(8, 0, 1, 10, id='decode-many-layers'),
        pytest.param(4, 5, 1, 3, id='decode-after-prefill'),
        pytest.param(8, 10, 1, 3, id='decode-long-prefill'),
        pytest.param(4, 5, 3, 3, id='chunk-after-prefill'),
        pytest.param(4, 0, 5, 4, id='chunk-from-empty'),
        pytest.param(2, 0, 10, 3, id='chunk-two-layers'),
    ],
)
def test_update_decode(cache, num_layers, prefill_length, update_length, num_steps):
    """Layers see the same pre-update RoPE offset: https://github.com/fla-org/flash-linear-attention/issues/766."""
    if prefill_length:
        for layer_idx in range(num_layers):
            cache.update(attn_state=make_attention_state(length=prefill_length), layer_idx=layer_idx, offset=prefill_length)

    for step in range(num_steps):
        past_length = prefill_length + step * update_length
        for layer_idx in range(num_layers):
            assert cache.get_seq_length(layer_idx=layer_idx) == past_length
            cache.update(
                attn_state=make_attention_state(length=update_length, offset=past_length),
                layer_idx=layer_idx,
                offset=update_length,
            )
        assert [cache.get_seq_length(layer_idx=i) for i in range(num_layers)] == [past_length + update_length] * num_layers


@pytest.mark.parametrize('offset', [None, 0, 3], ids=['default', 'zero', 'explicit'])
def test_update_offset(cache, offset):
    state = torch.ones(2, 2, 3, device=device_name)
    cache.update(recurrent_state=state, offset=offset)
    assert cache.get_seq_length() == (1 if offset is None else offset)
    assert cache[0]['recurrent_state'] is state


@pytest.mark.parametrize('with_recurrent_state', [False, True], ids=['empty', 'recurrent'])
@pytest.mark.parametrize(
    ('prefill_length', 'update_length', 'initial_window_size', 'window_size'),
    [
        pytest.param(4, 6, 4, 4, id='replace-window'),
        pytest.param(2, 4, 4, 4, id='overflow'),
        pytest.param(7, 2, 4, 4, id='oversized-prefill'),
        pytest.param(4, 1, 4, 4, id='decode'),
        pytest.param(4, 0, 4, 4, id='empty-update'),
        pytest.param(0, 2, 4, 4, id='empty-prefill'),
        pytest.param(1, 1, 4, 4, id='partial-window'),
        pytest.param(2, 3, 1, 1, id='window-one'),
        pytest.param(2, 3, None, None, id='unbounded'),
        pytest.param(6, 1, None, 4, id='enable-window'),
        pytest.param(6, 1, 8, 4, id='shrink-window'),
    ],
)
def test_update_attention(cache, with_recurrent_state, prefill_length, update_length, initial_window_size, window_size):
    recurrent_state = torch.ones(2, 2, 3, device=device_name) if with_recurrent_state else None
    if with_recurrent_state:
        cache.update(recurrent_state=recurrent_state, offset=0)

    prefill = tuple(state.requires_grad_() for state in make_attention_state(length=prefill_length))
    update = tuple(state.requires_grad_() for state in make_attention_state(length=update_length, offset=prefill_length))
    cache.update(attn_state=prefill, offset=prefill_length, cache_kwargs={'window_size': initial_window_size})
    assert cache.get_seq_length() == prefill_length
    prefill_start = 0 if initial_window_size is None else max(0, prefill_length - initial_window_size)
    torch.testing.assert_close(cache[0]['attn_state'], tuple(state[:, prefill_start:] for state in prefill))

    cache.update(attn_state=update, offset=update_length, cache_kwargs={'window_size': window_size})
    total_length = prefill_length + update_length
    start = 0 if window_size is None else max(0, total_length - window_size)
    torch.testing.assert_close(cache[0]['attn_state'], make_attention_state(length=total_length - start, offset=start))
    assert cache.get_seq_length() == total_length
    assert cache[0]['recurrent_state'] is recurrent_state

    sum(state.sum() for state in cache[0]['attn_state']).backward()
    for states, offset, length in ((prefill, 0, prefill_length), (update, prefill_length, update_length)):
        expected = (torch.arange(offset, offset + length, device=device_name) >= start).float().view(1, length, 1, 1)
        for state in states:
            grad = torch.zeros_like(state) if state.grad is None else state.grad
            torch.testing.assert_close(grad, expected.expand_as(state), rtol=0, atol=0)


@pytest.mark.parametrize('operation', ['reorder_cache', 'batch_repeat_interleave', 'batch_select_indices'])
def test_batch_operations(operation):
    cache = FLACache()
    state = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    cache.update(recurrent_state=state, conv_state=(state + 20, [state + 40]), offset=5)
    cache.update(attn_state=(state[:, :, None], state[:, :, None] + 60), layer_idx=1, offset=2)

    indices = torch.tensor([2, 0, 2])
    if operation == 'batch_repeat_interleave':
        cache.batch_repeat_interleave(repeats=2)
        indices = torch.arange(3).repeat_interleave(2)
    else:
        getattr(cache, operation)(indices)

    torch.testing.assert_close(cache[0]['recurrent_state'], state[indices])
    torch.testing.assert_close(cache[0]['conv_state'], (state[indices] + 20, [state[indices] + 40]))
    torch.testing.assert_close(cache[1]['attn_state'], (state[indices, :, None], state[indices, :, None] + 60))
    assert cache.get_seq_length(layer_idx=0) == 5
    assert cache.get_seq_length(layer_idx=1) == 2


@pytest.mark.parametrize('max_length', [5, -2], ids=['absolute', 'relative'])
def test_crop(max_length):
    cache = FLACache()
    state = make_attention_state(length=7)
    cache.update(attn_state=state)
    cache.crop(max_length=max_length)

    torch.testing.assert_close(cache[0]['attn_state'], tuple(tensor[:, :5] for tensor in state))
    assert cache.get_seq_length() == 5
    assert cache.get_mask_sizes(query_length=3, layer_idx=0) == (8, 0)


def test_crop_recurrent():
    cache = FLACache()
    attention_state = make_attention_state(length=4)
    recurrent_state = torch.ones(2, 1, 3, 3, device=device_name)
    cache.update(attn_state=attention_state)
    cache.update(recurrent_state=recurrent_state, layer_idx=1, offset=4)

    with pytest.raises(NotImplementedError, match='rollback'):
        cache.crop(max_length=2)
    assert cache[0]['attn_state'][0] is attention_state[0]
    assert cache[1]['recurrent_state'] is recurrent_state
    assert cache.get_seq_length(layer_idx=0) == cache.get_seq_length(layer_idx=1) == 4


def test_crop_sliding_window():
    cache = FLACache()
    cache.update(attn_state=make_attention_state(length=7), cache_kwargs={'window_size': 4})
    assert cache.get_mask_sizes(query_length=3, layer_idx=0) == (6, 4)

    with pytest.raises(NotImplementedError, match='sliding-window'):
        cache.crop(max_length=5)
    assert cache.get_seq_length() == 7


@pytest.mark.parametrize(('cache_class', 'expected_states'), [(FLACache, [None, None]), (LegacyFLACache, [])])
def test_reset(cache_class, expected_states):
    cache = cache_class()
    for layer_idx in range(2):
        cache.update(attn_state=make_attention_state(length=3, offset=layer_idx * 10), layer_idx=layer_idx, offset=3)
    assert cache.get_seq_length() == 3

    cache.reset()
    assert cache.get_seq_length() == 0
    assert list(cache) == expected_states


@pytest.mark.parametrize('state', [None, {}, {'recurrent_state': None}], ids=['uninitialized', 'empty', 'none-value'])
def test_layer_offload_empty(state):
    layer = FLALayer()
    layer.state = state
    layer.offload()
    layer.prefetch()
    assert layer.state == state


@pytest.mark.parametrize(
    ('state_name', 'as_tuple'),
    [('recurrent_state', False), ('recurrent_state', True), ('attn_state', True), ('conv_state', True), ('ffn_state', False)],
    ids=['recurrent', 'composite-recurrent', 'attention', 'convolution', 'feed-forward'],
)
def test_layer_offload(state_name, as_tuple):
    layer = FLALayer()
    tensor = torch.arange(12, dtype=torch.float32, device=device_name).reshape(1, 4, 1, 3)
    state = (tensor, tensor + 1) if as_tuple else tensor
    layer.update(**{state_name: state})

    for action, expected_device in ((layer.offload, torch.device('cpu')), (layer.prefetch, tensor.device)):
        action()
        restored = layer.state[state_name]
        tensors = restored if as_tuple else (restored,)
        expected = state if as_tuple else (state,)
        for actual, original in zip(tensors, expected, strict=True):
            assert actual.device == expected_device
            torch.testing.assert_close(actual.cpu(), original.cpu())


def test_layer_reset():
    layer = FLALayer()
    layer.update(attn_state=make_attention_state(length=5))
    assert layer.state is not None
    assert layer.get_seq_length() == 5

    layer.reset()
    assert layer.state is None
    assert layer.get_seq_length() == 0

    state = make_attention_state(length=2, offset=20)
    layer.update(attn_state=state)
    torch.testing.assert_close(layer.state['attn_state'], state)
    assert layer.get_seq_length() == 2


@pytest.mark.parametrize('container', [list, tuple], ids=['list', 'tuple'])
@pytest.mark.parametrize('seen_tokens', [0, 17], ids=['zero-tokens', 'seen-tokens'])
def test_legacy_conversion(cache, container, seen_tokens):
    cache.update(attn_state=make_attention_state(length=4, offset=10), offset=4)
    legacy_state = container(cache.to_legacy_cache())
    restored = type(cache).from_legacy_cache(past_key_values=legacy_state, seen_tokens=seen_tokens)

    torch.testing.assert_close(restored.to_legacy_cache(), tuple(legacy_state))
    assert restored.get_seq_length() == seen_tokens


@pytest.mark.parametrize('seen_tokens', [0, 5], ids=['zero-tokens', 'seen-tokens'])
def test_legacy_conversion_offload(seen_tokens):
    cache = FLACache()
    state = torch.arange(24, dtype=torch.float32, device=device_name).reshape(2, 3, 4)
    cache.update(recurrent_state=state, offset=seen_tokens)
    restored = FLACache.from_legacy_cache(past_key_values=cache.to_legacy_cache(), seen_tokens=seen_tokens)
    restored.layers[0].offload()
    restored.layers[0].prefetch()

    assert restored[0]['recurrent_state'].device == state.device
    torch.testing.assert_close(restored[0]['recurrent_state'], state)
    assert restored.get_seq_length() == seen_tokens


@pytest.mark.parametrize('module_name', ['fla.models.utils', 'fla.models.cache_utils'], ids=['legacy', 'canonical'])
@pytest.mark.parametrize('cache_class', [Cache, FLACache, LegacyFLACache])
def test_pickle(monkeypatch, module_name, cache_class):
    cache = cache_class()
    recurrent_state = (torch.arange(24).reshape(1, 2, 3, 4), torch.arange(6).reshape(1, 2, 3))
    cache.update(recurrent_state=recurrent_state, attn_state=make_attention_state(length=4, offset=10), offset=4)

    with monkeypatch.context() as patch:
        for cls in (Cache, FLACache, FLALayer, LegacyFLACache):
            patch.setattr(cls, '__module__', module_name)
        checkpoint = pickle.dumps(cache)
    restored = pickle.loads(checkpoint)

    assert type(restored) is cache_class
    torch.testing.assert_close(restored[0]['recurrent_state'], recurrent_state)
    torch.testing.assert_close(restored[0]['attn_state'], make_attention_state(length=4, offset=10))
    assert restored.get_seq_length() == 4

    restored.update(attn_state=make_attention_state(length=2, offset=14), offset=2)
    torch.testing.assert_close(restored[0]['recurrent_state'], recurrent_state)
    torch.testing.assert_close(restored[0]['attn_state'], make_attention_state(length=6, offset=10))
    assert restored.get_seq_length() == 6
