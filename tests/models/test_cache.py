# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect
import pickle

import pytest
import torch
from transformers.cache_utils import Cache as HFCacheBase

from fla.models.cache_utils import Cache, FLACache, FLALayer, LegacyFLACache
from fla.utils import device_name

_HAS_LAYER_CACHE = bool({'layer_class_to_replicate', 'layer_classes'} & inspect.signature(HFCacheBase).parameters.keys())


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(42)


# encode token positions into tensor values so window tail assertions are exact
def _make_attn_state(start: int, length: int, *, batch_size: int = 1, num_heads: int = 2, head_dim: int = 3):
    token_ids = torch.arange(start, start + length, dtype=torch.float32, device=device_name).view(1, length, 1, 1)
    key_states = token_ids.expand(batch_size, length, num_heads, head_dim).contiguous()
    value_states = (token_ids + 1000).expand(batch_size, length, num_heads, head_dim).contiguous()
    return key_states, value_states


def _assert_attn_state_tokens(attn_state, expected_tokens: torch.Tensor):
    expected = expected_tokens.to(device=device_name, dtype=torch.float32).view(1, -1, 1, 1)
    expected_key = expected.expand_as(attn_state[0])
    expected_value = (expected + 1000).expand_as(attn_state[1])

    torch.testing.assert_close(attn_state[0], expected_key)
    torch.testing.assert_close(attn_state[1], expected_value)


def _new_cache(cache_class):
    if issubclass(cache_class, FLACache) != _HAS_LAYER_CACHE:
        pytest.skip(f'{cache_class.__name__} requires a different Transformers cache constructor.')
    return cache_class()


def _raw_legacy_cache():
    cache = object.__new__(LegacyFLACache)
    cache.states = []
    cache._seen_tokens = 0
    return cache


@pytest.mark.parametrize(
    ['num_layers', 'batch_size', 'seq_len', 'hidden_size', 'num_heads'],
    [
        pytest.param(*test, id=f"L{test[0]}-B{test[1]}-T{test[2]}-D{test[3]}-H{test[4]}")
        for test in [
            (4, 2, 10, 64, 4),
            (8, 1, 20, 128, 8),
        ]
    ],
)
def test_cache_per_layer_seq_length(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
):
    """Each layer advances its own token counter: https://github.com/fla-org/flash-linear-attention/issues/747."""
    cache = _new_cache(FLACache)
    head_dim = hidden_size // num_heads

    for layer_idx in range(num_layers):
        assert cache.get_seq_length(layer_idx) == 0

        key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
        attn_state = (key_states, value_states)

        cache.update(
            attn_state=attn_state,
            layer_idx=layer_idx,
        )

        assert cache.get_seq_length(layer_idx) == seq_len, \
            f"Layer {layer_idx} should have seq_length={seq_len}, got {cache.get_seq_length(layer_idx)}"

        if layer_idx + 1 < num_layers:
            assert cache.get_seq_length(layer_idx + 1) == 0, \
                f"Layer {layer_idx + 1} should have seq_length=0 before being updated"

    for layer_idx in range(num_layers):
        assert cache.get_seq_length(layer_idx) == seq_len, \
            f"Layer {layer_idx} should have seq_length={seq_len}"


@pytest.mark.parametrize(
    ['num_layers', 'batch_size', 'chunk_size', 'num_chunks', 'hidden_size', 'num_heads'],
    [
        pytest.param(*test, id=f"L{test[0]}-B{test[1]}-chunk{test[2]}-n{test[3]}-D{test[4]}-H{test[5]}")
        for test in [
            (4, 1, 5, 4, 64, 4),
            (2, 2, 10, 3, 128, 8),
        ]
    ],
)
def test_cache_incremental_update(
    num_layers: int,
    batch_size: int,
    chunk_size: int,
    num_chunks: int,
    hidden_size: int,
    num_heads: int,
):
    cache = _new_cache(FLACache)
    head_dim = hidden_size // num_heads

    for chunk_idx in range(num_chunks):
        for layer_idx in range(num_layers):
            key_states = torch.randn(batch_size, chunk_size, num_heads, head_dim, device=device_name)
            value_states = torch.randn(batch_size, chunk_size, num_heads, head_dim, device=device_name)
            attn_state = (key_states, value_states)

            cache.update(
                attn_state=attn_state,
                layer_idx=layer_idx,
            )

            expected_seq_len = (chunk_idx + 1) * chunk_size
            actual_seq_len = cache.get_seq_length(layer_idx)
            assert actual_seq_len == expected_seq_len, \
                f"Layer {layer_idx} after chunk {chunk_idx} should have seq_length={expected_seq_len}, got {actual_seq_len}"


def test_cache_get_seq_length_nonexistent_layer():
    """Missing layers report zero; a `None` index addresses the first layer."""
    cache = _new_cache(FLACache)

    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(5) == 0
    assert cache.get_seq_length(None) == 0

    key_states = torch.randn(1, 10, 4, 16, device=device_name)
    value_states = torch.randn(1, 10, 4, 16, device=device_name)
    cache.update(attn_state=(key_states, value_states), layer_idx=0)

    assert cache.get_seq_length(None) == 10
    assert cache.get_seq_length(0) == 10


def test_legacy_cache_update_defaults_none_offset():
    torch.manual_seed(42)
    cache = _raw_legacy_cache()
    state = (torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1))

    cache.update(attn_state=state, offset=None)

    assert cache.get_seq_length(0) == 1


def test_cache_window_size_does_not_undercount():
    """Window eviction preserves the total processed-token count."""
    cache = _new_cache(FLACache)
    batch_size, seq_len, num_heads, head_dim = 1, 100, 4, 16
    window_size = 10

    key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
    value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)

    cache.update(
        attn_state=(key_states, value_states),
        layer_idx=0,
        cache_kwargs={"window_size": window_size}
    )

    assert cache.get_seq_length(0) == seq_len, \
        f"Expected seq_length={seq_len}, got {cache.get_seq_length(0)} (window_size={window_size})"


@pytest.mark.parametrize("cache_class", [FLACache, LegacyFLACache])
@pytest.mark.parametrize(
    ["first_len", "second_len", "window_size"],
    [
        pytest.param(4, 6, 4, id="full-window-oversized-update"),
        pytest.param(2, 4, 4, id="partial-window-overflow"),
    ],
)
def test_cache_window_overflow_keeps_tail(cache_class, first_len: int, second_len: int, window_size: int):
    cache = _new_cache(cache_class)

    first_state = _make_attn_state(0, first_len)
    second_state = _make_attn_state(first_len, second_len)

    cache.update(
        attn_state=first_state,
        layer_idx=0,
        offset=first_len,
        cache_kwargs={"window_size": window_size},
    )
    cache.update(
        attn_state=second_state,
        layer_idx=0,
        offset=second_len,
        cache_kwargs={"window_size": window_size},
    )

    expected_tokens = torch.arange(first_len + second_len - window_size, first_len + second_len)
    state = cache[0]
    assert state["attn_state"][0].shape[1] == window_size
    _assert_attn_state_tokens(state["attn_state"], expected_tokens)
    assert cache.get_seq_length(0) == first_len + second_len


def test_legacy_cache_window_overflow_without_hf_init():
    cache = _raw_legacy_cache()
    window_size = 4
    first_len = 2
    second_len = 4

    cache.update(
        attn_state=_make_attn_state(0, first_len),
        layer_idx=0,
        offset=first_len,
        cache_kwargs={"window_size": window_size},
    )
    cache.update(
        attn_state=_make_attn_state(first_len, second_len),
        layer_idx=0,
        offset=second_len,
        cache_kwargs={"window_size": window_size},
    )

    _assert_attn_state_tokens(
        cache[0]["attn_state"], torch.arange(first_len + second_len - window_size, first_len + second_len)
    )
    assert cache.get_seq_length(0) == first_len + second_len


@pytest.mark.parametrize('state', [None, {}, {'recurrent_state': None}], ids=['uninitialized', 'empty', 'none-value'])
def test_fla_layer_empty_state_offload_and_prefetch(state):
    layer = FLALayer()
    layer.state = state
    layer.offload()
    layer.prefetch()
    assert layer.state is state


@pytest.mark.parametrize(
    ('state_name', 'as_tuple'),
    [('recurrent_state', False), ('recurrent_state', True), ('attn_state', True), ('conv_state', True), ('ffn_state', False)],
    ids=['recurrent', 'composite-recurrent', 'attention', 'convolution', 'feed-forward'],
)
def test_fla_layer_offload_and_prefetch(state_name, as_tuple):
    layer = FLALayer()
    layer.offload()
    layer.prefetch()
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


def test_fla_layer_reset():
    layer = FLALayer()
    layer.update(attn_state=_make_attn_state(0, 5))

    assert layer.state is not None
    assert layer.get_seq_length() == 5

    layer.reset()

    assert layer.state is None
    assert layer.get_seq_length() == 0

    layer.update(attn_state=_make_attn_state(20, 2))

    assert layer.get_seq_length() == 2
    _assert_attn_state_tokens(layer.state["attn_state"], torch.arange(20, 22))


@pytest.mark.parametrize("cache_class", [FLACache, LegacyFLACache])
def test_cache_reset(cache_class):
    cache = _new_cache(cache_class)
    for layer_idx in range(2):
        cache.update(
            attn_state=_make_attn_state(layer_idx * 10, 3),
            layer_idx=layer_idx,
            offset=3,
        )

    assert cache.get_seq_length(0) == 3

    cache.reset()

    assert cache.get_seq_length(0) == 0
    if isinstance(cache, FLACache):
        assert len(cache) == 2
        assert cache[0] is None
        assert cache[1] is None
    else:
        assert len(cache) == 0


@pytest.mark.skipif(not _HAS_LAYER_CACHE, reason='Requires the Transformers cache-layer constructor.')
def test_cache_from_legacy_preserves_seen_tokens():
    seen_tokens = 17
    legacy_state = {
        "recurrent_state": None,
        "attn_state": _make_attn_state(10, 4),
        "conv_state": None,
        "ffn_state": None,
    }

    cache = FLACache.from_legacy_cache([legacy_state], seen_tokens=seen_tokens)

    assert cache.get_seq_length(0) == seen_tokens
    _assert_attn_state_tokens(cache[0]["attn_state"], torch.arange(10, 14))


@pytest.mark.parametrize('module_name', ['fla.models.utils', 'fla.models.cache_utils'], ids=['legacy', 'canonical'])
@pytest.mark.parametrize('cache_class', [Cache, FLACache, LegacyFLACache])
def test_cache_pickle_preserves_states(monkeypatch, module_name, cache_class):
    cache = _new_cache(cache_class)
    recurrent_state = (torch.arange(24).reshape(1, 2, 3, 4), torch.arange(6).reshape(1, 2, 3))
    cache.update(recurrent_state=recurrent_state, attn_state=_make_attn_state(10, 4), layer_idx=0, offset=4)

    with monkeypatch.context() as patch:
        for cls in (Cache, FLACache, FLALayer, LegacyFLACache):
            patch.setattr(cls, '__module__', module_name)
        checkpoint = pickle.dumps(cache)
    restored = pickle.loads(checkpoint)

    assert type(restored) is cache_class
    torch.testing.assert_close(restored[0]['recurrent_state'], recurrent_state)
    assert restored.get_seq_length(0) == 4
    _assert_attn_state_tokens(restored[0]['attn_state'], torch.arange(10, 14))
    restored.update(attn_state=_make_attn_state(14, 2), layer_idx=0, offset=2)
    torch.testing.assert_close(restored[0]['recurrent_state'], recurrent_state)
    assert restored.get_seq_length(0) == 6
    _assert_attn_state_tokens(restored[0]['attn_state'], torch.arange(10, 16))


def test_legacy_cache_tuple_round_trip():
    class InitializableLegacyFLACache(LegacyFLACache):
        def __init__(self, seen_tokens: int = 0):
            self.states = []
            self._seen_tokens = seen_tokens

    seen_tokens = 17
    cache = InitializableLegacyFLACache(seen_tokens=seen_tokens)
    cache.states.append({
        "recurrent_state": None,
        "attn_state": _make_attn_state(10, 4),
        "conv_state": None,
        "ffn_state": None,
    })

    legacy_cache = cache.to_legacy_cache()
    restored = InitializableLegacyFLACache.from_legacy_cache(legacy_cache, seen_tokens=seen_tokens)

    assert restored.to_legacy_cache() == legacy_cache
    assert restored.get_seq_length(0) == seen_tokens


@pytest.mark.parametrize(
    ['num_layers', 'batch_size', 'seq_len', 'hidden_size', 'num_heads'],
    [
        pytest.param(*test, id=f"L{test[0]}-B{test[1]}-T{test[2]}-D{test[3]}-H{test[4]}")
        for test in [
            (4, 1, 1, 64, 4),
            (8, 2, 1, 128, 8),
            (4, 1, 3, 64, 4),
        ]
    ],
)
def test_cache_decode_all_layers_see_same_past_length(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
):
    """Each layer sees the same pre-update RoPE offset: https://github.com/fla-org/flash-linear-attention/issues/766."""
    cache = _new_cache(FLACache)
    head_dim = hidden_size // num_heads

    initial_seq_len = 5
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device_name)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=initial_seq_len,
        )

    for layer_idx in range(num_layers):
        assert cache.get_seq_length(layer_idx) == initial_seq_len, \
            f"Layer {layer_idx} should have seq_length={initial_seq_len} after prefill"

    observed_lengths_before_update = []
    for layer_idx in range(num_layers):
        seq_length_before_update = cache.get_seq_length(layer_idx)
        observed_lengths_before_update.append(seq_length_before_update)

        key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
        attn_state = (key_states, value_states)

        cache.update(
            attn_state=attn_state,
            layer_idx=layer_idx,
            offset=seq_len,
        )

    expected_length = initial_seq_len
    for layer_idx, observed_len in enumerate(observed_lengths_before_update):
        assert observed_len == expected_length, (
            f"GitHub Issue #766 regression detected! Layer {layer_idx} saw wrong past length for RoPE: "
            f"expected {expected_length}, got {observed_len}. "
            f"Different layers are computing different RoPE offsets during the same forward pass."
        )


@pytest.mark.parametrize(
    ['num_layers', 'batch_size', 'seq_len', 'hidden_size', 'num_heads'],
    [
        pytest.param(*test, id=f"legacy-L{test[0]}-B{test[1]}-T{test[2]}")
        for test in [
            (4, 1, 1, 64, 4),
            (8, 2, 1, 128, 8),
        ]
    ],
)
def test_legacy_cache_decode_all_layers_see_same_past_length(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
):
    """The legacy counter advances after the last layer: https://github.com/fla-org/flash-linear-attention/issues/766."""
    cache = _new_cache(LegacyFLACache)

    head_dim = hidden_size // num_heads

    initial_seq_len = 5
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device_name)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=initial_seq_len,
        )

    observed_lengths = []
    for layer_idx in range(num_layers):
        seq_length_before_update = cache.get_seq_length(layer_idx)
        observed_lengths.append(seq_length_before_update)

        key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)

        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=seq_len,
        )

    expected_length = initial_seq_len
    for layer_idx, observed_len in enumerate(observed_lengths):
        assert observed_len == expected_length, (
            f"GitHub Issue #766 regression detected! LegacyFLACache Layer {layer_idx} saw wrong past length: "
            f"expected {expected_length}, got {observed_len}."
        )


@pytest.mark.parametrize(
    ['num_decode_steps', 'num_layers'],
    [
        pytest.param(5, 4, id="5steps-4layers"),
        pytest.param(10, 8, id="10steps-8layers"),
    ],
)
def test_cache_incremental_decode_consistency(num_decode_steps: int, num_layers: int):
    """Every layer starts each decoding step at the same offset."""
    cache = _new_cache(FLACache)
    batch_size, seq_len, num_heads, head_dim = 1, 1, 4, 16

    observed_lengths_per_step = []

    for decode_step in range(num_decode_steps):
        expected_past_len = decode_step
        observed_this_step = []

        for layer_idx in range(num_layers):
            actual_past_len = cache.get_seq_length(layer_idx)
            observed_this_step.append((layer_idx, actual_past_len))

            key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
            value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_name)
            cache.update(
                attn_state=(key_states, value_states),
                layer_idx=layer_idx,
                offset=seq_len,
            )

        observed_lengths_per_step.append(observed_this_step)

    for decode_step, observations in enumerate(observed_lengths_per_step):
        expected_past_len = decode_step
        for layer_idx, actual_past_len in observations:
            assert actual_past_len == expected_past_len, (
                f"GitHub Issue #766 regression detected! Cache shows inconsistent past lengths "
                f"during decode step {decode_step}. Expected all layers to see {expected_past_len}, "
                f"but Layer {layer_idx} saw {actual_past_len}. "
                f"This causes incorrect RoPE offset calculation in upper layers."
            )


def test_per_layer_independent_counter_regression():
    """Updating one layer leaves the other layers' token counts unchanged."""
    layers = [FLALayer() for _ in range(4)]

    for i, layer in enumerate(layers):
        layer._seen_tokens = i * 10

    key_states = torch.randn(1, 5, 4, 16, device=device_name)
    value_states = torch.randn(1, 5, 4, 16, device=device_name)
    layers[0].update(attn_state=(key_states, value_states))

    assert layers[0].get_seq_length() == 5
    assert layers[1].get_seq_length() == 10
    assert layers[2].get_seq_length() == 20
    assert layers[3].get_seq_length() == 30


def test_rope_offset_consistency_simulation_regression():
    """Layers use matching RoPE offsets when processing the same token."""
    cache = _new_cache(FLACache)
    batch_size, num_heads, head_dim = 1, 4, 16
    num_layers = 8
    prompt_len = 10

    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, prompt_len, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, prompt_len, num_heads, head_dim, device=device_name)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=prompt_len,
        )

    rope_offsets = []
    for layer_idx in range(num_layers):
        seqlen_offset = cache.get_seq_length(layer_idx)
        rope_offsets.append(seqlen_offset)

        key_states = torch.randn(batch_size, 1, num_heads, head_dim, device=device_name)
        value_states = torch.randn(batch_size, 1, num_heads, head_dim, device=device_name)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=1,
        )

    assert len(set(rope_offsets)) == 1, (
        f"GitHub Issue #766 regression: Different layers used different RoPE offsets: {rope_offsets}. "
        f"This means the same token gets different position encodings at different layers!"
    )
    assert rope_offsets[0] == prompt_len
