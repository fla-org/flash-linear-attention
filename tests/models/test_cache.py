# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models.utils import FLACache, FLALayer, LegacyFLACache
from fla.utils import device


# encode token positions into tensor values so window tail assertions are exact
def _make_attn_state(start: int, length: int, *, batch_size: int = 1, num_heads: int = 2, head_dim: int = 3):
    token_ids = torch.arange(start, start + length, dtype=torch.float32, device=device).view(1, length, 1, 1)
    key_states = token_ids.expand(batch_size, length, num_heads, head_dim).contiguous()
    value_states = (token_ids + 1000).expand(batch_size, length, num_heads, head_dim).contiguous()
    return key_states, value_states


def _assert_attn_state_tokens(attn_state, expected_tokens: torch.Tensor):
    expected = expected_tokens.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
    expected_key = expected.expand_as(attn_state[0])
    expected_value = (expected + 1000).expand_as(attn_state[1])

    torch.testing.assert_close(attn_state[0], expected_key)
    torch.testing.assert_close(attn_state[1], expected_value)


def _new_cache(cache_cls):
    try:
        return cache_cls()
    except (TypeError, ValueError) as exc:
        pytest.skip(f"{cache_cls.__name__} is not compatible with this transformers version: {exc}")


def _raw_legacy_cache():
    cache = object.__new__(LegacyFLACache)
    cache.states = []
    cache._seen_tokens = 0
    return cache


# ===================================================================================
# Test for FLACache per-layer get_seq_length behavior
# ===================================================================================
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
    """
    Test that FLACache.get_seq_length returns per-layer sequence length,
    not a global counter. This is important for multi-layer models where
    each layer should maintain its own sequence length.

    See: https://github.com/fla-org/flash-linear-attention/issues/747
    """
    cache = FLACache()
    head_dim = hidden_size // num_heads

    # Simulate updating cache for each layer sequentially
    for layer_idx in range(num_layers):
        # Initially, the layer doesn't exist, so seq_length should be 0
        assert cache.get_seq_length(layer_idx) == 0

        # Create dummy attention states (key, value)
        key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        attn_state = (key_states, value_states)

        # Update the cache for this layer
        cache.update(
            attn_state=attn_state,
            layer_idx=layer_idx,
        )

        # Verify that this layer's seq_length is updated
        assert cache.get_seq_length(layer_idx) == seq_len, \
            f"Layer {layer_idx} should have seq_length={seq_len}, got {cache.get_seq_length(layer_idx)}"

        # Verify that the next layer still has seq_length=0 (not updated yet)
        if layer_idx + 1 < num_layers:
            assert cache.get_seq_length(layer_idx + 1) == 0, \
                f"Layer {layer_idx + 1} should have seq_length=0 before being updated"

    # Now verify all layers have the correct seq_length
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
    """
    Test that FLACache correctly tracks incremental updates to each layer,
    simulating autoregressive generation where tokens are added one at a time.
    """
    cache = FLACache()
    head_dim = hidden_size // num_heads

    # Simulate incremental token generation
    for chunk_idx in range(num_chunks):
        for layer_idx in range(num_layers):
            # Create dummy attention states for this chunk
            key_states = torch.randn(batch_size, chunk_size, num_heads, head_dim, device=device)
            value_states = torch.randn(batch_size, chunk_size, num_heads, head_dim, device=device)
            attn_state = (key_states, value_states)

            # Update the cache for this layer
            cache.update(
                attn_state=attn_state,
                layer_idx=layer_idx,
            )

            # Verify seq_length accumulates correctly
            expected_seq_len = (chunk_idx + 1) * chunk_size
            actual_seq_len = cache.get_seq_length(layer_idx)
            assert actual_seq_len == expected_seq_len, \
                f"Layer {layer_idx} after chunk {chunk_idx} should have seq_length={expected_seq_len}, got {actual_seq_len}"


def test_cache_get_seq_length_nonexistent_layer():
    """
    Test that get_seq_length returns 0 for non-existent layers
    and handles None layer_idx correctly for populated caches.
    """
    cache = FLACache()

    # Should return 0 for layers that don't exist yet
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(5) == 0
    assert cache.get_seq_length(None) == 0

    # Populate the cache with one layer
    key_states = torch.randn(1, 10, 4, 16, device=device)
    value_states = torch.randn(1, 10, 4, 16, device=device)
    cache.update(attn_state=(key_states, value_states), layer_idx=0)

    # After populating, get_seq_length(None) should default to layer 0
    assert cache.get_seq_length(None) == 10
    assert cache.get_seq_length(0) == 10


def test_cache_window_size_does_not_undercount():
    """
    Test that window_size truncation doesn't undercount sequence length.
    When window_size is applied and input exceeds it, the full input size
    should still be counted in _seen_tokens.
    """
    cache = FLACache()
    batch_size, seq_len, num_heads, head_dim = 1, 100, 4, 16
    window_size = 10

    key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Update with window_size smaller than seq_len
    cache.update(
        attn_state=(key_states, value_states),
        layer_idx=0,
        cache_kwargs={"window_size": window_size}
    )

    # Sequence length should be the full seq_len, not window_size
    assert cache.get_seq_length(0) == seq_len, \
        f"Expected seq_length={seq_len}, got {cache.get_seq_length(0)} (window_size={window_size})"


@pytest.mark.parametrize("cache_cls", [FLACache, LegacyFLACache])
@pytest.mark.parametrize(
    ["first_len", "second_len", "window_size"],
    [
        pytest.param(4, 6, 4, id="full-window-oversized-update"),
        pytest.param(2, 4, 4, id="partial-window-overflow"),
    ],
)
def test_cache_window_overflow_keeps_tail(cache_cls, first_len: int, second_len: int, window_size: int):
    cache = _new_cache(cache_cls)

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


@pytest.mark.parametrize("cache_cls", [FLACache, LegacyFLACache])
def test_cache_reset(cache_cls):
    cache = _new_cache(cache_cls)
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


# ===================================================================================
# Tests for GitHub Issue #766: Cache seen-token count committed too early during decode
# ===================================================================================

@pytest.mark.parametrize(
    ['num_layers', 'batch_size', 'seq_len', 'hidden_size', 'num_heads'],
    [
        pytest.param(*test, id=f"L{test[0]}-B{test[1]}-T{test[2]}-D{test[3]}-H{test[4]}")
        for test in [
            (4, 1, 1, 64, 4),   # Single token decode
            (8, 2, 1, 128, 8),  # Batch decode
            (4, 1, 3, 64, 4),   # Multi-token prefill/decode
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
    """
    Test that during decode, all layers see the same past length for RoPE offset.

    GitHub Issue #766: The cache implementation updates each layer's seen-token count
    when that layer's update() is called. This means during a single forward pass,
    Layer 0 will have _seen_tokens incremented immediately after its update(), while
    Layer 1+ still see their old (lower) values until they are updated.

    This inconsistency causes different layers to compute different RoPE offsets
    during the same forward pass, breaking exact cache equivalence.

    Expected behavior: During a single forward pass, ALL layers should see the same
    past_len, regardless of their position in the stack. The seen-token count should
    only increment AFTER all layers have computed their RoPE for the current step.

    Current buggy behavior: Layer 0 sees N, Layer 1 sees N+1, Layer 2 sees N+2, etc.
    """
    cache = FLACache()
    head_dim = hidden_size // num_heads

    # First, pre-populate the cache to simulate we've already processed some tokens
    # This creates all layers with initial _seen_tokens = some_value
    initial_seq_len = 5
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=initial_seq_len,
        )

    # Now all layers should have _seen_tokens = initial_seq_len
    for layer_idx in range(num_layers):
        assert cache.get_seq_length(layer_idx) == initial_seq_len, \
            f"Layer {layer_idx} should have seq_length={initial_seq_len} after prefill"

    # Simulate a decode step (single token) through all layers
    observed_lengths_before_update = []
    for layer_idx in range(num_layers):
        # This is what attention layers do: get seq_length BEFORE computing attention
        # to determine RoPE offset (seqlen_offset)
        seq_length_before_update = cache.get_seq_length(layer_idx)
        observed_lengths_before_update.append(seq_length_before_update)

        # Create dummy attention states (key, value) for the new token
        key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        attn_state = (key_states, value_states)

        # Update the cache (this is what happens AFTER attention computation)
        cache.update(
            attn_state=attn_state,
            layer_idx=layer_idx,
            offset=seq_len,
        )

    # CRITICAL CHECK: All layers should have seen the SAME past length
    # during this forward pass. They should ALL see initial_seq_len,
    # not incrementally larger values as layers are updated.
    #
    # This is a regression test for GitHub Issue #766.
    # Before the fix (per-layer _seen_tokens):
    #   - Layer 0 sees initial_seq_len
    #   - Layer 1 sees initial_seq_len + 1 (because Layer 0 already updated)
    #   - Layer 2 sees initial_seq_len + 2 (because Layer 0 and 1 updated)
    #   etc.
    #
    # After the fix (PR #748):
    #   - ALL layers see initial_seq_len
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
    """
    Test that LegacyFLACache also has consistent past length across all layers during decode.

    Same issue as test_cache_decode_all_layers_see_same_past_length but for LegacyFLACache.

    See: https://github.com/fla-org/flash-linear-attention/issues/766
    """
    # Skip if LegacyFLACache is not compatible with current transformers version
    try:
        cache = LegacyFLACache()
    except (ValueError, TypeError) as e:
        pytest.skip(f"LegacyFLACache not compatible with current transformers version: {e}")

    head_dim = hidden_size // num_heads

    # First, pre-populate the cache to simulate we've already processed some tokens
    initial_seq_len = 5
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, initial_seq_len, num_heads, head_dim, device=device)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=initial_seq_len,
        )

    # Now simulate a decode step
    observed_lengths = []
    for layer_idx in range(num_layers):
        # Get seq_length BEFORE update (for RoPE offset)
        seq_length_before_update = cache.get_seq_length(layer_idx)
        observed_lengths.append(seq_length_before_update)

        # Create and update with dummy attention states
        key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=seq_len,
        )

    # All layers should see the same past length during the forward pass
    # Regression test for GitHub Issue #766
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
    """
    Test that cache remains consistent across multiple decode steps.

    Simulates autoregressive generation where we do multiple decode steps,
    each with a single token. During each decode step, all layers should see
    the same past_len before computing RoPE.

    This catches the GitHub #766 issue where upper layers see inflated past_len
    because Layer 0's update() increments the seen-token count too early.

    See: https://github.com/fla-org/flash-linear-attention/issues/766
    """
    cache = FLACache()
    batch_size, seq_len, num_heads, head_dim = 1, 1, 4, 16

    # Track what length each layer observes at each decode step
    observed_lengths_per_step = []

    for decode_step in range(num_decode_steps):
        expected_past_len = decode_step  # Before this step, we've seen 'decode_step' tokens
        observed_this_step = []

        # Simulate forward pass through all layers
        for layer_idx in range(num_layers):
            # Each layer should see the same past_len at this decode step
            actual_past_len = cache.get_seq_length(layer_idx)
            observed_this_step.append((layer_idx, actual_past_len))

            # Update cache for this layer
            key_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            value_states = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            cache.update(
                attn_state=(key_states, value_states),
                layer_idx=layer_idx,
                offset=seq_len,
            )

        observed_lengths_per_step.append(observed_this_step)

    # Check consistency: within each step, all layers should see the same past_len
    # Regression test for GitHub Issue #766
    for decode_step, observations in enumerate(observed_lengths_per_step):
        expected_past_len = decode_step
        for layer_idx, actual_past_len in observations:
            assert actual_past_len == expected_past_len, (
                f"GitHub Issue #766 regression detected! Cache shows inconsistent past lengths "
                f"during decode step {decode_step}. Expected all layers to see {expected_past_len}, "
                f"but Layer {layer_idx} saw {actual_past_len}. "
                f"This causes incorrect RoPE offset calculation in upper layers."
            )


# ===================================================================================
# Additional regression tests for GitHub Issue #766
# ===================================================================================

def test_per_layer_independent_counter_regression():
    """
    Verify that each FLALayer maintains its own independent _seen_tokens counter.

    This is the mechanism that prevents the GitHub #766 bug.
    """
    layers = [FLALayer() for _ in range(4)]

    # Initialize with different values
    for i, layer in enumerate(layers):
        layer._seen_tokens = i * 10  # 0, 10, 20, 30

    # Update only layer 0
    key_states = torch.randn(1, 5, 4, 16, device=device)
    value_states = torch.randn(1, 5, 4, 16, device=device)
    layers[0].update(attn_state=(key_states, value_states))

    # Verify only layer 0 changed
    assert layers[0].get_seq_length() == 5  # 0 + 5
    assert layers[1].get_seq_length() == 10  # unchanged
    assert layers[2].get_seq_length() == 20  # unchanged
    assert layers[3].get_seq_length() == 30  # unchanged


def test_rope_offset_consistency_simulation_regression():
    """
    Simulate how RoPE offsets would be calculated during decode.

    This test demonstrates that with the fix, all layers compute
    the same position encoding for the same token.
    """
    cache = FLACache()
    batch_size, num_heads, head_dim = 1, 4, 16
    num_layers = 8
    prompt_len = 10

    # Prefill
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, prompt_len, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, prompt_len, num_heads, head_dim, device=device)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=prompt_len,
        )

    # Decode and collect what RoPE offset each layer would use
    rope_offsets = []
    for layer_idx in range(num_layers):
        seqlen_offset = cache.get_seq_length(layer_idx)
        rope_offsets.append(seqlen_offset)

        # Update
        key_states = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
        value_states = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
        cache.update(
            attn_state=(key_states, value_states),
            layer_idx=layer_idx,
            offset=1,
        )

    # All layers should use the same offset
    assert len(set(rope_offsets)) == 1, (
        f"GitHub Issue #766 regression: Different layers used different RoPE offsets: {rope_offsets}. "
        f"This means the same token gets different position encodings at different layers!"
    )
    assert rope_offsets[0] == prompt_len
