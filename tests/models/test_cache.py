
import pytest
import torch

from fla.models.utils import FLACache
from fla.utils import device


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
    Test that get_seq_length returns 0 for non-existent layers.
    """
    cache = FLACache()

    # Should return 0 for layers that don't exist yet
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(5) == 0
    assert cache.get_seq_length(None) == 0
