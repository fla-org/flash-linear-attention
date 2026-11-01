# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.parallax import Parallax
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_empty_sliding_window_cache_prefill(dtype: torch.dtype):
    B, T, H, D, W = 2, 64, 2, 64, 16
    torch.manual_seed(42)
    layer = Parallax(
        hidden_size=H * D,
        num_heads=H,
        window_size=W,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    next_hidden_states = torch.randn(B, 1, H * D, device=device, dtype=dtype)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=hidden_states)
        expected_next, _, _ = layer(hidden_states=torch.cat([hidden_states, next_hidden_states], dim=1))
        cache = Cache()
        actual, _, returned_cache = layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        actual_next, _, _ = layer(hidden_states=next_hidden_states, past_key_values=cache, use_cache=True)

    assert_close("prefill", expected, actual, tol)
    assert_close("decode", expected_next[:, -1:], actual_next, tol)
    assert returned_cache is cache
    cached_k, cached_v = cache[0]['attn_state']
    assert cached_k.shape == cached_v.shape == (B, W, H * D)
    assert cache.get_seq_length(0) == T + 1


@torch.no_grad()
def test_left_padded_cache_matches_unpadded():
    torch.manual_seed(42)
    layer = Parallax(hidden_size=32, num_heads=2, layer_idx=0).to(device=device, dtype=torch.float16).eval()
    hidden_states = torch.randn(2, 4, 32, device=device, dtype=torch.float16)
    next_hidden_states = torch.randn(2, 1, 32, device=device, dtype=torch.float16)
    prefill_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], device=device, dtype=torch.long)
    decode_mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], device=device, dtype=torch.long)
    cache = Cache()
    actual, _, _ = layer(
        hidden_states=hidden_states,
        attention_mask=prefill_mask,
        past_key_values=cache,
        use_cache=True,
    )
    actual_next, _, _ = layer(
        hidden_states=next_hidden_states,
        attention_mask=decode_mask,
        past_key_values=cache,
        use_cache=True,
    )

    expected, expected_next = [], []
    for batch, start in ((0, 1), (1, 0)):
        reference_cache = Cache()
        output, _, _ = layer(
            hidden_states=hidden_states[batch:batch+1, start:],
            past_key_values=reference_cache,
            use_cache=True,
        )
        output_next, _, _ = layer(
            hidden_states=next_hidden_states[batch:batch+1],
            past_key_values=reference_cache,
            use_cache=True,
        )
        expected.append(output.squeeze(0))
        expected_next.append(output_next)

    assert_close("prefill", torch.cat(expected, dim=0), torch.cat([actual[0, 1:], actual[1]], dim=0), 0.005)
    assert_close("decode", torch.cat(expected_next, dim=0), actual_next, 0.005)
