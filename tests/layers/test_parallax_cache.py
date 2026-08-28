# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.parallax import Parallax
from fla.models.utils import Cache
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
