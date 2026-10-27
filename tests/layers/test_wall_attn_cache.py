# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.wall_attn import WallAttention
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window_size", [None, 16], ids=["full", "windowed"])
@pytest.mark.parametrize("cache_initialized", [False, True])
@pytest.mark.parametrize(
    ("T", "q_len", "use_scalar_gate"),
    [
        pytest.param(*test, id="T{}-q{}-scalar{}".format(*test))
        for test in [
            (64, 1, False),
            (24, 4, False),
            (24, 20, False),
            (60, 10, True),
            (8, 4, False),
            (8, 20, True),
        ]
    ],
)
def test_cache_continuation(
    T: int,
    q_len: int,
    use_scalar_gate: bool,
    cache_initialized: bool,
    window_size: int | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    B, H, D = 2, 2, 64
    layer = WallAttention(
        hidden_size=H * D,
        num_heads=H,
        window_size=window_size,
        use_scalar_gate=use_scalar_gate,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    next_hidden_states = torch.randn(B, q_len, H * D, device=device, dtype=dtype)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=hidden_states)
        expected_next, _, _ = layer(hidden_states=torch.cat([hidden_states, next_hidden_states], dim=1))
        cache = Cache()
        if cache_initialized:
            cache.update(layer_idx=0, offset=0)
        actual, _, returned_cache = layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        actual_next, _, _ = layer(hidden_states=next_hidden_states, past_key_values=cache, use_cache=True)

    assert_close("prefill", expected, actual, tol)
    assert_close("continuation", expected_next[:, -q_len:], actual_next, tol)
    assert returned_cache is cache
    state = cache[0]['attn_state']
    cache_len = T + q_len if window_size is None else min(T + q_len, window_size)
    assert state[0].shape == state[1].shape == state[2].shape == (B, cache_len, H, D)
    if use_scalar_gate:
        assert state[3].shape == (B, cache_len, H)
    assert cache.get_seq_length(0) == T + q_len


@pytest.mark.parametrize("window_size", [None, 16], ids=["full", "windowed"])
def test_cache_with_attention_mask_guard(window_size: int | None):
    B, T, H, D = 2, 16, 2, 64
    torch.manual_seed(42)
    layer = WallAttention(
        hidden_size=H * D,
        num_heads=H,
        window_size=window_size,
        layer_idx=0,
    ).to(device=device, dtype=torch.float16).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=torch.float16)
    attention_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    attention_mask[:, -8:] = False

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=hidden_states)
        cache = Cache()
        actual, _, _ = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        assert_close("prefill", expected[:, :-8], actual[:, :-8], 0.005)
        with pytest.raises(AssertionError, match="attention_mask"):
            layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
