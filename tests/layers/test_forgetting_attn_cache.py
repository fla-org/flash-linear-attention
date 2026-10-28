# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.bitattn import BitAttention
from fla.layers.forgetting_attn import ForgettingAttention
from fla.models.cache_utils import Cache
from fla.utils import assert_close, device

try:
    import flash_attn  # noqa: F401
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_empty_sliding_window_cache_prefill(dtype: torch.dtype):
    B, T, H, D, W = 2, 64, 2, 64, 16
    torch.manual_seed(42)
    layer = ForgettingAttention(
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
        expected_next, _, _ = layer(hidden_states=torch.cat([hidden_states, next_hidden_states], dim=1)[:, -W:])
        cache = Cache()
        actual, _, returned_cache = layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        actual_next, _, _ = layer(
            hidden_states=next_hidden_states,
            attention_mask=torch.ones(B, W, dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )

    assert_close("prefill", expected, actual, tol)
    assert_close("decode", expected_next[:, -1:], actual_next, tol)
    assert returned_cache is cache
    cached_k, cached_v, cached_f = cache[0]['attn_state']
    assert cached_k.shape == cached_v.shape == (B, W, H * D)
    assert cached_f.shape == (B, W, H)
    assert cache.get_seq_length(0) == T + 1


@pytest.mark.parametrize('left_padding', [0, 2], ids=['all_valid', 'left_padding'])
@pytest.mark.parametrize(
    'layer_class',
    [
        pytest.param(
            BitAttention,
            id='bitattention',
            marks=pytest.mark.skipif(not HAS_FLASH, reason="BitAttention requires flash-attn"),
        ),
        pytest.param(ForgettingAttention, id='forgetting'),
    ],
)
def test_cached_decode_mask_length(layer_class, left_padding: int):
    B, P, H, D, W = 1, 6, 2, 64, 4
    torch.manual_seed(42)
    layer_kwargs = {'max_position_embeddings': P + 1} if layer_class is BitAttention else {}
    layer = layer_class(hidden_size=H * D, num_heads=H, window_size=W, layer_idx=0, **layer_kwargs)
    layer = layer.to(device=device, dtype=torch.float16).eval()
    hidden_states = torch.randn(B, P + 1, H * D, device=device, dtype=torch.float16)
    attention_mask = torch.ones(B, P + 1, dtype=torch.long, device=device)
    attention_mask[:, :left_padding] = 0

    with torch.no_grad():
        if layer_class is BitAttention:
            expected = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=Cache(),
                use_cache=True,
            )[0][:, -1:]
        else:
            expected = layer(hidden_states=hidden_states[:, -W:])[0][:, -1:]
        cache = Cache()
        layer(
            hidden_states=hidden_states[:, :P],
            attention_mask=attention_mask[:, :P],
            past_key_values=cache,
            use_cache=True,
        )
        actual = layer(
            hidden_states=hidden_states[:, P:],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )[0]

    assert_close('decode', expected, actual, 0.005)
    assert cache[0]['attn_state'][0].shape[1] == W
