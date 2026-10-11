# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.path_attn import PaTHAttention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_rejects_multi_token_query(dtype: torch.dtype):
    B, T, H, D = 2, 8, 2, 64
    torch.manual_seed(42)
    layer = PaTHAttention(
        hidden_size=H * D,
        num_heads=H,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    chunk_hidden_states = torch.randn(B, 3, H * D, device=device, dtype=dtype)
    next_hidden_states = torch.randn(B, 1, H * D, device=device, dtype=dtype)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=hidden_states)
        expected_next, _, _ = layer(hidden_states=torch.cat([hidden_states, next_hidden_states], dim=1))
        cache = Cache()
        actual, _, returned_cache = layer(
            hidden_states=hidden_states, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        cached_k, cached_v = cache[0]['attn_state']
        cached_k, cached_v = cached_k.clone(), cached_v.clone()
        cached_conv = cache[0]['conv_state'].clone()
        with pytest.raises(AssertionError, match="only support q_len == 1 for decoding"):
            layer(
                hidden_states=chunk_hidden_states,
                attention_mask=torch.ones(B, 3, dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
        assert cache.get_seq_length(0) == T
        assert torch.equal(cache[0]['attn_state'][0], cached_k)
        assert torch.equal(cache[0]['attn_state'][1], cached_v)
        assert torch.equal(cache[0]['conv_state'], cached_conv)
        actual_next, _, _ = layer(
            hidden_states=next_hidden_states,
            attention_mask=torch.ones(B, T + 1, dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )

    assert_close("prefill", expected, actual, tol)
    assert_close("decode", expected_next[:, -1:], actual_next, tol)
    assert returned_cache is cache
