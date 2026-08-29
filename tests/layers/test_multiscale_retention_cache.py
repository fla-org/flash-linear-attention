# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.multiscale_retention import MultiScaleRetention
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_rotary_offset_with_attention_mask(dtype: torch.dtype):
    B, T, H, D = 2, 32, 4, 64
    torch.manual_seed(42)
    layer = MultiScaleRetention(
        hidden_size=H * D,
        num_heads=H,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    next_hidden_states = torch.randn(B, 2, H * D, device=device, dtype=dtype)
    # Manual decode/serving callers may pass an attention_mask covering only the new tokens.
    decode_mask = torch.ones(B, 1, dtype=torch.long, device=device)
    tol = 0.005 if dtype == torch.float16 else 0.02

    with torch.no_grad():
        expected, _, _ = layer(hidden_states=hidden_states)
        expected_steps, _, _ = layer(hidden_states=torch.cat([hidden_states, next_hidden_states], dim=1))
        cache = Cache()
        actual, _, returned_cache = layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        actual_steps = []
        for i in range(2):
            actual_step, _, _ = layer(
                hidden_states=next_hidden_states[:, i:i + 1],
                attention_mask=decode_mask,
                past_key_values=cache,
                use_cache=True,
            )
            actual_steps.append(actual_step)

    assert_close("prefill", expected, actual, tol)
    for i in range(2):
        assert_close(f"decode {i}", expected_steps[:, T + i:T + i + 1], actual_steps[i], tol)
    assert returned_cache is cache
    assert cache.get_seq_length(0) == T + 2
