# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.forgetting_attn import ForgettingAttention
from fla.models.utils import Cache
from fla.utils import device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_decode_without_mask_is_rejected(dtype: torch.dtype):
    B, T, H, D = 2, 8, 2, 64
    torch.manual_seed(42)
    layer = ForgettingAttention(
        hidden_size=H * D,
        num_heads=H,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    next_hidden_states = torch.randn(B, 1, H * D, device=device, dtype=dtype)

    with torch.no_grad():
        cache = Cache()
        layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        with pytest.raises(AssertionError, match="only support q_len == kv_len"):
            layer(hidden_states=next_hidden_states, past_key_values=cache, use_cache=True)
