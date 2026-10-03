# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mom import MomAttention
from fla.models.utils import Cache
from fla.utils import device


@pytest.mark.parametrize('decode_steps', [pytest.param(1, id='decode1'), pytest.param(3, id='decode3')])
def test_cache_offset_counts_new_tokens(decode_steps: int):
    B, T, H, D = 2, 8, 128, 64
    torch.manual_seed(42)
    layer = MomAttention(
        hidden_size=H,
        num_heads=2,
        head_dim=D,
        layer_idx=0,
    ).to(device=device).eval()
    cache = Cache()

    with torch.no_grad():
        layer(hidden_states=torch.randn(B, T, H, device=device), past_key_values=cache, use_cache=True)
        assert cache.get_seq_length(0) == T
        for step in range(decode_steps):
            layer(hidden_states=torch.randn(B, 1, H, device=device), past_key_values=cache, use_cache=True)
            assert cache.get_seq_length(0) == T + step + 1
