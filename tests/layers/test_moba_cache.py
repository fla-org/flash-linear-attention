# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.moba import MoBA
from fla.models.utils import Cache
from fla.utils import device


def test_moba_rejects_cu_seqlens_with_nonempty_cache():
    T, H, D = 64, 2, 64
    torch.manual_seed(42)
    layer = MoBA(
        hidden_size=H * D,
        num_heads=H,
        layer_idx=0,
    ).to(device=device, dtype=torch.float16).eval()
    hidden_states = torch.randn(1, T, H * D, device=device, dtype=torch.float16)

    cache = Cache()
    with torch.no_grad():
        layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        assert cache.get_seq_length(0) == T

        cu_seqlens = torch.tensor([0, T // 2, T], dtype=torch.int32, device=device)
        with pytest.raises(AssertionError, match="cu_seqlens should not be provided"):
            layer(hidden_states=hidden_states, past_key_values=cache, cu_seqlens=cu_seqlens)
