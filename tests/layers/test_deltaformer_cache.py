# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.deltaformer import DeltaFormerAttention
from fla.models.utils import Cache
from fla.utils import device, find_spec_cached

# Mark (not importorskip): a fully skipped module reports "no tests collected" and exits
# with code 5, which CI per-file loops treat as a failure.
pytestmark = pytest.mark.skipif(find_spec_cached(
    "flash_attn") is None, reason="DeltaFormer attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cache_request_rejected(dtype: torch.dtype):
    B, T, H, D = 2, 64, 2, 64
    torch.manual_seed(42)
    layer = DeltaFormerAttention(
        hidden_size=H * D,
        num_heads=H,
        layer_idx=0,
    ).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(B, T, H * D, device=device, dtype=dtype)
    next_hidden_states = torch.randn(B, 1, H * D, device=device, dtype=dtype)
    cache = Cache()

    with torch.no_grad():
        o, _, past_key_values = layer(hidden_states=hidden_states)
        assert o.shape == (B, T, H * D)
        assert past_key_values is None
        with pytest.raises(NotImplementedError, match="does not support `past_key_values`"):
            layer(hidden_states=hidden_states, past_key_values=cache, use_cache=True)
        with pytest.raises(NotImplementedError, match="does not support `past_key_values`"):
            layer(hidden_states=next_hidden_states, past_key_values=cache, use_cache=True)

    assert cache.get_seq_length(0) == 0
