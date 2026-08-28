# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.attn import Attention
from fla.layers.bitattn import BitAttention
from fla.layers.mla import MultiheadLatentAttention
from fla.layers.moba import MoBA
from fla.layers.nsa import NativeSparseAttention
from fla.models.utils import Cache
from fla.utils import device

try:
    import flash_attn  # noqa: F401
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


@pytest.mark.parametrize(
    ("layer_cls", "layer_kwargs", "requires_flash"),
    [
        pytest.param(
            MultiheadLatentAttention,
            dict(hidden_size=128, num_heads=2, q_lora_rank=None, qk_rope_head_dim=32, kv_lora_rank=64,
                 v_head_dim=64, qk_nope_head_dim=64, qk_head_dim=None, layer_idx=0),
            True,
            id="mla",
        ),
        pytest.param(
            MoBA,
            dict(hidden_size=128, num_heads=2, num_kv_heads=2, moba_chunk_size=32, moba_topk=2, layer_idx=0),
            True,
            id="moba",
        ),
        pytest.param(
            NativeSparseAttention,
            dict(hidden_size=128, num_heads=2, num_kv_heads=1, head_dim=64, block_size=32, block_counts=2,
                 window_size=16, layer_idx=0),
            False,
            id="nsa",
        ),
        pytest.param(
            BitAttention,
            dict(hidden_size=128, num_heads=2, num_kv_heads=2, layer_idx=0),
            True,
            id="bitattn",
        ),
        pytest.param(
            Attention,
            dict(hidden_size=128, num_heads=2, layer_idx=0),
            True,
            id="attn",
        ),
    ],
)
def test_cu_seqlens_rejected_with_non_empty_cache(layer_cls: type, layer_kwargs: dict, requires_flash: bool):
    if requires_flash and not HAS_FLASH:
        pytest.skip(reason="Skipping test because flash-attn is not installed")

    torch.manual_seed(42)
    layer = layer_cls(**layer_kwargs).to(device=device, dtype=torch.float16).eval()
    prefill_states = torch.randn(2, 64, 128, device=device, dtype=torch.float16)
    packed_states = torch.randn(1, 16, 128, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 7, 16], device=device, dtype=torch.int32)

    cache = Cache()
    layer(hidden_states=packed_states, past_key_values=cache, cu_seqlens=cu_seqlens, use_cache=True)

    cache = Cache()
    layer(hidden_states=prefill_states, past_key_values=cache, use_cache=True)
    with pytest.raises(AssertionError, match="cu_seqlens should not be provided"):
        layer(hidden_states=packed_states, past_key_values=cache, cu_seqlens=cu_seqlens, use_cache=True)
