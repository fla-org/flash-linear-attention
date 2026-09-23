# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest

from fla.layers.delta_net import DeltaNet
from fla.layers.gla import GatedLinearAttention
from fla.layers.gsa import GatedSlotAttention
from fla.layers.hgrn2 import HGRN2Attention
from fla.layers.linear_attn import LinearAttention
from fla.layers.mamba2 import Mamba2
from fla.layers.multiscale_retention import MultiScaleRetention
from fla.layers.rwkv6 import RWKV6Attention
from fla.layers.rwkv7 import RWKV7Attention
from fla.layers.simple_gla import SimpleGatedLinearAttention

# hidden_size * expand is an integer only in floating-point representation
_EXPAND_LAYERS = [
    GatedLinearAttention, GatedSlotAttention, LinearAttention,
    SimpleGatedLinearAttention, MultiScaleRetention, DeltaNet, RWKV6Attention,
]


@pytest.mark.parametrize("layer_cls", _EXPAND_LAYERS)
def test_expand_dim_rounding(layer_cls):
    """`hidden_size * expand` that is an integer up to fp error must round, not truncate."""
    # 100 * 0.58 == 57.999...: must round to 58 (truncating to 57 was the bug)
    layer = layer_cls(hidden_size=100, expand_k=0.58, expand_v=0.5, num_heads=1)
    assert layer.key_dim == 58


@pytest.mark.parametrize("layer_cls", _EXPAND_LAYERS)
def test_expand_dim_rejects_non_integer(layer_cls):
    """A truly non-integer `hidden_size * expand` must fail loudly instead of being used silently."""
    # 101 * 0.58 == 58.58: not an integer
    with pytest.raises(AssertionError):
        layer_cls(hidden_size=101, expand_k=0.58, expand_v=0.5, num_heads=1)


def test_rwkv7_head_dim_divisibility():
    RWKV7Attention(hidden_size=1000, head_dim=None, num_heads=5)
    with pytest.raises(AssertionError):
        RWKV7Attention(hidden_size=1000, head_dim=None, num_heads=6)
    RWKV7Attention(hidden_size=1000, head_dim=100)
    with pytest.raises(AssertionError):
        RWKV7Attention(hidden_size=1000, head_dim=64)


def test_mamba2_n_groups_divisibility():
    Mamba2(hidden_size=512, n_groups=4)
    with pytest.raises(ValueError, match="n_groups"):
        Mamba2(hidden_size=512, n_groups=3)


def test_hgrn2_state_size():
    layer = HGRN2Attention(hidden_size=64, expand_ratio=2, use_short_conv=False)
    assert layer.state_size() == 128
