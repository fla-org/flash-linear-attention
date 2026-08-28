# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest

from fla.layers.abc import ABCAttention
from fla.layers.lightnet import LightNetAttention


@pytest.mark.parametrize(
    ('expand_k', 'expand_v', 'use_short_conv'),
    [
        pytest.param(0.5, 1.0, False, id='default'),
        pytest.param(2.0, 2.0, True, id='expanded-short-conv'),
    ],
)
def test_abc_state_size(expand_k: float, expand_v: float, use_short_conv: bool):
    hidden_size, num_heads, num_slots, conv_size = 128, 4, 32, 4
    layer = ABCAttention(
        hidden_size=hidden_size,
        expand_k=expand_k,
        expand_v=expand_v,
        num_heads=num_heads,
        num_slots=num_slots,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        layer_idx=0,
    )

    key_dim = int(hidden_size * expand_k)
    value_dim = int(hidden_size * expand_v)
    expected = num_slots * (key_dim + value_dim)
    if use_short_conv:
        expected += (2 * key_dim + value_dim) * conv_size

    assert layer.state_size() == expected


def test_lightnet_state_size():
    hidden_size, num_heads, conv_size = 128, 4, 4
    expand_ratio = 64
    layer = LightNetAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        expand_ratio=expand_ratio,
        use_short_conv=True,
        conv_size=conv_size,
        layer_idx=0,
    )

    key_dim = num_heads * expand_ratio
    head_i_dim = hidden_size // num_heads
    expected = key_dim * head_i_dim + key_dim
    expected += (2 * key_dim + hidden_size) * conv_size

    assert layer.state_size() == expected
