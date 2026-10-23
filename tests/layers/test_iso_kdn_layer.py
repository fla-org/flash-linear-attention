# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch

from fla.layers import IsotropicKalmanDeltaNetwork
from fla.models.utils import Cache
from fla.utils import assert_close, device


@pytest.mark.parametrize('T', [8, 100])
@pytest.mark.parametrize('use_short_conv', [False, True])
@torch.no_grad()
def test_cache_without_state_uses_learned_prior(T: int, use_short_conv: bool):
    torch.manual_seed(42)
    layer = IsotropicKalmanDeltaNetwork(
        hidden_size=128, head_dim=64, num_heads=2, use_short_conv=use_short_conv, layer_idx=0,
    ).to(device).eval()
    # a learned prior precision of 5, so that falling back to the default of 1 changes the output
    layer.initial_precision_param.fill_(math.log(math.expm1(4.9)))
    x = torch.randn(2, T, 128, device=device)
    expected = layer(x)[0]

    cache = Cache()
    layer(torch.randn(2, T, 128, device=device), past_key_values=cache, use_cache=False)
    actual = layer(x, past_key_values=cache, use_cache=False)[0]
    assert_close('o', expected, actual, 1e-6)
