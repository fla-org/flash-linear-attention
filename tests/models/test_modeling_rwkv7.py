# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import RWKV7Config

from .utils import check_forward, check_generation


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(L, B, T, H, D, RWKV7Config, use_l2warp=use_l2warp, dtype=dtype, varlen=varlen)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    check_generation(L, B, T, H, D, RWKV7Config, dtype)


def test_single_layer():
    # the per-layer init interpolates over num_hidden_layers - 1 layers
    config = RWKV7Config(hidden_size=64, head_dim=32, num_hidden_layers=1, vocab_size=128)
    model = AutoModelForCausalLM.from_config(config)
    assert all(torch.isfinite(p).all() for p in model.parameters())
