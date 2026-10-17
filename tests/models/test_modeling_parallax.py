# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import ParallaxConfig

from .utils import check_forward, check_generation


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 2,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 4,    torch.bfloat16),
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
    attnres_block_size: int | None,
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L,
        B,
        T,
        H,
        D,
        ParallaxConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.usefixtures('triton_conv')
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
    check_generation(L, B, T, H, D, ParallaxConfig, dtype)
