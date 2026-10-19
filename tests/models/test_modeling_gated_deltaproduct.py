# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import GatedDeltaProductConfig
from fla.utils import device

from .utils import _init_weights, check_forward, check_generation


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
        GatedDeltaProductConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'use_forget_gate', 'num_householders', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-use_forget_gate{}-num_householders{}".format(*test))
        for test in [
            (1, 3, 2000, False, 2, torch.float16),
            (2, 4, 4000, True, 3, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    use_forget_gate: bool,
    num_householders: int,
    dtype: torch.dtype,
):
    config = GatedDeltaProductConfig()
    config.num_hidden_layers = L
    config.use_forget_gate = use_forget_gate
    config.num_householders = num_householders
    model = AutoModelForCausalLM.from_config(config)
    model.apply(_init_weights)
    model = model.to(dtype).to(device)
    check_generation(L, B, T, None, None, GatedDeltaProductConfig, dtype, model=model, config=config, tol=3e-3)
