# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import RavenConfig

from .test_modeling_base import run_test_generation, run_test_model_forward_backward
from .test_modeling_utils import create_model_and_config


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'expand_v', 'use_l2warp', 'decay_type', 'use_output_gate', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-ev{}-l2{}-decay{}-og{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, 1, False, 'Mamba2', False, torch.bfloat16),
            (4, 4, 1024, 4, 64, 1, False, 'GLA', True, torch.bfloat16),
            (4, 4, 1024, 4, 64, 2, False, 'Mamba2', False, torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    expand_v: float,
    use_l2warp: bool,
    decay_type: str,
    use_output_gate: bool,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        RavenConfig,
        use_l2warp=use_l2warp,
        add_gumbel_noise=False,
        decay_type=decay_type,
        use_output_gate=use_output_gate,
        expand_v=expand_v,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_rope', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-rope{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, False, torch.float16),
            (2, 4, 2000, 8, 64, True, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_rope: bool,
    dtype: torch.dtype,
):
    if not use_rope:
        run_test_generation(L, B, T, H, D, RavenConfig, dtype)
        return
    # `use_rope=True` exercises the RoPE position offset under left-padded prefill,
    # a path the default config (`use_rope=False`) leaves uncovered.
    model, config = create_model_and_config(RavenConfig, L, H, D, dtype=dtype, use_rope=True)
    run_test_generation(L, B, T, H, D, RavenConfig, dtype, model=model, config=config)
