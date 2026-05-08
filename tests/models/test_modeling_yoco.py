# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import YOCOConfig, YOCOForCausalLM
from fla.utils import device

from .test_modeling_base import run_test_generation


def _create_yoco_config(
    L: int,
    H: int,
    D: int,
    use_l2warp: bool = False,
    vocab_size: int = 1000,
):
    return YOCOConfig(
        num_hidden_layers=L,
        num_self_decoder_layers=L // 2,
        hidden_size=H * D,
        self_decoder_attn={
            'type': 'gated_deltanet',
            'mode': 'chunk',
            'num_heads': H,
            'num_v_heads': H,
            'head_dim': D,
        },
        cross_decoder_attn={
            'num_heads': H,
            'num_kv_heads': H,
        },
        intermediate_size=4 * H * D,
        vocab_size=vocab_size,
        use_l2warp=use_l2warp,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, torch.bfloat16),
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
):
    config = _create_yoco_config(L, H, D, use_l2warp=use_l2warp)
    model = YOCOForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = model(x)
    assert y.logits.shape == (B, T, config.vocab_size)
    y.logits.sum().backward()


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (4, 4, 2000, 8, 64, torch.float16),
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
    config = _create_yoco_config(L, H, D, vocab_size=128)
    model = YOCOForCausalLM(config)
    run_test_generation(L, B, T, H, D, YOCOConfig, dtype, model=model, config=config, tol=3e-3)
