# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
from fla.utils import assert_close, device

from .test_modeling_base import (
    run_test_generate_matches_forward,
    run_test_generation,
    run_test_model_forward_backward,
)


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 4,    torch.bfloat16),
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
):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        GatedDeltaNetConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
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
    run_test_generation(L, B, T, H, D, GatedDeltaNetConfig, dtype)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 2, 64, 8, 64, torch.float32),
        ]
    ],
)
def test_generate_prefill(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generate_matches_forward(L, B, T, H, D, GatedDeltaNetConfig, dtype)


@torch.no_grad()
def test_logits_to_keep_with_labels():
    B, T, H, D, V = 2, 8, 2, 8, 32
    config = GatedDeltaNetConfig(
        hidden_size=H * D,
        num_hidden_layers=1,
        num_heads=H,
        head_dim=D,
        expand_v=1,
        vocab_size=V,
        fuse_cross_entropy=False,
    )
    model = GatedDeltaNetForCausalLM(config).eval().to(device)
    input_ids = torch.randint(V, (B, T), device=device)

    expected = model(input_ids, labels=input_ids)
    actual = model(input_ids, labels=input_ids, logits_to_keep=1)
    inference = model(input_ids, logits_to_keep=1)

    assert actual.logits.shape == expected.logits.shape == (B, T, V)
    assert inference.logits.shape == (B, 1, V)
    assert torch.isfinite(actual.loss)
    assert_close('loss', expected.loss, actual.loss, 1e-6)
