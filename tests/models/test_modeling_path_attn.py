# -*- coding: utf-8 -*-

import pytest
import torch

from fla.models import PaTHAttentionConfig
from fla.utils import check_shared_mem

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, False, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, torch.bfloat16),
        ]
    ]
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
    if not check_shared_mem('hopper') and D > 64:
        pytest.skip("For PaTHAttention, head dimension 128 only supported on Hopper or later. Stay tuned for Ampere support!")
    run_test_model_forward_backward(L, B, T, H, D, PaTHAttentionConfig, use_l2warp=use_l2warp, dtype=dtype)

# ===================================================================================
# Test for Generation
# ===================================================================================


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
            (2, 2, 2000, 8, 128, torch.float16),
        ]
    ]
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 64:
        pytest.skip("For PaTHAttention, head dimension 128 only supported on Hopper or later. Stay tuned for Ampere support!")
    run_test_generation(L, B, T, H, D, PaTHAttentionConfig, dtype)
