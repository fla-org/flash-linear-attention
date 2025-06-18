# -*- coding: utf-8 -*-

import pytest
import torch

from fla.models import MambaConfig

# Import both base test functions
from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize("L", [4])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_l2warp", [True, False])
def test_mamba_modeling(L, B, T, H, D, dtype, use_l2warp):
    run_test_model_forward_backward(L, B, T, H, D, MambaConfig, dtype, use_l2warp)


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize("L", [1])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [8])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mamba_generation(L, B, T, H, D, dtype):
    run_test_generation(L, B, T, H, D, MambaConfig, dtype)
