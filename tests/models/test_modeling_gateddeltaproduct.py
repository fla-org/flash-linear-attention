# -*- coding: utf-8 -*-

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import GatedDeltaProductConfig
from fla.utils import device

from .test_modeling_base import run_test_generation, run_test_model_forward_backward
from .testing_utils import init_weights_recursively


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
def test_gateddeltaproduct_modeling(L, B, T, H, D, dtype, use_l2warp):
    run_test_model_forward_backward(L, B, T, H, D, GatedDeltaProductConfig, dtype, use_l2warp)


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize("L", [2])
@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("T", [4000])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gateddeltaproduct_generation(L, B, T, dtype):
    config = GatedDeltaProductConfig()
    config.num_hidden_layers = L
    config.use_forget_gate = False
    config.num_householders = 2
    model = AutoModelForCausalLM.from_config(config)
    model.apply(init_weights_recursively)
    model = model.to(dtype).to(device)
    run_test_generation(L, B, T, None, None, GatedDeltaProductConfig, dtype, model=model, config=config, tol=3e-3)

    config.use_forget_gate = True
    config.num_householders = 3
    model = AutoModelForCausalLM.from_config(config)
    model.apply(init_weights_recursively)
    model = model.to(dtype).to(device)
    run_test_generation(L, B, T, None, None, GatedDeltaProductConfig, dtype, model=model, config=config, tol=3e-3)
