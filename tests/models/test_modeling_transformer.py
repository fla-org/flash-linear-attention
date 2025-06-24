# -*- coding: utf-8 -*-

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import TransformerConfig
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
def test_modeling(L, B, T, H, D, dtype, use_l2warp):
    run_test_model_forward_backward(L, B, T, H, D, TransformerConfig, dtype, use_l2warp)


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize("L", [2])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [2000])
@pytest.mark.parametrize("H", [3])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_generation(L, B, T, H, D, dtype):
    config = TransformerConfig()
    config.num_hidden_layers = L
    config.num_heads = H
    config.hidden_size = H * D
    config.head_dim = D

    model = AutoModelForCausalLM.from_config(config)
    model.apply(init_weights_recursively)
    model = model.to(dtype).to(device)
    run_test_generation(L, B, T, H, D, TransformerConfig, dtype, model=model, config=config, tol=7e-3)

    config.window_size = 100
    model = AutoModelForCausalLM.from_config(config)
    model.apply(init_weights_recursively)
    model = model.to(dtype).to(device)
    run_test_generation(L, B, T, H, D, TransformerConfig, dtype, model=model, config=config, tol=7e-3)
