# -*- coding: utf-8 -*-

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import MLAConfig
from fla.utils import device

from .test_modeling_base import run_test_generation
from .test_modeling_utils import init_weights_recursively


# TODO: add forward backward test


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
    config = MLAConfig()
    config.num_hidden_layers = L
    config.num_heads = H
    config.hidden_size = H * D

    # MLA specific params
    config.q_lora_rank = None
    config.qk_rope_head_dim = D // 2  # partial rope, half of D
    config.kv_lora_rank = 256
    config.v_head_dim = D
    config.qk_nope_head_dim = D
    config.qk_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
    config.rope_scaling = None

    model = AutoModelForCausalLM.from_config(config)
    model.apply(init_weights_recursively)
    model = model.to(dtype).to(device)
    run_test_generation(L, B, T, H, D, MLAConfig, dtype, model=model, config=config, tol=7e-3)
