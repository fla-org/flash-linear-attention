
import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.models import GatedDeltaProductConfig
from fla.utils import device

from .test_modeling_base import run_test_generation, run_test_model_forward_backward
from .test_modeling_utils import init_weights_recursively


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
            (4, 4, 1024, 4, 128, False, torch.bfloat16),
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
    run_test_model_forward_backward(L, B, T, H, D, GatedDeltaProductConfig, use_l2warp=use_l2warp, dtype=dtype)


# ===================================================================================
# Test for Generation
# ===================================================================================
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
    model.apply(init_weights_recursively)
    model = model.to(dtype).to(device)
    run_test_generation(L, B, T, None, None, GatedDeltaProductConfig, dtype, model=model, config=config, tol=3e-3)
