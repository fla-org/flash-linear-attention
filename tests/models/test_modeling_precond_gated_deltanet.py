# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2023-2025
# Tests for PrecondGatedDeltaNet model

import pytest
import torch

from fla.layers.precond_gated_deltanet import PrecondGatedDeltaNet
from fla.models import PrecondGatedDeltaNetConfig
from fla.utils import device

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Model-level Tests (Forward/Backward Pass)
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
    run_test_model_forward_backward(L, B, T, H, D, PrecondGatedDeltaNetConfig, use_l2warp=use_l2warp, dtype=dtype)


# ===================================================================================
# Model-level Tests (Generation)
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
    run_test_generation(L, B, T, H, D, PrecondGatedDeltaNetConfig, dtype)


# ===================================================================================
# Layer-level Tests
# ===================================================================================
@pytest.mark.parametrize(
    ['B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 256, 4, 64, torch.bfloat16),
            (2, 512, 4, 64, torch.bfloat16),
        ]
    ],
)
def test_layer(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    """Test that the layer works with forward and backward passes."""
    hidden_size = H * D

    layer = PrecondGatedDeltaNet(
        hidden_size=hidden_size,
        num_heads=H,
        head_dim=D,
        expand_v=1,
        mode='chunk',
    ).to(device).to(dtype)

    # Forward pass
    hidden_states = torch.randn(B, T, hidden_size, dtype=dtype, device=device, requires_grad=True)
    output, _, _ = layer(hidden_states)

    assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape} vs {hidden_states.shape}"

    # Backward pass
    loss = output.sum()
    loss.backward()

    assert hidden_states.grad is not None, "hidden_states.grad is None"
    assert hidden_states.grad.shape == hidden_states.shape

    # Check that ATK parameters have gradients
    assert layer.a_atk_proj.weight.grad is not None, "a_atk_proj.weight.grad is None"
    assert layer.b_atk_proj.weight.grad is not None, "b_atk_proj.weight.grad is None"
    assert layer.A_log_atk.grad is not None, "A_log_atk.grad is None"
    assert layer.dt_bias_atk.grad is not None, "dt_bias_atk.grad is None"
