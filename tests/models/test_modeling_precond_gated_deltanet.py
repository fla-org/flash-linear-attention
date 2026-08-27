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
from fla.utils import assert_close, device

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


def test_naive_layer_varlen():
    B, T, H, D = 3, 7, 2, 8
    torch.manual_seed(42)
    layer = PrecondGatedDeltaNet(
        hidden_size=H * D,
        num_heads=H,
        head_dim=D,
        expand_v=1,
        mode='naive',
    ).to(device)
    hidden_states = torch.randn(B, T, H * D, device=device, requires_grad=True)
    seq_start = [0, 3, 5]
    attention_mask = torch.arange(T, device=device) >= torch.tensor(seq_start, device=device)[:, None]

    actual = layer(hidden_states, attention_mask=attention_mask)[0][attention_mask]
    expected = torch.cat([
        layer(hidden_states[i:i + 1, start:])[0].squeeze(0) for i, start in enumerate(seq_start)
    ])

    assert_close('o', expected, actual, 1e-5)

    do = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * do).sum(), hidden_states)[0]
    expected_grad = torch.autograd.grad((expected * do).sum(), hidden_states)[0]
    assert_close('dx', expected_grad, actual_grad, 1e-5)
