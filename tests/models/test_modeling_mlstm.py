# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.mlstm import MLSTM
from fla.models import MLSTMConfig, MLSTMForCausalLM
from fla.utils import device

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ["L", "B", "T", "H", "D", "use_l2warp", "dtype"],
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
    run_test_model_forward_backward(L, B, T, H, D, MLSTMConfig, use_l2warp=use_l2warp, dtype=dtype)


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize(
    ["L", "B", "T", "H", "D", "dtype"],
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
    run_test_generation(L, B, T, H, D, MLSTMConfig, dtype)


@pytest.mark.parametrize('num_hidden_layers', [1, 2])
def test_initialization(num_hidden_layers: int):
    torch.manual_seed(42)
    config = MLSTMConfig(
        hidden_size=64,
        num_hidden_layers=num_hidden_layers,
        num_heads=2,
        proj_factor=1,
        hidden_ratio=1,
        vocab_size=32,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )
    model = MLSTMForCausalLM(config).to(device)
    for block in model.model.layers:
        layer = block.attn
        assert not hasattr(layer, 'reset_parameters')
        assert torch.count_nonzero(layer.fgate.weight) == 0
        assert torch.count_nonzero(layer.igate.weight) == 0
        assert torch.equal(layer.learnable_skip, torch.ones_like(layer.learnable_skip))
        assert layer.fgate.bias.requires_grad
        expected_bias = torch.linspace(3, 6, config.num_heads, device=device)
        assert torch.equal(layer.fgate.bias, expected_bias)
    input_ids = torch.randint(config.vocab_size, (1, 7), device=device)
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    for block in model.model.layers:
        assert block.attn.fgate.bias.grad is not None
        assert torch.isfinite(block.attn.fgate.bias.grad).all()

    layer = MLSTM(hidden_size=64, num_heads=2, proj_factor=1).to(device)
    assert layer.fgate.bias.requires_grad
    hidden_states = torch.randn(1, 7, 64, device=device, requires_grad=True)
    output, _, _ = layer(hidden_states=hidden_states)
    output.square().mean().backward()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
