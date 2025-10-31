
import os

import pytest
import torch

from fla.models import Mamba2Config, Mamba2ForCausalLM
from fla.utils import device


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype', 'conv_backend'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}-conv-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 64, False, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 128, False, torch.bfloat16, 'cuda'),
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
    conv_backend: str,
):
    """
    Test the forward and backward pass of the Mamba2 model by manually
    instantiating the configuration and the model.
    """
    os.environ['FLA_CONV_BACKEND'] = conv_backend

    # Manually create a consistent configuration
    # The key relationship is: num_heads = expand * hidden_size / head_dim
    # To ensure consistency, we derive hidden_size from other parameters.
    expand = 2
    hidden_size = H * D // expand

    config = Mamba2Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        num_heads=H,
        head_dim=D,
        use_l2warp=use_l2warp,
        vocab_size=1000,  # dummy vocab size
    )

    model = Mamba2ForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    # Create random input tensor
    x = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Forward pass
    y = model(x)

    # Assert output shape is correct
    assert y.logits.shape == (B, T, config.vocab_size)

    # Backward pass
    y.logits.sum().backward()
    print(f"Test test_modeling passed with H={H}, D={D}, backend={conv_backend}.")
