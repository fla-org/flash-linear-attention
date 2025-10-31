
import os

import pytest
import torch

from fla.models import LogLinearMamba2Config, LogLinearMamba2ForCausalLM
from fla.utils import device


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype', 'conv_backend'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}-conv-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 64, torch.bfloat16, 'triton'),
            (4, 4, 1024, 4, 128, torch.bfloat16, 'cuda'),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
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

    config = LogLinearMamba2Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        num_heads=H,
        head_dim=D,
        vocab_size=1000,  # dummy vocab size
    )

    model = LogLinearMamba2ForCausalLM(config).to(device=device, dtype=dtype)
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
