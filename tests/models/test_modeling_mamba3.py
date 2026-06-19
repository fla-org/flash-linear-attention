# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util

import pytest
import torch

# Mamba-3 mixer relies on upstream `mamba_ssm` kernels; skip the whole module
# if the prefill kernel is not installed in the current environment.
pytest.importorskip("mamba_ssm.ops.triton.mamba3.mamba3_siso_combined")

from fla.models import Mamba3Config, Mamba3ForCausalLM  # noqa: E402
from fla.utils import device  # noqa: E402

from .test_modeling_base import run_test_generation  # noqa: E402


def _mamba3_decode_kernels_available() -> bool:
    """Decode requires the cute step kernel and the rotary step kernel,
    which are independent of the Triton prefill kernel."""
    for mod in (
        "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
        "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step",
    ):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'is_mimo', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-mimo{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, False, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, False, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, False, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, True, torch.bfloat16),
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
    is_mimo: bool,
    dtype: torch.dtype,
):
    """
    Test the forward and backward pass of the Mamba3 model by manually
    instantiating the configuration and the model.
    """
    # Mirror Mamba2's derivation so num_heads = H, head_dim = D.
    expand = 2
    hidden_size = H * D // expand

    config = Mamba3Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        head_dim=D,
        use_l2warp=use_l2warp,
        is_mimo=is_mimo,
        mimo_rank=2 if is_mimo else 1,
        vocab_size=1000,
    )

    model = Mamba3ForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = model(x)
    assert y.logits.shape == (B, T, config.vocab_size)
    y.logits.sum().backward()


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.skipif(
    not _mamba3_decode_kernels_available(),
    reason="Mamba-3 decode kernels (cute step_fn / rotary) are not installed.",
)
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
    expand = 2
    hidden_size = H * D // expand

    config = Mamba3Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        head_dim=D,
        vocab_size=1000,
    )
    model = Mamba3ForCausalLM(config).to(device=device, dtype=dtype)
    run_test_generation(L, B, T, H, D, Mamba3Config, dtype, model=model, config=config)
