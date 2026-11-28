# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.models import MomConfig
from fla.models.mom.modeling_mom import MomBlock
from fla.modules import RMSNorm
from fla.utils import assert_close, device_name

from .utils import check_forward, check_generation


@pytest.mark.parametrize('fuse_norm', [False, True], ids=['unfused', 'fused'])
def test_norm(fuse_norm, monkeypatch):
    """Match residual addition, normalization, and gradients with PyTorch branch computations."""
    if fuse_norm and device_name == 'cpu':
        pytest.skip('Fused normalization requires an accelerator.')
    torch.manual_seed(42)
    config = MomConfig(hidden_size=64, num_heads=2, head_dim=32, intermediate_size=128, fuse_norm=fuse_norm)
    block = MomBlock(config=config, layer_idx=0).to(device_name)
    norm_class = RMSNorm if fuse_norm else torch.nn.RMSNorm
    assert isinstance(block.attn_norm, norm_class)
    assert isinstance(block.mlp_norm, norm_class)
    monkeypatch.setattr(block.attn, 'forward', lambda hidden_states, **kwargs: (hidden_states.tanh(), None, None, None))
    monkeypatch.setattr(block.mlp, 'forward', lambda hidden_states, **kwargs: hidden_states.sigmoid())
    x = torch.randn(2, 7, 64, device=device_name, requires_grad=True)
    actual = block(hidden_states=x)[0]
    expected = x + F.rms_norm(x, (64,), block.attn_norm.weight, config.norm_eps).tanh()
    expected = expected + F.rms_norm(expected, (64,), block.mlp_norm.weight, config.norm_eps).sigmoid()
    assert_close('output', expected, actual, 1e-3)
    gradient = torch.randn_like(expected)
    parameters = (x, block.attn_norm.weight, block.mlp_norm.weight)
    expected_gradients = torch.autograd.grad(expected, parameters, gradient)
    actual_gradients = torch.autograd.grad(actual, parameters, gradient)
    for index, (ref, tri) in enumerate(zip(expected_gradients, actual_gradients, strict=True)):
        assert torch.isfinite(tri).all()
        assert_close(f'gradient_{index}', ref, tri, 1e-3)


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.skip(reason="Bug not fixed yet")
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 4,    torch.bfloat16),
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
    attnres_block_size: int | None,
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L,
        B,
        T,
        H,
        D,
        MomConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
        varlen=varlen,
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
    pytest.skip("Known bugs in mom")
    check_generation(L, B, T, H, D, MomConfig, dtype)
