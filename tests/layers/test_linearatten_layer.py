# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.linear_attn import LinearAttention


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("H", [2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_linearatten_layer(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype
):
    from fla.utils import device
    x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
    dy = torch.randn(B, T, H).to(dtype).to(device)
    model = LinearAttention(hidden_size=H, mode='chunk').to(dtype).to(device)
    y = model(x)
    y.backward(dy, retain_graph=True)
    
    # the correct of gradient will be checked in tests/ops
    print('success')
