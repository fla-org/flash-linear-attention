# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.based import BasedLinearAttention


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("H", [2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_based(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype
):
    x = torch.randn(B, T, H).to(dtype).cuda().requires_grad_(True)
    dy = torch.randn(B, T, H).to(dtype).cuda()
    model = BasedLinearAttention(H, mode='chunk').to(dtype).cuda()
    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    y2 = model.forward_reference(x)
    y2.backward(dy)
    assert y.allclose(y2, 0, 1e-3), (y - y2).abs().max()
    assert x_grad.allclose(x.grad, 0, 1e-3), (x_grad - x.grad).abs().max()
