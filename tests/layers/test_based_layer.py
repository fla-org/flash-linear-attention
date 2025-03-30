# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.based import BasedLinearAttention
from fla.utils import is_intel_alchemist


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [16*12])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Intel A770 do not have enough shared memory for float32"
)
def test_based_layer(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype
):
    from fla.utils import device
    x = torch.randn(B, T, H).to(dtype).to(device).requires_grad_(True)
    dy = torch.randn(B, T, H).to(dtype).to(device)
    model = BasedLinearAttention(H, mode='fused_chunk').to(dtype).to(device)
    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    y2 = model.forward_reference(x)
    y2.backward(dy)
    assert y.allclose(y2, 0, 1e-3), (y - y2).abs().max()
    assert x_grad.allclose(x.grad, 0, 1e-3), (x_grad - x.grad).abs().max()
