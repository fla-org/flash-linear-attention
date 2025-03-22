# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.modules import FusedKLDivLoss
from fla.utils import device


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ["batchmean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused(B: int, T: int, D: int, V: int, reduction: str, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
    x_weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
    target_x = torch.randn(B * T, D).to(device).to(dtype=dtype)
    target_weight = torch.randn(V, D).to(device).to(dtype=dtype)

    ref = F.kl_div(
        F.linear(x, x_weight).log_softmax(-1),
        F.linear(target_x, target_weight).softmax(-1),
        reduction=reduction
    ).to(dtype)
    do = torch.randn_like(ref).to(device)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, x_weight.grad = x_weight.grad.clone(), None

    tri = FusedKLDivLoss(reduction)(x, target_x, x_weight, target_weight).to(dtype=dtype)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, x_weight.grad = x_weight.grad.clone(), None

    assert torch.allclose(ref, tri, atol=1e-5), f" o: {(ref-tri).abs().max()}"
    assert torch.allclose(ref_dx, tri_dx, atol=1e-5), f"dx: {(ref_dx-tri_dx).abs().max()}"
    assert torch.allclose(ref_dw, tri_dw, atol=1e-5), f"dx_weight: {(ref_dw-tri_dw).abs().max()}"
