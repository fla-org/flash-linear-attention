# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.ops.utils.testing import assert_close
from fla.utils import device, device_platform


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_fused_cross_entropy(B: int, T: int, D: int, V: int, reduction: str, dtype: torch.dtype):
    torch.manual_seed(42)
    logits = torch.randn(B * T, V).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T,)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()

    ref = nn.CrossEntropyLoss(reduction=reduction)(logits, target).to(dtype=dtype)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_d, logits.grad = logits.grad.clone(), None

    tri = FusedCrossEntropyLoss(reduction=reduction)(logits, target).to(dtype=dtype)
    tri.backward(do)
    tri_d, logits.grad = logits.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dl", ref_d, tri_d, ratio=1e-2)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("scale", [1., 0.5])
@pytest.mark.parametrize("reduction", ['mean'])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_fused_linear_cross_entropy(B: int, T: int, D: int, V: int, scale: float, reduction: str, dtype: torch.dtype):
    torch.manual_seed(42)

    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
    target = torch.randint(0, V, (B, T,)).to(device)
    target = torch.cat((target[..., 1:], torch.full_like(target[..., :1], -100)), -1)
    target = target.flatten()
    weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
    bias = torch.randn(V).to(device).to(dtype=dtype).requires_grad_()

    logits = F.linear(x, weight, bias)
    ref = FusedCrossEntropyLoss(logit_scale=scale, reduction=reduction)(logits, target)
    do = torch.randn_like(ref).to(device).to(dtype=dtype)

    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, weight.grad = weight.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    tri = FusedLinearCrossEntropyLoss(logit_scale=scale, reduction=reduction)(x, target, weight, bias)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, weight.grad = weight.grad.clone(), None
    tri_db, bias.grad = bias.grad.clone(), None

    assert_close(" o", ref, tri, ratio=1e-2)
    assert_close("dx", ref_dx, tri_dx, ratio=1e-2)
    assert_close("dw", ref_dw, tri_dw, ratio=1e-2)
    assert_close("db", ref_db, tri_db, ratio=1e-2)
