# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp as standalone_l2_warp
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("H", [256])
@pytest.mark.parametrize("V", [2000])
@pytest.mark.parametrize("l2_penalty_factor", [1e-4, 1])
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST is True,
    reason="Intel Triton Failure",
)
def test_fused_linear_cross_entropy_l2_warp(
    B: int,
    T: int,
    H: int,
    V: int,
    l2_penalty_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    lm_head = nn.Linear(H, V, bias=True, device=device, dtype=dtype)
    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device=device)

    ignore_index = -100
    shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], ignore_index)), 1)

    ref_criterion = nn.CrossEntropyLoss()

    ref_logits = F.linear(x.view(-1, H), lm_head.weight, lm_head.bias)
    ref_loss_ce = ref_criterion(ref_logits.view(B * T, V), shift_labels.view(-1))
    ref_loss = standalone_l2_warp(ref_loss_ce, ref_logits.view(B, T, V), l2_penalty_factor)

    ref_loss.backward()
    ref_x_grad = x.grad.clone()
    ref_w_grad = lm_head.weight.grad.clone()
    ref_b_grad = lm_head.bias.grad.clone()

    x.grad = None
    lm_head.zero_grad()

    fused_criterion = FusedLinearCrossEntropyLoss(
        l2_penalty_factor=l2_penalty_factor,
        use_l2warp=True,  # Make sure to enable it
    )

    fused_loss = fused_criterion(x, shift_labels, lm_head.weight, lm_head.bias)

    fused_loss.backward()
    fused_x_grad = x.grad.clone()
    fused_w_grad = lm_head.weight.grad.clone()
    fused_b_grad = lm_head.bias.grad.clone()

    ratio = 4e-3 if dtype == torch.bfloat16 else 1e-3

    assert_close("Loss", ref_loss, fused_loss, ratio)
    assert_close("dx", ref_x_grad, fused_x_grad, ratio)
    assert_close("dw", ref_w_grad, fused_w_grad, ratio)
    assert_close("db", ref_b_grad, fused_b_grad, ratio)
