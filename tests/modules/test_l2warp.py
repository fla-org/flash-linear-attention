# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp as standalone_l2_warp
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("H", [256])
@pytest.mark.parametrize("V", [2000])
@pytest.mark.parametrize("l2_penalty_factor", [1e-4, 1])
@pytest.mark.parametrize("grad_scale", [1.0, 3.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
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
    grad_scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    lm_head = nn.Linear(H, V, bias=True, device=device, dtype=dtype)
    x = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device=device)

    ignore_index = -100
    shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], ignore_index)), 1)
    shift_labels[:, ::2] = ignore_index

    ref_criterion = nn.CrossEntropyLoss()

    ref_logits = F.linear(x.view(-1, H), lm_head.weight, lm_head.bias)
    ref_loss_ce = ref_criterion(ref_logits.view(B * T, V), shift_labels.view(-1))
    ref_loss = standalone_l2_warp(ref_loss_ce, ref_logits.view(B, T, V), l2_penalty_factor)

    # scale the loss so backward sees a non-unit upstream gradient when grad_scale != 1
    (grad_scale * ref_loss).backward()
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

    (grad_scale * fused_loss).backward()
    fused_x_grad = x.grad.clone()
    fused_w_grad = lm_head.weight.grad.clone()
    fused_b_grad = lm_head.bias.grad.clone()

    ratio = 4e-3 if dtype == torch.float16 else 1e-3

    assert_close("Loss", ref_loss, fused_loss, ratio)
    assert_close("dx", ref_x_grad, fused_x_grad, ratio)
    assert_close("dw", ref_w_grad, fused_w_grad, ratio)
    assert_close("db", ref_b_grad, fused_b_grad, ratio)


@pytest.mark.parametrize("grad_scale", [3.0])
@pytest.mark.parametrize(
    ("logits_dtype", "loss_dtype"),
    [
        pytest.param(torch.float32, torch.float32, id="fp32-logits-fp32-loss"),
        pytest.param(torch.bfloat16, torch.float32, id="bf16-logits-fp32-loss"),
    ],
)
@pytest.mark.skipif(
    IS_INTEL_ALCHEMIST is True,
    reason="Intel Triton Failure",
)
def test_l2_warp_grad_output_scaling(grad_scale: float, logits_dtype: torch.dtype, loss_dtype: torch.dtype):
    torch.manual_seed(42)

    logits = torch.randn(2, 64, 100, device=device, dtype=logits_dtype, requires_grad=True)
    loss = torch.randn((), device=device, dtype=loss_dtype, requires_grad=True)
    wrapped = standalone_l2_warp(loss, logits, 1.0)

    dloss_1, dlogits_1 = torch.autograd.grad(
        wrapped,
        (loss, logits),
        grad_outputs=torch.ones((), device=device, dtype=loss_dtype),
        retain_graph=True,
    )
    dloss_k, dlogits_k = torch.autograd.grad(
        wrapped,
        (loss, logits),
        grad_outputs=torch.full((), grad_scale, device=device, dtype=loss_dtype),
    )

    assert dlogits_k.dtype == logits_dtype
    assert_close("  dloss", dloss_k, grad_scale * dloss_1, 1e-5)
    assert_close("dlogits", dlogits_k, grad_scale * dlogits_1, 1e-5)
