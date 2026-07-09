# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Shared gate helpers reused across delta-rule family ops (KDA, GDN, ...).

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit
def fused_beta_sigmoid_fwd_kernel(
    x,
    y,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_y = scale * tl.sigmoid(b_x)
    tl.store(y + offs, b_y.to(y.dtype.element_ty), mask=mask)


@triton.jit
def fused_beta_sigmoid_bwd_kernel(
    x,
    dy,
    dx,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_dy = tl.load(dy + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    b_dx = b_dy * scale * b_y * (1.0 - b_y)
    tl.store(dx + offs, b_dx.to(dx.dtype.element_ty), mask=mask)


_BETA_SIGMOID_BLOCK_SIZE = 2048
_BETA_SIGMOID_NUM_WARPS = 8


def fused_beta_sigmoid_fwd(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    y = torch.empty_like(x, dtype=torch.float32)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, _BETA_SIGMOID_BLOCK_SIZE),)
    fused_beta_sigmoid_fwd_kernel[grid](
        x,
        y,
        scale,
        n_elements,
        BLOCK_SIZE=_BETA_SIGMOID_BLOCK_SIZE,
        num_warps=_BETA_SIGMOID_NUM_WARPS,
    )
    return y


def fused_beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    dx = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, _BETA_SIGMOID_BLOCK_SIZE),)
    fused_beta_sigmoid_bwd_kernel[grid](
        x,
        dy,
        dx,
        scale,
        n_elements,
        BLOCK_SIZE=_BETA_SIGMOID_BLOCK_SIZE,
        num_warps=_BETA_SIGMOID_NUM_WARPS,
    )
    return dx


class BetaSigmoidFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        y = fused_beta_sigmoid_fwd(x, scale)
        ctx.save_for_backward(x)
        ctx.scale = scale
        return y

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dy: torch.Tensor):
        (x,) = ctx.saved_tensors
        dx = fused_beta_sigmoid_bwd(x, dy, ctx.scale)
        return dx.type_as(x), None


def fused_beta_sigmoid(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return BetaSigmoidFunction.apply(x, scale)
