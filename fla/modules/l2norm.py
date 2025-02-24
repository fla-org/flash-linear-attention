# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl
from fla.utils import set_torch_device


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["N"]
)
@triton.jit
def l2norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    y = x * rstd
    # Write output
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["N"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
# @triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def l2norm_bwd_kernel(
    X,  # pointer to the input
    # Y, # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row

    # Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    # y = x * rstd
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    # dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
        M, N = x.shape
    assert x.stride(-1) == 1
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]
    # rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    set_torch_device(x)
    l2norm_fwd_kernel[(M,)](
        x,
        y,
        x.stride(0),
        N,
        eps,
        # is_rms_norm,
        BLOCK_N,
        # residual is not None,
        # residual_out is not None,
        # bias is not None,
    )
    return y.reshape(x_shape_og)


def l2norm_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-5
):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    # allocate output
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    # rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    set_torch_device(x)
    l2norm_bwd_kernel[(M,)](
        x,
        dy,
        dx,
        x.stride(0),
        N,
        eps,
        BLOCK_N,
    )
    return dx.reshape(x_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        eps=1e-6,
        output_dtype=None
    ):
        # reshape input data into 2D tensor
        y = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy, *args):
        x, = ctx.saved_tensors
        dx = l2norm_bwd(x, dy, ctx.eps)
        return dx, None, None


def l2_norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)
