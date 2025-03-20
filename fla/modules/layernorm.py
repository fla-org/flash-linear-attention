# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.
# https://github.com/state-spaces/mamba/blob/fb7b5310fa865dbd62aa059b1e26f2b431363e2a/mamba_ssm/ops/triton/layernorm.py
# Implement residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.utils import get_multiprocessor_count, input_guard


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False
):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    return out if not prenorm else (out, x)


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False
):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    N,  # number of columns in X
    G,  # number of groups
    eps,  # epsilon to avoid division by zero
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = row % G
    X += row * N
    Y += row * N
    if HAS_RESIDUAL:
        RESIDUAL += row * N
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + group * N + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + group * N + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd

    y = tl.fma(x_hat, w, b) if HAS_WEIGHT and HAS_BIAS else \
        x_hat * w if HAS_WEIGHT else \
        x_hat + b if HAS_BIAS else x_hat
    # Write output
    y = tl.cast(y, dtype=Y.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N, G = *x.shape, num_groups
    if residual is not None:
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (G * N,)
    if bias is not None:
        assert bias.shape == (G * N,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    layer_norm_fwd_kernel[(M,)](
        x,
        y,
        weight,
        bias,
        residual,
        residual_out,
        mean,
        rstd,
        N,
        G,
        eps,
        is_rms_norm,
        BLOCK_N,
        residual is not None,
        residual_out is not None,
        weight is not None,
        bias is not None,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.heuristics({
    "RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,  # number of rows in X
    N,  # number of columns in X
    G,  # number of groups
    rows_per_program,
    programs_per_group,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    group_id, program_id_in_group = row_block_id // programs_per_group, row_block_id % programs_per_group

    row_start = group_id + program_id_in_group * G * rows_per_program
    row_end = min(row_start + G * rows_per_program, M)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    if HAS_WEIGHT:
        w = tl.load(W + group_id * N + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + group_id * N + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row in range(row_start, row_end, G):
        # Load data to SRAM
        x = tl.load(X + row * N + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + row * N + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w if HAS_WEIGHT else xhat
            if HAS_BIAS:
                y = y + b
            tl.store(Y + row * N + cols, y, mask=mask)
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + row * N + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        dx = tl.cast(dx, dtype=DX.dtype.element_ty, fp_downcast_rounding="rtne")
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + row * N + cols, dx, mask=mask)
        tl.store(DX + row * N + cols, dx, mask=mask)

    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
    num_groups: int = 1
):
    M, N, G = *x.shape, num_groups
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (G * N,)
    if bias is not None:
        assert bias.shape == (G * N,)
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # each program handles one group only
    S = triton.cdiv(get_multiprocessor_count(x.device.index), G) * G
    dw = torch.empty((S, N), dtype=torch.float32, device=weight.device) if weight is not None else None
    db = torch.empty((S, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = triton.cdiv(M, S)
    programs_per_group = S // G
    grid = (S,)
    layer_norm_bwd_kernel[grid](
        x,
        weight,
        bias,
        y,
        dy,
        dx,
        dw,
        db,
        dresidual,
        dresidual_in,
        mean,
        rstd,
        M,
        N,
        G,
        rows_per_program,
        programs_per_group,
        is_rms_norm,
        BLOCK_N,
        dresidual is not None,
        dresidual_in is not None,
        weight is not None,
        bias is not None,
    )
    dw = dw.view(G, -1, N).sum(1).to(weight).view_as(weight) if weight is not None else None
    db = db.view(G, -1, N).sum(1).to(bias).view_as(bias) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual=None,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        num_groups=1
    ):
        x_shape_og = x.shape

        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        # reshape input data into 2D tensor
        x = x.reshape(-1, (x.shape[-1] // num_groups))
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            num_groups=num_groups
        )
        ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dy, *args):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, (dy.shape[-1] // ctx.num_groups))
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in = layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            num_groups=ctx.num_groups
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
            None
        )


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False
):
    return LayerNormFunction.apply(
        x,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm
    )


def group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    return LayerNormFunction.apply(
        x,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        num_groups
    )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False
):
    return LayerNormFunction.apply(
        x,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True
    )


def layer_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    return LayerNormLinearFunction.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        num_groups
    )


def rms_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False
):
    return layer_norm_linear(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True
    )


def group_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    return layer_norm_linear(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=is_rms_norm,
        num_groups=num_groups
    )


class LayerNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> LayerNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


class GroupNorm(nn.Module):

    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> GroupNorm:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return group_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            num_groups=self.num_groups
        )


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> RMSNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class LayerNormLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        num_groups=1
    ):
        x_shape_og = x.shape

        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        # reshape input data into 2D tensor
        x = x.reshape(-1, (x.shape[-1] // num_groups))
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_fwd(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            num_groups=num_groups
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dy = dy.reshape(-1, (dy.shape[-1] // ctx.num_groups))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(
            dy,
            x,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
            num_groups=ctx.num_groups
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y.view(-1, linear_weight.shape[-1]))
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
            None
        )


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> LayerNormLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear(
            x=x,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False
        )


class GroupNormLinear(nn.Module):

    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> GroupNormLinear:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear(
            x=x,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            num_groups=self.num_groups
        )


class RMSNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> RMSNormLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear(
            x=x,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True
        )


class NormParallel(ParallelStyle):

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False):
        super().__init__()
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            # if the passed in input DTensor is not sharded on the sequence dim, we need to redistribute it
            if input_tensor.placements != sequence_sharding:
                input_tensor = input_tensor.redistribute(
                    placements=sequence_sharding, async_op=True
                )
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, sequence_sharding, run_check=False
            )
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),
            partial(self._prepare_output_fn, self.use_local_output),
        )
