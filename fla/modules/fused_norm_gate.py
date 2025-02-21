# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.
# https://github.com/state-spaces/mamba/blob/fb7b5310fa865dbd62aa059b1e26f2b431363e2a/mamba_ssm/ops/triton/layernorm.py
# Implement residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import contiguous, set_torch_device


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
    O,  # pointer to the gate
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    N,  # number of columns in X
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
    X += row * N
    Y += row * N
    O += row * N
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
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b

    # Swish output gate
    o = tl.load(O + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = y * o * tl.sigmoid(o)

    # Write output
    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd(
    x: torch.Tensor,
    o: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    if residual is not None:
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
    if bias is not None:
        assert bias.shape == (N,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    set_torch_device(x)
    layer_norm_fwd_kernel[(M,)](
        x,
        o,
        y,
        weight,
        bias,
        residual,
        residual_out,
        mean,
        rstd,
        N,
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
    O,  # pointer to the gate
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DO,  # pointer to the gate gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * N
    O += row_start * N
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * N
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * N
    DY += row_start * N
    DX += row_start * N
    DO += row_start * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        o = tl.load(O + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)

        y = xhat * w if HAS_WEIGHT else xhat
        if HAS_BIAS:
            y = y + b
        if RECOMPUTE_OUTPUT:
            tl.store(Y + cols, y, mask=mask)

        sigmoid_o = tl.sigmoid(o)
        do = dy * y * (sigmoid_o + o * sigmoid_o * (1 - sigmoid_o))
        dy = dy * o * sigmoid_o
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
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)
        tl.store(DO + cols, do, mask=mask)

        X += N
        O += N
        if HAS_DRESIDUAL:
            DRESIDUAL += N
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += N
        if RECOMPUTE_OUTPUT:
            Y += N
        DY += N
        DX += N
        DO += N
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    o: torch.Tensor,
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
):
    M, N = x.shape
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
    if bias is not None:
        assert bias.shape == (N,)
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N, dtype=x_dtype, device=x.device)
    do = torch.empty_like(o) if x_dtype is None else torch.empty(M, N, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    dw = torch.empty((sm_count, N), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((sm_count, N), dtype=torch.float, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    set_torch_device(x)
    layer_norm_bwd_kernel[grid](
        x,
        o,
        weight,
        bias,
        y,
        dy,
        dx,
        do,
        dw,
        db,
        dresidual,
        dresidual_in,
        mean,
        rstd,
        M,
        N,
        eps,
        rows_per_program,
        is_rms_norm,
        BLOCK_N,
        dresidual is not None,
        dresidual_in is not None,
        weight is not None,
        bias is not None,
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, do, dw, db, dresidual_in) if not recompute_output else (dx, do, dw, db, dresidual_in, y)


class LayerNormSwishGateFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        o,
        weight,
        bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        o_shape_og = o.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_fwd(
            x, o, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm
        )
        ctx.save_for_backward(residual_out, o, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dy, *args):
        x, o, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, do, dw, db, dresidual_in = layer_norm_bwd(
            dy,
            x,
            o,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            do.reshape(ctx.o_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


class LayerNormSwishGateLinearFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        o,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        o_shape_og = o.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        o = o.reshape(-1, o.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_fwd(
            x,
            o,
            norm_weight,
            norm_bias,
            eps,
            residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(residual_out, o, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.o_shape_og = o_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        x, o, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, do, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(
            dy,
            x,
            o,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            do.reshape(ctx.o_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_swish_gate_fn(
    x,
    o,
    weight,
    bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6
):
    return LayerNormSwishGateFn.apply(
        x,
        o,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        False
    )


def rms_norm_swish_gate_fn(
    x,
    o,
    weight,
    bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6
):
    return LayerNormSwishGateFn.apply(
        x,
        o,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True
    )


def layer_norm_swish_gate_linear_fn(
    x,
    o,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6
):
    return LayerNormSwishGateLinearFn.apply(
        x,
        o,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        False
    )


def rms_norm_swish_gate_linear_fn(
    x,
    o,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6
):
    return LayerNormSwishGateLinearFn.apply(
        x,
        o,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True
    )


class FusedLayerNormSwishGate(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps=1e-5
    ) -> FusedLayerNormSwishGate:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_swish_gate_fn(
            x,
            o,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


class FusedRMSNormSwishGate(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps=1e-5
    ) -> FusedRMSNormSwishGate:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_swish_gate_fn(
            x,
            o,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


class FusedLayerNormSwishGateLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps=1e-5
    ) -> FusedLayerNormSwishGateLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_swish_gate_linear_fn(
            x,
            o,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


class FusedRMSNormSwishGateLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps=1e-5
    ) -> FusedRMSNormSwishGateLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_swish_gate_linear_fn(
            x,
            o,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )
