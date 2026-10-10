# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# BitLinear with fused normalization and quantized linear projection.
# [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
# [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)
#
# adapted from https://github.com/ridgerchu/matmulfreellm/

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import IS_AMD, autotune_cache_kwargs, get_multiprocessor_count, input_guard, require_version

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize activations to 8 bits per token and return dequantized values."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Quantize weights to 1.58 bits per tensor and return dequantized values."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_fwd_kernel_quant(
    X,
    Y,
    W,
    B,
    RESIDUAL,
    RESIDUAL_OUT,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    N,
    eps,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
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
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd

    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b

    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
    y = tl.extra.libdevice.round(y * scale)
    y = tl.maximum(tl.minimum(y, 127), -128) / scale

    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd_quant(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    layer_norm_fwd_kernel_quant[(M,)](
        X=x,
        Y=y,
        W=weight,
        B=bias,
        RESIDUAL=residual,
        RESIDUAL_OUT=residual_out,
        Mean=mean,
        Rstd=rstd,
        stride_x_row=x.stride(0),
        stride_y_row=y.stride(0),
        stride_res_row=residual.stride(0) if residual is not None else 0,
        stride_res_out_row=residual_out.stride(0) if residual_out is not None else 0,
        N=N,
        eps=eps,
        IS_RMS_NORM=is_rms_norm,
        BLOCK_N=BLOCK_N,
        HAS_RESIDUAL=residual is not None,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.heuristics({
    "RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_bwd_kernel(
    X,
    W,
    B,
    Y,
    DY,
    DX,
    DW,
    DB,
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dres_in_row,
    M,
    N,
    eps,
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w if HAS_WEIGHT else xhat
            if HAS_BIAS:
                y = y + b

            scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
            y = tl.extra.libdevice.round(y * scale)
            y = tl.maximum(tl.minimum(y, 127), -128) / scale

            tl.store(Y + cols, y, mask=mask)
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
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    mean: torch.Tensor | None,
    rstd: torch.Tensor,
    dresidual: torch.Tensor | None = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype | None = None,
    recompute_output: bool = False,
):
    M, N = x.shape
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = get_multiprocessor_count(x.device.index)
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device) if weight is not None else None
    _db = torch.empty((sm_count, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    layer_norm_bwd_kernel[grid](
        X=x,
        W=weight,
        B=bias,
        Y=y,
        DY=dy,
        DX=dx,
        DW=_dw,
        DB=_db,
        DRESIDUAL=dresidual,
        DRESIDUAL_IN=dresidual_in,
        Mean=mean,
        Rstd=rstd,
        stride_x_row=x.stride(0),
        stride_y_row=0 if not recompute_output else y.stride(0),
        stride_dy_row=dy.stride(0),
        stride_dx_row=dx.stride(0),
        stride_dres_row=dresidual.stride(0) if dresidual is not None else 0,
        stride_dres_in_row=dresidual_in.stride(0) if dresidual_in is not None else 0,
        M=M,
        N=N,
        eps=eps,
        rows_per_program=rows_per_program,
        IS_RMS_NORM=is_rms_norm,
        BLOCK_N=BLOCK_N,
        HAS_DRESIDUAL=dresidual is not None,
        STORE_DRESIDUAL=dresidual_in is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    dw = _dw.sum(0).to(weight.dtype) if weight is not None else None
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    # the residual shares the input gradient when their dtypes match
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)


class LayerNormLinearQuantFn(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor | None,
        norm_bias: torch.Tensor | None,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else None)
        y, mean, rstd, residual_out = layer_norm_fwd_quant(
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            eps=eps,
            residual=residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_dtype('cuda'),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else y.dtype
        linear_weight = weight_quant(linear_weight).to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(input=y.to(linear_weight.dtype), weight=linear_weight, bias=linear_bias)
        # recompute y in backward to save memory
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout: torch.Tensor, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(input=dout, weight=linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(
            dy=dy,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            eps=ctx.eps,
            mean=mean,
            rstd=rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
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
        )


def layer_norm_linear_quant_fn(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return LayerNormLinearQuantFn.apply(
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
    )


def rms_norm_linear_quant(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return layer_norm_linear_quant_fn(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True,
    )


@require_version("triton>=3.0", "Triton >= 3.0 is required to do online quantization.")
def bit_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply RMS normalization and a linear projection with quantized activations and weights."""
    return layer_norm_linear_quant_fn(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=weight,
        linear_bias=bias,
        is_rms_norm=True,
    )


class BitLinear(nn.Linear):
    """
    RMS-normalized linear layer with 8-bit activations and ternary weights.

    Quantization uses a straight-through estimator during training.
    Efficient deployment requires specialized kernels.

    Args:
        in_features (int):
            Size of each input sample.
        out_features (int):
            Size of each output sample.
        bias (bool, Optional):
            Whether to allocate an additive bias. Default: `False`.
        norm_eps (float, Optional):
            Epsilon for RMS normalization. Default: 1e-8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, norm_eps: float = 1e-8) -> None:
        from fla.modules import RMSNorm

        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.norm = RMSNorm(hidden_size=in_features, eps=norm_eps, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().extra_repr()}, norm_eps={self.norm.eps})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        x_norm = self.norm(x)

        # straight-through gradients bypass quantization
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(input=x_quant, weight=w_quant)

        return y


class FusedBitLinear(BitLinear):
    """BitLinear with fused RMS normalization and activation quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm_linear_quant_fn(
            x=x,
            norm_weight=self.norm.weight,
            norm_bias=self.norm.bias,
            linear_weight=self.weight,
            linear_bias=self.bias,
            is_rms_norm=True,
        )
