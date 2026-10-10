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

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import (
    IS_AMD,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    get_multiprocessor_count,
    input_guard,
)

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


@triton.jit
def _activation_quant_fwd(b_x):
    b_scale = 127.0 / tl.maximum(tl.max(tl.abs(b_x), 0), 1e-5)
    b_scaled = b_x * b_scale
    # triton-ascend does not implement libdevice.round; use round-half-away-from-zero.
    b_y = tl.where(b_scaled >= 0, tl.floor(b_scaled + 0.5), tl.ceil(b_scaled - 0.5))
    return tl.maximum(tl.minimum(b_y, 127), -128) / b_scale


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=['D', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM', 'HAS_BIAS'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_quant_fwd_kernel(
    x,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    stride_x,
    stride_y,
    stride_res,
    stride_res_out,
    D: tl.constexpr,
    eps,
    IS_RMS_NORM: tl.constexpr,
    BD: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    x += i_t * stride_x
    y += i_t * stride_y
    if HAS_RESIDUAL:
        residual += i_t * stride_res
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * stride_res_out
    o_d = tl.arange(0, BD)
    b_x = tl.load(x + o_d, mask=o_d < D, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_res = tl.load(residual + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_x += b_res
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=o_d < D)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(o_d < D, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(o_d < D, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)
    m_d = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd

    b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
    if HAS_BIAS:
        b_y = b_y + b_b

    b_y = _activation_quant_fwd(b_y)

    tl.store(y + o_d, b_y, mask=m_d)


@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=['D', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM', 'HAS_BIAS'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def layer_norm_quant_bwd_kernel(
    x,
    w,
    b,
    y,
    dy,
    dx,
    dw,
    db,
    dres,
    dres_in,
    mean,
    rstd,
    stride_x,
    stride_y,
    stride_dy,
    stride_dx,
    stride_dres,
    stride_dres_in,
    T,
    D: tl.constexpr,
    eps,
    BS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    BD: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0).to(tl.int64)
    bos = i_s * BS
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    x += bos * stride_x
    if HAS_DRESIDUAL:
        dres += bos * stride_dres
    if STORE_DRESIDUAL:
        dres_in += bos * stride_dres_in
    dy += bos * stride_dy
    dx += bos * stride_dx
    if RECOMPUTE_OUTPUT:
        y += bos * stride_y
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)
    eos = min((i_s + 1) * BS, T)
    for i_t in range(bos, eos):
        b_x = tl.load(x + o_d, mask=m_d, other=0).to(tl.float32)
        b_dy = tl.load(dy + o_d, mask=m_d, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(m_d, b_xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
            if HAS_BIAS:
                b_y = b_y + b_b

            b_y = _activation_quant_fwd(b_y)

            tl.store(y + o_d, b_y, mask=m_d)
        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_xhat * b_c1 + b_c2)) * b_rstd
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
        if HAS_DRESIDUAL:
            b_dres = tl.load(dres + o_d, mask=m_d, other=0).to(tl.float32)
            b_dx += b_dres
        if STORE_DRESIDUAL:
            tl.store(dres_in + o_d, b_dx, mask=m_d)
        tl.store(dx + o_d, b_dx, mask=m_d)

        x += stride_x
        if HAS_DRESIDUAL:
            dres += stride_dres
        if STORE_DRESIDUAL:
            dres_in += stride_dres_in
        if RECOMPUTE_OUTPUT:
            y += stride_y
        dy += stride_dy
        dx += stride_dx
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=m_d)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=m_d)


def layer_norm_quant_fwd(
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
    T, D = x.shape
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    layer_norm_quant_fwd_kernel[(T,)](
        x=x,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        stride_x=x.stride(0),
        stride_y=y.stride(0),
        stride_res=residual.stride(0) if residual is not None else 0,
        stride_res_out=residual_out.stride(0) if residual_out is not None else 0,
        D=D,
        eps=eps,
        IS_RMS_NORM=is_rms_norm,
        BD=BD,
        HAS_RESIDUAL=residual is not None,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    return y, mean, rstd, residual_out if residual_out is not None else x


def layer_norm_quant_bwd(
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
    T, D = x.shape
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    NS = get_multiprocessor_count(x.device.index)
    dw = torch.empty((NS, D), dtype=torch.float32, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float32, device=bias.device) if bias is not None else None
    BS = triton.cdiv(T, NS)
    grid = (NS,)
    layer_norm_quant_bwd_kernel[grid](
        x=x,
        w=weight,
        b=bias,
        y=y,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        dres=dresidual,
        dres_in=dresidual_in,
        mean=mean,
        rstd=rstd,
        stride_x=x.stride(0),
        stride_y=0 if not recompute_output else y.stride(0),
        stride_dy=dy.stride(0),
        stride_dx=dx.stride(0),
        stride_dres=dresidual.stride(0) if dresidual is not None else 0,
        stride_dres_in=dresidual_in.stride(0) if dresidual_in is not None else 0,
        T=T,
        D=D,
        eps=eps,
        BS=BS,
        IS_RMS_NORM=is_rms_norm,
        BD=BD,
        HAS_DRESIDUAL=dresidual is not None,
        STORE_DRESIDUAL=dresidual_in is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    # the residual shares the input gradient when their dtypes match
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)


class LayerNormLinearQuantFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
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
        y, mean, rstd, residual_out = layer_norm_quant_fwd(
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
    @autocast_custom_bwd
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
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_quant_bwd(
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


def layer_norm_linear_quant(
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
    return LayerNormLinearQuantFunction.apply(
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
    return layer_norm_linear_quant(
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


def bit_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply RMS normalization and a linear projection with quantized activations and weights."""
    return layer_norm_linear_quant(
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
        return layer_norm_linear_quant(
            x=x,
            norm_weight=self.norm.weight,
            norm_bias=self.norm.bias,
            linear_weight=self.weight,
            linear_bias=self.bias,
            is_rms_norm=True,
        )
