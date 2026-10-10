# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Fused normalization and quantized linear projection.
# [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
# [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)
#
# adapted from https://github.com/ridgerchu/matmulfreellm/

from __future__ import annotations

import torch
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


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Quantize weights to 1.58 bits per tensor and return dequantized values."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    y = (w * scale).round().clamp_(-1, 1) / scale
    return y


@triton.jit
def _activation_quant_fwd(b_x):
    b_scale = 127.0 / tl.maximum(tl.max(tl.abs(b_x), 0), 1e-5)
    b_scaled = b_x * b_scale
    # triton-ascend does not implement libdevice.round; use round-half-away-from-zero.
    b_y = tl.where(b_scaled >= 0, tl.floor(b_scaled + 0.5), tl.ceil(b_scaled - 0.5))
    return tl.maximum(tl.minimum(b_y, 127), -128) / b_scale


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM', 'HAS_BIAS'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_quant_fwd_kernel(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)

    o_d = tl.arange(0, BD).to(tl.int64)
    m_d = o_d < D

    p_x = x + i_t * D + o_d
    b_x = tl.load(p_x, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = res + i_t * D + o_d
        b_x += tl.load(p_res, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = res_out + i_t * D + o_d
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        p_mean = mean + i_t
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty))
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = rstd + i_t
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    b_y = _activation_quant_fwd(b_y)

    p_y = y + i_t * D + o_d
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_d)


@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
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
    T,
    D: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0).to(tl.int64)

    o_d = tl.arange(0, BD).to(tl.int64)
    m_d = o_d < D

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_s * BS, min((i_s + 1) * BS, T)):
        p_x = x + i_t * D + o_d
        p_dy = dy + i_t * D + o_d
        b_x = tl.load(p_x, mask=m_d, other=0.0).to(tl.float32)
        b_dy = tl.load(p_dy, mask=m_d, other=0.0).to(tl.float32)
        if not IS_RMS_NORM:
            p_mean = mean + i_t
            b_mean = tl.load(p_mean)
        p_rstd = rstd + i_t
        b_rstd = tl.load(p_rstd)

        b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_x_hat = tl.where(m_d, b_x_hat, 0.0)
        if RECOMPUTE_OUTPUT:
            b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
            if HAS_BIAS:
                b_y = b_y + b_b

            b_y = _activation_quant_fwd(b_y)

            p_y = y + i_t * D + o_d
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_d)
        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_x_hat
        if HAS_BIAS:
            b_db += b_dy
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_x_hat * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_x_hat * b_c1 + b_c2)) * b_rstd
        else:
            b_c1 = tl.sum(b_x_hat * b_wdy, axis=0) / D
            b_dx = (b_wdy - b_x_hat * b_c1) * b_rstd
        if HAS_DRESIDUAL:
            p_dres = dres + i_t * D + o_d
            b_dx += tl.load(p_dres, mask=m_d, other=0.0).to(tl.float32)

        if STORE_DRESIDUAL:
            p_dres_in = dres_in + i_t * D + o_d
            tl.store(p_dres_in, b_dx.to(p_dres_in.dtype.element_ty), mask=m_d)

        p_dx = dx + i_t * D + o_d
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=m_d)

    if HAS_WEIGHT:
        p_dw = dw + i_s * D + o_d
        tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), mask=m_d)
    if HAS_BIAS:
        p_db = db + i_s * D + o_d
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_d)


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
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        res_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        res_out = None
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
        res=residual,
        res_out=res_out,
        mean=mean,
        rstd=rstd,
        eps=eps,
        D=D,
        BD=BD,
        IS_RMS_NORM=is_rms_norm,
        HAS_RESIDUAL=residual is not None,
        STORE_RESIDUAL_OUT=res_out is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    return y, mean, rstd, res_out if res_out is not None else x


def layer_norm_quant_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    mean: torch.Tensor | None,
    rstd: torch.Tensor,
    dres: torch.Tensor | None = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype | None = None,
    recompute_output: bool = False,
):
    T, D = x.shape
    assert dy.shape == (T, D)
    if dres is not None:
        assert dres.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dres_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    NS = get_multiprocessor_count(x.device.index)
    BS = triton.cdiv(T, NS)

    dw = torch.empty((NS, D), dtype=torch.float32, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float32, device=bias.device) if bias is not None else None
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
        dres=dres,
        dres_in=dres_in,
        mean=mean,
        rstd=rstd,
        T=T,
        D=D,
        BS=BS,
        BD=BD,
        IS_RMS_NORM=is_rms_norm,
        HAS_DRESIDUAL=dres is not None,
        STORE_DRESIDUAL=dres_in is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    # the residual shares the input gradient when their dtypes match
    if has_residual and dx.dtype == x.dtype:
        dres_in = dx
    return (dx, dw, db, dres_in) if not recompute_output else (dx, dw, db, dres_in, y)


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
        y, mean, rstd, res_out = layer_norm_quant_fwd(
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
        ctx.save_for_backward(res_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, res_out.reshape(x_shape_og))

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
            dres = args[0]
            dres = dres.reshape(-1, dres.shape[-1])
            assert dres.shape == x.shape
        else:
            dres = None
        dx, dnorm_weight, dnorm_bias, dres_in, y = layer_norm_quant_bwd(
            dy=dy,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            mean=mean,
            rstd=rstd,
            dres=dres,
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
            dres_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )
