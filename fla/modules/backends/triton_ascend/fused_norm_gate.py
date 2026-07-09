# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused LayerNorm/RMSNorm + gate kernels adapted for triton-ascend on Huawei NPU."""

import math

import torch
import triton
import triton.language as tl

from fla.utils import get_multiprocessor_count
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, compute_ub_block_size, iter_axis_launch_chunks

# Peak live fp32 vectors in row-wise gated kernel1 (norm + gate tensors).
_FWD_MEM_MULT = 7.0
_BWD_MEM_MULT = 10.0
_UB_SAFETY_MARGIN = 0.85
# Legacy byte cap when UB capacity cannot be detected (65536 // fp32).
_FALLBACK_MAX_BD = 65536 // 4

# ACTIVATION constexpr: 0 = swish/silu, 1 = sigmoid
_ACTIVATION_SWISH = 0
_ACTIVATION_SIGMOID = 1


def _activation_id(activation: str) -> int:
    if activation in ("swish", "silu"):
        return _ACTIVATION_SWISH
    if activation == "sigmoid":
        return _ACTIVATION_SIGMOID
    raise ValueError(f"Unsupported activation: {activation}")


def _get_layer_norm_gated_bd(D: int, is_forward: bool) -> int:
    """Return power-of-2 block size for feature dim D under UB constraints."""
    memory_multiplier = _FWD_MEM_MULT if is_forward else _BWD_MEM_MULT
    return compute_ub_block_size(
        D,
        memory_multiplier,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=triton.next_power_of_2(D),
    )


def _layer_norm_gated_bwd_launch_config(T: int, device_index: int) -> tuple[int, int]:
    """Return (NS, BS) capped under Ascend grid limit."""
    NS = min(get_multiprocessor_count(device_index), T)
    NS = min(NS, ASCEND_MAX_GRID_DIM)
    BS = math.ceil(T / NS) if NS > 0 else T
    return NS, BS


@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == 0:
        b_y = b_y * b_g * tl.sigmoid(b_g)
    else:
        b_y = b_y * tl.sigmoid(b_g)

    tl.store(y + o_d, b_y, mask=m_d)


@triton.jit
def layer_norm_gated_bwd_kernel1(
    x,
    g,
    w,
    b,
    y,
    dy,
    dx,
    dg,
    dw,
    db,
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    BS,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    mask = o_d < D
    x += i_s * BS * D
    g += i_s * BS * D
    if HAS_DRESIDUAL:
        dresidual += i_s * BS * D
    if STORE_DRESIDUAL:
        dresidual_in += i_s * BS * D
    dy += i_s * BS * D
    dx += i_s * BS * D
    dg += i_s * BS * D
    if RECOMPUTE_OUTPUT:
        y += i_s * BS * D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_s * BS, min(i_s * BS + BS, T)):
        b_x = tl.load(x + o_d, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(g + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)

        b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b
        if RECOMPUTE_OUTPUT:
            tl.store(y + o_d, b_y, mask=mask)

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 0:
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        else:
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
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
            b_dres = tl.load(dresidual + o_d, mask=mask, other=0).to(tl.float32)
            b_dx += b_dres
        if STORE_DRESIDUAL:
            tl.store(dresidual_in + o_d, b_dx, mask=mask)
        tl.store(dx + o_d, b_dx, mask=mask)
        tl.store(dg + o_d, b_dg, mask=mask)

        x += D
        g += D
        if HAS_DRESIDUAL:
            dresidual += D
        if STORE_DRESIDUAL:
            dresidual_in += D
        if RECOMPUTE_OUTPUT:
            y += D
        dy += D
        dx += D
        dg += D
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)


def _launch_layer_norm_gated_fwd_kernel1(
    x: torch.Tensor,
    g: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    D: int,
    BD: int,
    act_id: int,
    is_rms_norm: bool,
):
    chunk_T = x.shape[0]
    layer_norm_gated_fwd_kernel1[(chunk_T,)](
        x=x,
        g=g,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        eps=eps,
        D=D,
        BD=BD,
        ACTIVATION=act_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )


def layer_norm_gated_fwd_npu(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
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
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)

    BD = _get_layer_norm_gated_bd(D, is_forward=True)
    if D > BD:
        raise RuntimeError(
            f"LayerNormGated feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )

    act_id = _activation_id(activation)
    for row_start, row_len in iter_axis_launch_chunks(T, 1, max_grid=ASCEND_MAX_GRID_DIM):
        row_end = row_start + row_len
        _launch_layer_norm_gated_fwd_kernel1(
            x[row_start:row_end],
            g[row_start:row_end],
            y[row_start:row_end],
            weight,
            bias,
            None if residual is None else residual[row_start:row_end],
            None if residual_out is None else residual_out[row_start:row_end],
            None if mean is None else mean[row_start:row_end],
            rstd[row_start:row_end],
            eps,
            D,
            BD,
            act_id,
            is_rms_norm,
        )
    return y, mean, rstd, residual_out if residual_out is not None else x


def layer_norm_gated_bwd_npu(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
):
    T, D = x.shape
    assert dy.shape == (T, D)
    if dresidual is not None:
        assert dresidual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    BD = _get_layer_norm_gated_bd(D, is_forward=False)
    if D > BD:
        raise RuntimeError(
            f"LayerNormGated feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )

    NS, BS = _layer_norm_gated_bwd_launch_config(T, x.device.index)

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None

    act_id = _activation_id(activation)
    layer_norm_gated_bwd_kernel1[(NS,)](
        x=x,
        g=g,
        w=weight,
        b=bias,
        y=y,
        dy=dy,
        dx=dx,
        dg=dg,
        dw=dw,
        db=db,
        dresidual=dresidual,
        dresidual_in=dresidual_in,
        mean=mean,
        rstd=rstd,
        T=T,
        BS=BS,
        D=D,
        BD=BD,
        ACTIVATION=act_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_DRESIDUAL=dresidual_in is not None,
        HAS_DRESIDUAL=dresidual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        RECOMPUTE_OUTPUT=y is not None,
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, dw, db, dresidual_in) if not recompute_output else (dx, dg, dw, db, dresidual_in, y)
