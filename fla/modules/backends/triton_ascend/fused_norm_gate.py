# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused LayerNorm/RMSNorm + gate kernels adapted for triton-ascend on Huawei NPU.

Grid-stride BT-tiled kernels keep weight/bias resident and vectorize a small
row tile under UB. Ascend910 @ D=1024 (multi-buffer compile-validated):
fwd BT=8 / bwd BT=4.
"""

import torch
import triton
import triton.language as tl

from fla.utils import get_multiprocessor_count
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    compute_ub_block_size,
)

# Peak live fp32 tiles relative to [BT, BD].
# BD uses a single-row budget so large D is not rejected when BT can still be 1.
# Fwd grid-stride keeps w/b resident (needs higher mult than one-tile launch).
# Bwd stores dg early before the dx path. Calibrated on Ascend910 (192 KiB UB,
# 0.85 margin): fwd BT=8 @ D=1024 (mult=3); bwd BT=4 @ D=1024 (mult=8).
# Do not lower bwd mult below ~6 without re-validating (BT=8 bwd overflows).
_BD_MEM_MULT = 6.0
_FWD_MEM_MULT = 3.0
_BWD_MEM_MULT = 8.0
_UB_SAFETY_MARGIN = 0.85
# Legacy byte cap when UB capacity cannot be detected (65536 // fp32).
_FALLBACK_MAX_BD = 65536 // 4
_MAX_BT = 128
# PoT-padded BD>=2048 needs extra headroom under multi-buffering.
_LARGE_BD = 2048
_LARGE_BD_FWD_MEM_MULT = 4.0
_LARGE_BD_BWD_MEM_MULT = 12.0

# ACTIVATION constexpr: 0 = swish/silu, 1 = sigmoid
_ACTIVATION_SWISH = 0
_ACTIVATION_SIGMOID = 1


def _activation_id(activation: str) -> int:
    if activation in ("swish", "silu"):
        return _ACTIVATION_SWISH
    if activation == "sigmoid":
        return _ACTIVATION_SIGMOID
    raise ValueError(f"Unsupported activation: {activation}")


def _tile_memory_multiplier(base: float, large_bd_mult: float, BD: int) -> float:
    """Bump multiplier when PoT-padded BD is large enough to stress UB."""
    if BD >= _LARGE_BD:
        return max(base, large_bd_mult)
    return base


def _fwd_memory_multiplier(BD: int) -> float:
    """Return fwd tile multiplier; grid-stride + large BD needs more headroom."""
    return _tile_memory_multiplier(_FWD_MEM_MULT, _LARGE_BD_FWD_MEM_MULT, BD)


def _bwd_memory_multiplier(is_rms_norm: bool, BD: int) -> float:
    """Return bwd tile multiplier; larger BD needs a higher mult (smaller BT).

    ``is_rms_norm`` is accepted for callers/host UB scripts; LN and RMS currently
    share the same calibrated budget.
    """
    del is_rms_norm  # reserved if LN/RMS budgets diverge
    return _tile_memory_multiplier(_BWD_MEM_MULT, _LARGE_BD_BWD_MEM_MULT, BD)


def _get_layer_norm_gated_tiles(
    D: int,
    is_forward: bool,
    *,
    is_rms_norm: bool = False,
) -> tuple[int, int]:
    """Return (BD, BT) for row-tiled kernels under UB constraints."""
    # BD: fit feature dim with BT=1 using a single-row budget.
    BD = compute_ub_block_size(
        D,
        _BD_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=triton.next_power_of_2(D),
    )
    if D > BD:
        raise RuntimeError(
            f"LayerNormGated feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )
    if is_forward:
        memory_multiplier = _fwd_memory_multiplier(BD)
    else:
        memory_multiplier = _bwd_memory_multiplier(is_rms_norm, BD)
    # Large synthetic row dim so BT is limited by UB, not by a host-side T guess.
    BT = compute_row_tile_block_size(
        1 << 20,
        BD,
        memory_multiplier,
        tiling_row=True,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=16,
        min_block=1,
        max_block=_MAX_BT,
    )
    return BD, BT


def _launch_config(
    T: int,
    D: int,
    device_index: int,
    *,
    is_forward: bool,
    is_rms_norm: bool = False,
) -> tuple[int, int, int]:
    """Return (BD, BT, NS) for a grid-stride launch over T rows."""
    BD, BT = _get_layer_norm_gated_tiles(D, is_forward=is_forward, is_rms_norm=is_rms_norm)
    NT = triton.cdiv(T, BT)
    NS = max(1, min(get_multiprocessor_count(device_index), NT, ASCEND_MAX_GRID_DIM))
    return BD, BT, NS


@triton.jit(do_not_specialize=['T'])
def layer_norm_gated_fwd_kernel(
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
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Grid-stride forward: each program owns a BT-row tile stream."""
    i_s = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D

    if HAS_WEIGHT:
        b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + cols, mask=col_mask).to(tl.float32)

    NT = tl.cdiv(T, BT)
    for i_t in range(i_s, NT, NS):
        rows = i_t * BT + tl.arange(0, BT)
        row_mask = rows < T
        mask = row_mask[:, None] & col_mask[None, :]
        row_off = rows[:, None] * D + cols[None, :]

        b_x = tl.load(x + row_off, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            b_x += tl.load(residual + row_off, mask=mask, other=0.0).to(tl.float32)
        if STORE_RESIDUAL_OUT:
            tl.store(residual_out + row_off, b_x.to(residual_out.dtype.element_ty), mask=mask)

        if not IS_RMS_NORM:
            b_mean = tl.sum(b_x, axis=1) / D
            tl.store(mean + rows, b_mean, mask=row_mask)
            b_xbar = tl.where(mask, b_x - b_mean[:, None], 0.0)
            b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
        else:
            b_xbar = tl.where(mask, b_x, 0.0)
            b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
        b_rstd = 1 / tl.sqrt(b_var + eps)
        tl.store(rstd + rows, b_rstd, mask=row_mask)

        b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]

        b_g = tl.load(g + row_off, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == 0:
            b_y = b_y * b_g * tl.sigmoid(b_g)
        else:
            b_y = b_y * tl.sigmoid(b_g)

        tl.store(y + row_off, b_y.to(y.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=['T'])
def layer_norm_gated_bwd_kernel(
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
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    """Grid-stride backward: each program owns a BT-row tile stream.

    Gate grads are stored before the LayerNorm/RMSNorm dx path so Ascend can
    release those temporaries and keep BT within the UB budget.
    """
    i_s = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D

    if HAS_WEIGHT:
        b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + cols, mask=col_mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    NT = tl.cdiv(T, BT)
    for i_t in range(i_s, NT, NS):
        rows = i_t * BT + tl.arange(0, BT)
        row_mask = rows < T
        mask = row_mask[:, None] & col_mask[None, :]
        row_off = rows[:, None] * D + cols[None, :]

        b_x = tl.load(x + row_off, mask=mask, other=0.0).to(tl.float32)
        if not IS_RMS_NORM:
            b_mean = tl.load(mean + rows, mask=row_mask, other=0.0)
        b_rstd = tl.load(rstd + rows, mask=row_mask, other=0.0)
        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_xhat = tl.where(mask, b_xhat, 0.0)

        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]
        if RECOMPUTE_OUTPUT:
            tl.store(y + row_off, b_y.to(y.dtype.element_ty), mask=mask)

        b_g = tl.load(g + row_off, mask=mask, other=0.0).to(tl.float32)
        b_dy = tl.load(dy + row_off, mask=mask, other=0.0).to(tl.float32)
        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 0:
            # silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
            b_dsilu = b_sigmoid_g * (1 + b_g * (1 - b_sigmoid_g))
            tl.store(dg + row_off, (b_dy * b_y * b_dsilu).to(dg.dtype.element_ty), mask=mask)
            b_dy = b_dy * b_g * b_sigmoid_g
        else:
            tl.store(
                dg + row_off,
                (b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)).to(dg.dtype.element_ty),
                mask=mask,
            )
            b_dy = b_dy * b_sigmoid_g

        if HAS_WEIGHT:
            b_dw += tl.sum(tl.where(mask, b_dy * b_xhat, 0.0), axis=0)
            b_wdy = b_dy * b_w[None, :]
        else:
            b_wdy = b_dy
        if HAS_BIAS:
            b_db += tl.sum(tl.where(mask, b_dy, 0.0), axis=0)

        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_c2 = tl.sum(b_wdy, axis=1) / D
            b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[:, None]
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]

        if HAS_DRESIDUAL:
            b_dx += tl.load(dresidual + row_off, mask=mask, other=0.0).to(tl.float32)
        if STORE_DRESIDUAL:
            tl.store(dresidual_in + row_off, b_dx.to(dresidual_in.dtype.element_ty), mask=mask)
        tl.store(dx + row_off, b_dx.to(dx.dtype.element_ty), mask=mask)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + cols, b_dw, mask=col_mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + cols, b_db, mask=col_mask)


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

    BD, BT, NS = _launch_config(
        T, D, x.device.index, is_forward=True, is_rms_norm=is_rms_norm,
    )
    act_id = _activation_id(activation)
    layer_norm_gated_fwd_kernel[(NS,)](
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
        T=T,
        NS=NS,
        D=D,
        BD=BD,
        BT=BT,
        ACTIVATION=act_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
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

    BD, BT, NS = _launch_config(
        T, D, x.device.index, is_forward=False, is_rms_norm=is_rms_norm,
    )

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None

    act_id = _activation_id(activation)
    layer_norm_gated_bwd_kernel[(NS,)](
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
        NS=NS,
        D=D,
        BD=BD,
        BT=BT,
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
