# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""LayerNorm / RMSNorm / GroupNorm kernels adapted for triton-ascend on Huawei NPU."""

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, get_multiprocessor_count

# Ascend vector UB is small; avoid make_block_ptr tiled kernels and large warp counts.
NUM_WARPS_AUTOTUNE = [2, 4]


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=['D', 'IS_RMS_NORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_fwd_kernel1(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    G: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_g = i_t % G

    x += i_t * D
    y += i_t * D
    if HAS_RESIDUAL:
        res += i_t * D
    if STORE_RESIDUAL_OUT:
        res_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(res + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(res_out + o_d, b_x, mask=m_d)
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
        b_w = tl.load(w + i_g * D + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    tl.store(y + o_d, b_y, mask=m_d)


@triton.heuristics({
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=['D', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_bwd_kernel1(
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
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    GS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_g, i_sg = i_s // GS, i_s % GS

    o_d = tl.arange(0, BD)
    mask = o_d < D

    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_sg * BS * G + i_g, min((i_sg * BS + BS) * G + i_g, T), G):
        b_x = tl.load(x + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + i_t * D + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
            if HAS_BIAS:
                b_y = b_y + b_b
            tl.store(y + i_t * D + o_d, b_y, mask=mask)
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
            b_dres = tl.load(dres + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
            b_dx += b_dres
        b_dx = tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne')
        if STORE_DRESIDUAL:
            tl.store(dres_in + i_t * D + o_d, b_dx, mask=mask)
        tl.store(dx + i_t * D + o_d, b_dx, mask=mask)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)


def layer_norm_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
    num_groups: int = 1,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D, G = *x.shape, num_groups
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)

    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        res_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        res_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # Ascend: use row-wise kernel1 (no make_block_ptr) for all feature dims.
    layer_norm_fwd_kernel1[(T,)](
        x,
        y,
        weight,
        bias,
        residual,
        res_out,
        mean,
        rstd,
        eps,
        G=G,
        D=D,
        BD=BD,
        IS_RMS_NORM=is_rms_norm,
        HAS_RESIDUAL=residual is not None,
        STORE_RESIDUAL_OUT=res_out is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    return y, mean, rstd, res_out if res_out is not None else x


def layer_norm_bwd_npu(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dres: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
    num_groups: int = 1,
):
    T, D, G = *x.shape, num_groups
    assert dy.shape == (T, D)
    if dres is not None:
        assert dres.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)

    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dres_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    NS = min(triton.cdiv(get_multiprocessor_count(x.device.index), G), T // G) * G
    BS = triton.cdiv(T, NS)
    GS = NS // G

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None
    grid = (NS,)

    layer_norm_bwd_kernel1[grid](
        x,
        weight,
        bias,
        y,
        dy,
        dx,
        dw,
        db,
        dres,
        dres_in,
        mean,
        rstd,
        T=T,
        G=G,
        D=D,
        BS=BS,
        BD=BD,
        GS=GS,
        IS_RMS_NORM=is_rms_norm,
        HAS_DRESIDUAL=dres is not None,
        STORE_DRESIDUAL=dres_in is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    dw = dw.view(G, -1, D).sum(1).to(weight).view_as(weight) if weight is not None else None
    db = db.view(G, -1, D).sum(1).to(bias).view_as(bias) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dres_in = dx
    return (dx, dw, db, dres_in) if not recompute_output else (dx, dw, db, dres_in, y)
