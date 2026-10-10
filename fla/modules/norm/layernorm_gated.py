# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2024, Tri Dao.
#
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# accumulate affine gradients in registers; wide dimensions can cause register spilling.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from fla.utils import get_multiprocessor_count, input_guard


@triton.heuristics({"HAS_BIAS": lambda args: args["b"] is not None, "HAS_GATE": lambda args: args["g"] is not None})
@triton.jit(do_not_specialize=['T'])
def layer_norm_fwd_kernel_group(
    x,
    y,
    w,
    b,
    g,
    mean,
    rstd,
    stride_x_row,
    stride_y_row,
    stride_g_row,
    T,
    D,
    eps,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_g = tl.program_id(1).to(tl.int64)
    x += i_t * stride_x_row + i_g * D
    y += i_t * stride_y_row + i_g * D
    if HAS_GATE:
        g += i_t * stride_g_row + i_g * D
    if not IS_RMS_NORM:
        mean += i_g * T
    rstd += i_g * T
    w += i_g * D
    if HAS_BIAS:
        b += i_g * D

    o_d = tl.arange(0, BD)
    b_x = tl.load(x + o_d, mask=o_d < D, other=0.).to(tl.float32)
    if HAS_GATE and not NORM_BEFORE_GATE:
        b_g = tl.load(g + o_d, mask=o_d < D).to(tl.float32)
        b_x *= b_g * tl.sigmoid(b_g)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(o_d < D, b_x - b_mean, 0.)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(o_d < D, b_x, 0.)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    m_d = o_d < D
    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_xhat * b_w + b_b if HAS_BIAS else b_xhat * b_w
    if HAS_GATE and NORM_BEFORE_GATE:
        b_g = tl.load(g + o_d, mask=m_d).to(tl.float32)
        b_y *= b_g * tl.sigmoid(b_g)

    tl.store(y + o_d, b_y, mask=m_d)


@triton.heuristics({
    "HAS_BIAS": lambda args: args["b"] is not None,
    "HAS_GATE": lambda args: args["g"] is not None,
    "RECOMPUTE_OUTPUT": lambda args: args["y"] is not None,
})
@triton.jit(do_not_specialize=['T'])
def layer_norm_bwd_kernel_group(
    x,
    w,
    b,
    g,
    y,
    dy,
    dx,
    dw,
    db,
    dg,
    mean,
    rstd,
    stride_x_row,
    stride_g_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dg_row,
    stride_dw_row,
    stride_db_row,
    T,
    D,
    eps,
    BS,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BD: tl.constexpr,
):
    i_s = tl.program_id(0).to(tl.int64)
    i_g = tl.program_id(1).to(tl.int64)
    bos = i_s * BS
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    x += bos * stride_x_row + i_g * D
    if HAS_GATE:
        g += bos * stride_g_row + i_g * D
        dg += bos * stride_dg_row + i_g * D
    dy += bos * stride_dy_row + i_g * D
    dx += bos * stride_dx_row + i_g * D
    if RECOMPUTE_OUTPUT:
        y += bos * stride_y_row + i_g * D
    if not IS_RMS_NORM:
        mean += i_g * T
    rstd += i_g * T
    w += i_g * D
    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if (RECOMPUTE_OUTPUT or HAS_GATE) and HAS_BIAS:
        b += i_g * D
        b_b = tl.load(b + o_d, mask=m_d, other=0.).to(tl.float32)
    b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)
    eos = min((i_s + 1) * BS, T)
    for i_t in range(bos, eos):
        b_x = tl.load(x + o_d, mask=m_d, other=0).to(tl.float32)
        b_dy = tl.load(dy + o_d, mask=m_d, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        if HAS_GATE and not NORM_BEFORE_GATE:
            b_g = tl.load(g + o_d, mask=m_d, other=0.).to(tl.float32)
            b_x_og = b_x
            b_x = b_x_og * b_g * tl.sigmoid(b_g)
        b_rstd = tl.load(rstd + i_t)

        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(m_d, b_xhat, 0.)
        if HAS_GATE and NORM_BEFORE_GATE:
            b_g = tl.load(g + o_d, mask=m_d, other=0.).to(tl.float32)
            b_sigmoid_g = tl.sigmoid(b_g)
            b_y = b_xhat * b_w + b_b if HAS_BIAS else b_xhat * b_w
            if RECOMPUTE_OUTPUT:
                tl.store(y + o_d, b_y * b_g * b_sigmoid_g, mask=m_d)
            b_dg = b_dy * b_y * b_sigmoid_g * (1 + b_g * (1 - b_sigmoid_g))
            tl.store(dg + o_d, b_dg, mask=m_d)
            b_dy *= b_g * b_sigmoid_g
        else:
            if RECOMPUTE_OUTPUT:
                b_y = b_xhat * b_w + b_b if HAS_BIAS else b_xhat * b_w
                tl.store(y + o_d, b_y, mask=m_d)
        b_wdy = b_w * b_dy
        b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
        if not IS_RMS_NORM:
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_xhat * b_c1 + b_c2)) * b_rstd
        else:
            b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
        b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy
        if HAS_GATE and not NORM_BEFORE_GATE:
            b_sigmoid_g = tl.sigmoid(b_g)
            b_dg = b_dx * b_x_og * b_sigmoid_g * (1 + b_g * (1 - b_sigmoid_g))
            tl.store(dg + o_d, b_dg, mask=m_d)
            b_dx *= b_g * b_sigmoid_g

        tl.store(dx + o_d, b_dx, mask=m_d)

        x += stride_x_row
        if HAS_GATE:
            g += stride_g_row
            dg += stride_dg_row
        if RECOMPUTE_OUTPUT:
            y += stride_y_row
        dy += stride_dy_row
        dx += stride_dx_row
    tl.store(dw + i_s * stride_dw_row + i_g * D + o_d, b_dw, mask=m_d)
    if HAS_BIAS:
        tl.store(db + i_s * stride_db_row + i_g * D + o_d, b_db, mask=m_d)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    T, D = x.shape
    if group_size is None:
        group_size = D
    assert D % group_size == 0
    G = D // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (T, D)
    assert weight.shape == (D,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (D,)

    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((G * T,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((G * T,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BD // 256, 1), 8)
    grid = (T, G)
    layer_norm_fwd_kernel_group[grid](
        x=x,
        y=out,
        w=weight,
        b=bias,
        g=z,
        mean=mean,
        rstd=rstd,
        stride_x_row=x.stride(0),
        stride_y_row=out.stride(0),
        stride_g_row=z.stride(0) if z is not None else 0,
        T=T,
        D=group_size,
        eps=eps,
        BD=BD,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out, mean, rstd


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    z: torch.Tensor | None = None,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    recompute_output: bool = False,
    dz: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    T, D = x.shape
    if group_size is None:
        group_size = D
    assert D % group_size == 0
    G = D // group_size
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (T, D)
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (T, D)
    assert weight.shape == (D,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (D,)

    dx = torch.empty_like(x)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
        assert dz.stride(-1) == 1
    else:
        dz = torch.empty_like(z) if z is not None else None
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        assert out.shape == x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BD // 256, 1), 8)
    sm_count = get_multiprocessor_count(x.device.index)
    # use more programs for small groups to keep enough warps resident.
    NS = math.ceil(sm_count * math.ceil(4 / num_warps) / G)
    _dw = torch.empty((NS, D), dtype=torch.float32, device=weight.device)
    _db = torch.empty((NS, D), dtype=torch.float32, device=bias.device) if bias is not None else None
    BS = math.ceil(T / NS)
    grid = (NS, G)
    layer_norm_bwd_kernel_group[grid](
        x=x,
        w=weight,
        b=bias,
        g=z,
        y=out if recompute_output else None,
        dy=dy,
        dx=dx,
        dw=_dw,
        db=_db,
        dg=dz,
        mean=mean,
        rstd=rstd,
        stride_x_row=x.stride(0),
        stride_g_row=z.stride(0) if z is not None else 0,
        stride_y_row=0 if not recompute_output else out.stride(0),
        stride_dy_row=dy.stride(0),
        stride_dx_row=dx.stride(0),
        stride_dg_row=dz.stride(0) if dz is not None else 0,
        stride_dw_row=_dw.stride(0),
        stride_db_row=_db.stride(0) if _db is not None else 0,
        T=T,
        D=group_size,
        eps=eps,
        BS=BS,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        BD=BD,
        num_warps=num_warps,
    )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None = None,
        eps: float = 1e-6,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        is_rms_norm: bool = False,
    ):
        """Apply normalization before or after the optional SiLU gate."""

        x_shape_og = x.shape

        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = layer_norm_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        x, weight, bias, mean, rstd, z = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dw, db, dz = layer_norm_bwd(
            dy=dy,
            x=x,
            weight=weight,
            bias=bias,
            eps=ctx.eps,
            mean=mean,
            rstd=rstd,
            z=z,
            group_size=ctx.group_size,
            norm_before_gate=ctx.norm_before_gate,
            is_rms_norm=ctx.is_rms_norm,
        )
        dx = dx.reshape(ctx.x_shape_og)
        dz = dz.reshape(ctx.x_shape_og) if dz is not None else None
        return dx, dw, db, dz, None, None, None, None


def layernorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> torch.Tensor:
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


def rmsnorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
) -> torch.Tensor:
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, True)


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


class LayerNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Apply normalization before or after the optional SiLU gate."""
        return layernorm_fn(
            x=x,
            weight=self.weight,
            bias=self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Apply normalization before or after the optional SiLU gate."""
        return rmsnorm_fn(
            x=x,
            weight=self.weight,
            bias=self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


__all__ = [
    'LayerNormGated',
    'RMSNormGated',
    'layer_norm_bwd',
    'layer_norm_fwd',
    'layernorm_fn',
    'rms_norm_ref',
    'rmsnorm_fn',
]
