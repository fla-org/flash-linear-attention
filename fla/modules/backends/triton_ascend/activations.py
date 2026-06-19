# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Activation kernels adapted for triton-ascend on Huawei NPU."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

# Ascend vector UB is small; keep warps and tile sizes conservative.
NUM_WARPS_AUTOTUNE = [2, 4]
BLOCK_SIZES_AUTOTUNE = [512, 1024, 2048]


def _get_stride(x: torch.Tensor) -> int:
    if x.ndim < 2:
        return 0
    return x.stride(-2)


def _is_inner_contiguous(x: torch.Tensor) -> bool:
    ndim = x.ndim
    if ndim < 2:
        return True
    if x.stride(-1) != 1:
        return False
    if ndim == 2:
        return True
    if ndim == 3:
        return x.stride(0) == x.stride(-2) * x.shape[-2]
    if ndim == 4:
        if x.stride(1) != x.stride(-2) * x.shape[-2]:
            return False
        return x.stride(0) == x.stride(1) * x.shape[1]
    expected = x.stride(-2) * x.shape[-2]
    for d in range(ndim - 3, -1, -1):
        if x.stride(d) != expected:
            return False
        expected *= x.shape[d]
    return True


def _ensure_inner_contiguous(x: torch.Tensor) -> torch.Tensor:
    if _is_inner_contiguous(x):
        return x
    return x.contiguous()


def _alloc_output(x: torch.Tensor, contiguous: bool = False) -> torch.Tensor:
    if contiguous:
        return x.new_empty(x.shape)
    return torch.empty_like(x)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def sigmoid_fwd_kernel(
    x, y,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    x_off = row * stride_x_row + col
    y_off = row * stride_y_row + col
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    y_val = 1.0 / (1.0 + exp(-x_val))
    tl.store(y + y_off, y_val.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def sigmoid_bwd_kernel(
    x, dy, dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    x_off = row * stride_x_row + col
    dy_off = row * stride_dy_row + col
    dx_off = row * stride_dx_row + col
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    dx_val = g_val * s * (1.0 - s)
    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_fwd_kernel(
    x,
    y,
    temperature,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    row = o_i // D
    col = o_i % D
    x_off = row * stride_x_row + col
    y_off = row * stride_y_row + col

    b_x = tl.load(x + x_off, mask=m_i, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + y_off, b_y.to(y.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_bwd_kernel(
    x,
    dx,
    dy,
    temperature,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dx_row,
    stride_dy_row,
    B: tl.constexpr,
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    row = o_i // D
    col = o_i % D
    x_off = row * stride_x_row + col
    dx_off = row * stride_dx_row + col
    dy_off = row * stride_dy_row + col

    b_x = tl.load(x + x_off, mask=m_i, other=0.).to(tl.float32)
    b_dy = tl.load(dy + dy_off, mask=m_i, other=0.).to(tl.float32)
    b_s = 1.0 / (1.0 + exp(-b_x))
    b_dx = b_dy * ((1. - b_s) / temperature)
    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swish_fwd_kernel(
    x, y,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    x_off = row * stride_x_row + col
    y_off = row * stride_y_row + col
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    y_val = x_val * s
    tl.store(y + y_off, y_val.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swish_bwd_kernel(
    x, dy, dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    x_off = row * stride_x_row + col
    dy_off = row * stride_dy_row + col
    dx_off = row * stride_dx_row + col
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s))
    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwd_kernel(
    x, y, z,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    x_off = row * stride_x_row + col
    y_off = row * stride_y_row + col
    z_off = row * stride_z_row + col
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    y_val = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    z_val = x_val * s * y_val
    tl.store(z + z_off, z_val.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in BLOCK_SIZES_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwdbwd_kernel(
    x, y, g, dx, dy, z,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    stride_g_row,
    stride_dx_row,
    stride_dy_row,
    stride_z_row,
    B: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    x_off = row * stride_x_row + col
    y_off = row * stride_y_row + col
    g_off = row * stride_g_row + col
    dx_off = row * stride_dx_row + col
    dy_off = row * stride_dy_row + col
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    y_val = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(g + g_off, mask=mask, other=0.).to(tl.float32)

    s = 1.0 / (1.0 + exp(-x_val))
    x_s = x_val * s
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s)) * y_val
    dy_val = g_val * x_s

    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + dy_off, dy_val.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        z_off = row * stride_z_row + col
        z_val = x_s * y_val
        tl.store(z + z_off, z_val.to(z.dtype.element_ty), mask=mask)


@torch.compiler.disable
def sigmoid_fwd_npu(x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    sigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x, y, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
    )
    return y


@torch.compiler.disable
def sigmoid_bwd_npu(x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    sigmoid_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x, dy, dx, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
    )
    return dx


@torch.compiler.disable
def logsigmoid_fwd_npu(x: torch.Tensor, temperature: float = 1., output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        temperature=temperature,
        T=T,
        D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
    )
    return y


@torch.compiler.disable
def logsigmoid_bwd_npu(
    x: torch.Tensor,
    dy: torch.Tensor,
    temperature: float = 1.,
    output_contiguous: bool = False,
) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    logsigmoid_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        dx=dx,
        dy=dy,
        temperature=temperature,
        T=T,
        D=D,
        stride_x_row=_get_stride(x),
        stride_dx_row=_get_stride(dx),
        stride_dy_row=_get_stride(dy),
    )
    return dx


@torch.compiler.disable
def swish_fwd_npu(x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    swish_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x, y, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
    )
    return y


@torch.compiler.disable
def swish_bwd_npu(x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    swish_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x, dy, dx, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
    )
    return dx


@torch.compiler.disable
def swiglu_fwd_npu(x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    assert x.shape == y.shape, f"swiglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    T, D = x.numel(), x.shape[-1]
    z = _alloc_output(x, output_contiguous)
    swiglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x, y, z, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_z_row=_get_stride(z),
    )
    return z


@torch.compiler.disable
def swiglu_fwdbwd_npu(
    x: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    use_weight: bool = False,
    output_contiguous: bool = False,
):
    assert x.shape == y.shape == g.shape, f"swiglu_fwdbwd: shape mismatch x={x.shape} y={y.shape} g={g.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    g = _ensure_inner_contiguous(g)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    dy = _alloc_output(y, output_contiguous)
    if use_weight:
        z = _alloc_output(x, output_contiguous)
    else:
        z = None
    swiglu_fwdbwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x, y, g, dx, dy, z, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_g_row=_get_stride(g),
        stride_dx_row=_get_stride(dx),
        stride_dy_row=_get_stride(dy),
        stride_z_row=_get_stride(z) if z is not None else 0,
    )
    if use_weight:
        return dx, dy, z
    return dx, dy


class SwiGLULinearFunctionNPU(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd_npu(x, y, output_contiguous=True)
        out = F.linear(z, weight, bias)
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        dx, dy, z = swiglu_fwdbwd_npu(x, y, dz, use_weight=True, output_contiguous=True)
        z_flat = z.reshape(-1, z.shape[-1])
        dlinear_weight = dout.t() @ z_flat
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


def swiglu_linear_npu(x, y, weight, bias):
    return SwiGLULinearFunctionNPU.apply(x, y, weight, bias)
