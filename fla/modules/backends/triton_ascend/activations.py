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
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# Ascend launch limits: grid dim and per-core vector width.
_PREFERRED_BLOCK_SIZES = (2048, 1024, 512)
_MAX_GRID_DIM = 65535
_MAX_CORE_DIM = 65535


def _activation_launch_config(T: int) -> tuple[tuple[int], int]:
    """Pick block size under Ascend launch limits."""
    for B in _PREFERRED_BLOCK_SIZES:
        if triton.cdiv(T, B) <= _MAX_GRID_DIM and B * 8 <= _MAX_CORE_DIM:
            return (triton.cdiv(T, B),), B
    B = min(triton.cdiv(T, _MAX_GRID_DIM), _MAX_CORE_DIM // 8)
    return (triton.cdiv(T, B),), max(B, 1)


@triton.jit
def _flat_offset(
    offs,
    D: tl.constexpr,
    stride,
    IS_LINEAR: tl.constexpr,
):
    if IS_LINEAR:
        return offs
    row = offs // D
    col = offs % D
    return row * stride + col


def _get_stride(x: torch.Tensor) -> int:
    if x.ndim < 2:
        return 0
    return x.stride(-2)


def _is_linear_stride(stride: int, D: int) -> bool:
    return stride == D


_LINEAR_HEURISTICS_XY = {
    'X_LINEAR': lambda args: _is_linear_stride(args['stride_x_row'], args['D']),
    'Y_LINEAR': lambda args: _is_linear_stride(args['stride_y_row'], args['D']),
}

_LINEAR_HEURISTICS_XYZ = {
    **_LINEAR_HEURISTICS_XY,
    'Z_LINEAR': lambda args: _is_linear_stride(args['stride_z_row'], args['D']),
}

_LINEAR_HEURISTICS_BWD = {
    'X_LINEAR': lambda args: _is_linear_stride(args['stride_x_row'], args['D']),
    'DY_LINEAR': lambda args: _is_linear_stride(args['stride_dy_row'], args['D']),
    'DX_LINEAR': lambda args: _is_linear_stride(args['stride_dx_row'], args['D']),
}

_LINEAR_HEURISTICS_FWDBWD = {
    **_LINEAR_HEURISTICS_XYZ,
    'G_LINEAR': lambda args: _is_linear_stride(args['stride_g_row'], args['D']),
    'DX_LINEAR': lambda args: _is_linear_stride(args['stride_dx_row'], args['D']),
    'DY_LINEAR': lambda args: _is_linear_stride(args['stride_dy_row'], args['D']),
}


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


@triton.heuristics(_LINEAR_HEURISTICS_XY)
@triton.jit(do_not_specialize=['T'])
def sigmoid_fwd_kernel(
    x, y,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    y_val = tl.sigmoid(x_val)
    tl.store(y + y_off, y_val.to(y.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_BWD)
@triton.jit(do_not_specialize=['T'])
def sigmoid_bwd_kernel(
    x, dy, dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    s = tl.sigmoid(x_val)
    dx_val = g_val * s * (1.0 - s)
    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_XY)
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
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
):
    i = tl.program_id(0)
    offs = i * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)

    b_x = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + y_off, b_y.to(y.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_BWD)
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
    X_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
):
    i = tl.program_id(0)
    offs = i * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)

    b_x = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    b_dy = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    b_s = tl.sigmoid(b_x)
    b_dx = b_dy * ((1. - b_s) / temperature)
    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_XY)
@triton.jit(do_not_specialize=['T'])
def swish_fwd_kernel(
    x, y,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    s = tl.sigmoid(x_val)
    y_val = x_val * s
    tl.store(y + y_off, y_val.to(y.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_BWD)
@triton.jit(do_not_specialize=['T'])
def swish_bwd_kernel(
    x, dy, dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    s = tl.sigmoid(x_val)
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s))
    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_XYZ)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwd_kernel(
    x, y, z,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
    Z_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    z_off = _flat_offset(offs, D, stride_z_row, Z_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    y_val = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
    s = tl.sigmoid(x_val)
    z_val = x_val * s * y_val
    tl.store(z + z_off, z_val.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
    **_LINEAR_HEURISTICS_FWDBWD,
})
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
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
    G_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
    Z_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    g_off = _flat_offset(offs, D, stride_g_row, G_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    y_val = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(g + g_off, mask=mask, other=0.).to(tl.float32)

    s = tl.sigmoid(x_val)
    x_s = x_val * s
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s)) * y_val
    dy_val = g_val * x_s

    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + dy_off, dy_val.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        z_off = _flat_offset(offs, D, stride_z_row, Z_LINEAR)
        z_val = x_s * y_val
        tl.store(z + z_off, z_val.to(z.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_XY)
@triton.jit(do_not_specialize=['T'])
def gelu_fwd_kernel(
    x, y,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    t = 0.79788456 * x_val * (1.0 + 0.044715 * x_val * x_val)
    tanh_out = tl.tanh(t)
    y_val = x_val * 0.5 * (1.0 + tanh_out)
    tl.store(y + y_off, y_val.to(y.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_BWD)
@triton.jit(do_not_specialize=['T'])
def gelu_bwd_kernel(
    x, dy, dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    t = 0.79788456 * x_val * (1.0 + 0.044715 * x_val * x_val)
    tanh_out = tl.tanh(t)
    ff = 0.5 * x_val * (
        (1.0 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x_val * x_val)
    ) + 0.5 * (1.0 + tanh_out)
    dx_val = ff * g_val
    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_XY)
@triton.jit(do_not_specialize=['T'])
def sqrelu_fwd_kernel(
    x, y,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    r = tl.maximum(x_val, 0.0)
    y_val = r * r
    tl.store(y + y_off, y_val.to(y.dtype.element_ty), mask=mask)


@triton.heuristics(_LINEAR_HEURISTICS_BWD)
@triton.jit(do_not_specialize=['T'])
def sqrelu_bwd_kernel(
    x, dy, dx,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    x_val = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + dy_off, mask=mask, other=0.).to(tl.float32)
    dx_val = 2.0 * g_val * tl.maximum(x_val, 0.0)
    tl.store(dx + dx_off, dx_val.to(dx.dtype.element_ty), mask=mask)


@torch.compiler.disable
def gelu_fwd_npu(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x)
    grid, B = _activation_launch_config(T)
    gelu_fwd_kernel[grid](
        x, y, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        BLOCK_SIZE=B,
    )
    return y


@torch.compiler.disable
def gelu_bwd_npu(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    g = _ensure_inner_contiguous(g)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x)
    grid, B = _activation_launch_config(T)
    gelu_bwd_kernel[grid](
        x, g, dx, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(g),
        stride_dx_row=_get_stride(dx),
        BLOCK_SIZE=B,
    )
    return dx


@torch.compiler.disable
def sqrelu_fwd_npu(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x)
    grid, B = _activation_launch_config(T)
    sqrelu_fwd_kernel[grid](
        x, y, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        BLOCK_SIZE=B,
    )
    return y


@torch.compiler.disable
def sqrelu_bwd_npu(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    g = _ensure_inner_contiguous(g)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x)
    grid, B = _activation_launch_config(T)
    sqrelu_bwd_kernel[grid](
        x, g, dx, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(g),
        stride_dx_row=_get_stride(dx),
        BLOCK_SIZE=B,
    )
    return dx


@torch.compiler.disable
def sigmoid_fwd_npu(x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    sigmoid_fwd_kernel[grid](
        x, y, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        B=B,
    )
    return y


@torch.compiler.disable
def sigmoid_bwd_npu(x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    sigmoid_bwd_kernel[grid](
        x, dy, dx, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
        B=B,
    )
    return dx


@torch.compiler.disable
def logsigmoid_fwd_npu(x: torch.Tensor, temperature: float = 1., output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    logsigmoid_fwd_kernel[grid](
        x=x,
        y=y,
        temperature=temperature,
        T=T,
        D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        B=B,
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
    grid, B = _activation_launch_config(T)
    logsigmoid_bwd_kernel[grid](
        x=x,
        dx=dx,
        dy=dy,
        temperature=temperature,
        T=T,
        D=D,
        stride_x_row=_get_stride(x),
        stride_dx_row=_get_stride(dx),
        stride_dy_row=_get_stride(dy),
        B=B,
    )
    return dx


@torch.compiler.disable
def swish_fwd_npu(x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    swish_fwd_kernel[grid](
        x, y, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        B=B,
    )
    return y


@torch.compiler.disable
def swish_bwd_npu(x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    swish_bwd_kernel[grid](
        x, dy, dx, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
        B=B,
    )
    return dx


@torch.compiler.disable
def swiglu_fwd_npu(x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    assert x.shape == y.shape, f"swiglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    T, D = x.numel(), x.shape[-1]
    z = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    swiglu_fwd_kernel[grid](
        x, y, z, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_z_row=_get_stride(z),
        B=B,
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
    grid, B = _activation_launch_config(T)
    swiglu_fwdbwd_kernel[grid](
        x, y, g, dx, dy, z, T=T, D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_g_row=_get_stride(g),
        stride_dx_row=_get_stride(dx),
        stride_dy_row=_get_stride(dy),
        stride_z_row=_get_stride(z) if z is not None else 0,
        B=B,
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


@triton.heuristics(_LINEAR_HEURISTICS_XYZ)
@triton.jit(do_not_specialize=['T'])
def powglu_fwd_kernel(
    x, y, z,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    m,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
    Z_LINEAR: tl.constexpr,
):
    i_n = tl.program_id(0)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    z_off = _flat_offset(offs, D, stride_z_row, Z_LINEAR)
    b_x = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
    b_s = tl.sigmoid(b_x)
    b_pos = b_x > 0
    # feed only positive lanes to log/sqrt; masked lanes give x**p = 1 and are dropped by the where
    b_xp = tl.where(b_pos, b_x, 1.0)
    b_sqrt = tl.sqrt(b_xp)
    b_p = m / (b_sqrt + 1.0)
    b_pow = exp(b_p * log(b_xp))
    b_g = tl.where(b_pos, b_pow * b_s, b_x * b_s)
    b_z = b_g * b_y
    tl.store(z + z_off, b_z.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
    **_LINEAR_HEURISTICS_FWDBWD,
})
@triton.jit(do_not_specialize=['T'])
def powglu_fwdbwd_kernel(
    x, y, g, dx, dy, z,
    stride_x_row,
    stride_y_row,
    stride_g_row,
    stride_dx_row,
    stride_dy_row,
    stride_z_row,
    m,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    X_LINEAR: tl.constexpr,
    Y_LINEAR: tl.constexpr,
    G_LINEAR: tl.constexpr,
    DX_LINEAR: tl.constexpr,
    DY_LINEAR: tl.constexpr,
    Z_LINEAR: tl.constexpr,
):
    i_n = tl.program_id(0)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    x_off = _flat_offset(offs, D, stride_x_row, X_LINEAR)
    y_off = _flat_offset(offs, D, stride_y_row, Y_LINEAR)
    g_off = _flat_offset(offs, D, stride_g_row, G_LINEAR)
    dx_off = _flat_offset(offs, D, stride_dx_row, DX_LINEAR)
    dy_off = _flat_offset(offs, D, stride_dy_row, DY_LINEAR)
    b_x = tl.load(x + x_off, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + y_off, mask=mask, other=0.).to(tl.float32)
    b_g = tl.load(g + g_off, mask=mask, other=0.).to(tl.float32)

    b_s = tl.sigmoid(b_x)
    b_pos = b_x > 0
    b_xp = tl.where(b_pos, b_x, 1.0)
    b_sqrt = tl.sqrt(b_xp)
    b_ln = log(b_xp)
    b_p = m / (b_sqrt + 1.0)
    b_pow = exp(b_p * b_ln)

    b_gate_pos = b_pow * b_s
    # d/dx of the exponent term: p' = -m / (2*sqrt(x)*(sqrt(x)+1)**2)
    b_pprime = -m / (2.0 * b_sqrt * (b_sqrt + 1.0) * (b_sqrt + 1.0))
    b_dgate_pos = b_gate_pos * (b_pprime * b_ln + b_p / b_xp + 1.0 - b_s)
    b_gate_neg = b_x * b_s
    b_dgate_neg = b_s * (1.0 + b_x * (1.0 - b_s))

    b_gate = tl.where(b_pos, b_gate_pos, b_gate_neg)
    b_dgate = tl.where(b_pos, b_dgate_pos, b_dgate_neg)

    b_dx = b_g * b_y * b_dgate
    b_dy = b_g * b_gate

    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + dy_off, b_dy.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        b_z = b_gate * b_y
        z_off = _flat_offset(offs, D, stride_z_row, Z_LINEAR)
        tl.store(z + z_off, b_z.to(z.dtype.element_ty), mask=mask)


@torch.compiler.disable
def powglu_fwd_npu(x: torch.Tensor, y: torch.Tensor, power: float = 3.0, output_contiguous: bool = False) -> torch.Tensor:
    assert x.shape == y.shape, f"powglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    T, D = x.numel(), x.shape[-1]
    z = _alloc_output(x, output_contiguous)
    grid, B = _activation_launch_config(T)
    powglu_fwd_kernel[grid](
        x=x,
        y=y,
        z=z,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_z_row=_get_stride(z),
        m=power,
        T=T,
        D=D,
        B=B,
    )
    return z


@torch.compiler.disable
def powglu_fwdbwd_npu(
    x: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    power: float = 3.0,
    use_weight: bool = False,
    output_contiguous: bool = False,
):
    assert x.shape == y.shape == g.shape, f"powglu_fwdbwd: shape mismatch x={x.shape} y={y.shape} g={g.shape}"
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
    grid, B = _activation_launch_config(T)
    powglu_fwdbwd_kernel[grid](
        x=x,
        y=y,
        g=g,
        dx=dx,
        dy=dy,
        z=z,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_g_row=_get_stride(g),
        stride_dx_row=_get_stride(dx),
        stride_dy_row=_get_stride(dy),
        stride_z_row=_get_stride(z) if z is not None else 0,
        m=power,
        T=T,
        D=D,
        B=B,
    )
    if use_weight:
        return dx, dy, z
    return dx, dy


class PowGLULinearFunctionNPU(torch.autograd.Function):
    r"""
    Power-Gated Linear Unit (PowGLU) function followed by a linear transformation.

    .. math::
        \text{PowGLULinear}(x, y, W, b) = (g(x) * y) W + b

    This simple wrap discards the intermediate results of PowGLU(x, y) to save memory.
    """

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias, power):
        z = powglu_fwd_npu(x, y, power, output_contiguous=True)
        out = F.linear(z, weight, bias)
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        ctx.power = power
        return out

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        dx, dy, z = powglu_fwdbwd_npu(x, y, dz, ctx.power, use_weight=True, output_contiguous=True)
        z_flat = z.reshape(-1, z.shape[-1])
        dlinear_weight = dout.t() @ z_flat
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias, None


def powglu_linear_npu(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    power: float = 3.0,
) -> torch.Tensor:
    return PowGLULinearFunctionNPU.apply(x, y, weight, bias, power)
