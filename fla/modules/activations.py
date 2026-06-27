# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused activation kernels.

The kernels address their inputs through the row stride instead of assuming a fully contiguous buffer.
An inner-contiguous input — such as one half of ``x.chunk(2, dim=-1)`` — is therefore read in place, sparing the extra
``.contiguous()`` copy (and its memory traffic) that a plain flat element-wise kernel would force on every call.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.modules.backends import dispatch
from fla.ops.utils.op import exp, log
from fla.utils import IS_AMD, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


def _get_stride(x: torch.Tensor) -> int:
    """Get the row stride for viewing a tensor as 2D (num_rows, D) where D = shape[-1].

    Returns stride(-2) if the tensor is at least 2D, or 0 for 1D tensors.
    The caller must ensure the tensor is "inner-contiguous" (stride(-1) == 1 and
    higher dims are contiguous relative to dim -2) before using this value.
    """
    if x.ndim < 2:
        return 0
    return x.stride(-2)


def _is_inner_contiguous(x: torch.Tensor) -> bool:
    """Check if a tensor can be safely viewed as 2D (num_rows, D) with row stride = stride(-2).

    This holds when stride(-1) == 1 and all dimensions above -2 are contiguous
    with respect to the dimension below them.
    """
    ndim = x.ndim
    if ndim < 2:
        return True
    if x.stride(-1) != 1:
        return False
    if ndim == 2:
        # 2D: any layout with stride(-1)==1 is valid (can view as (T, D))
        return True
    if ndim == 3:
        # 3D (B, T, D): stride should be (T*D, D, 1)
        return x.stride(0) == x.stride(-2) * x.shape[-2]
    if ndim == 4:
        # 4D (B, H, T, D): stride should be (H*T*D, T*D, D, 1)
        if x.stride(1) != x.stride(-2) * x.shape[-2]:
            return False
        return x.stride(0) == x.stride(1) * x.shape[1]
    # 5D+ fallback to loop
    expected = x.stride(-2) * x.shape[-2]
    for d in range(ndim - 3, -1, -1):
        if x.stride(d) != expected:
            return False
        expected *= x.shape[d]
    return True


def _ensure_inner_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Make the tensor inner-contiguous if it isn't already."""
    if _is_inner_contiguous(x):
        return x
    return x.contiguous()


def _alloc_output(x: torch.Tensor, contiguous: bool = False) -> torch.Tensor:
    """Allocate the output: a fresh contiguous buffer, or ``empty_like`` otherwise.

    ``empty_like`` keeps the input's memory format only when it is dense; a non-dense
    strided view (e.g. a ``chunk`` slice) falls back to contiguous, not the input stride.
    """
    if contiguous:
        return x.new_empty(x.shape)
    return torch.empty_like(x)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def sigmoid_fwd_kernel(
    x, y,
    stride_x_row,
    stride_y_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    tl.store(y + row * stride_y_row + col, b_y.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def sigmoid_bwd_kernel(
    x, dy, dx,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_dy = tl.load(dy + row * stride_dy_row + col, mask=mask, other=0.).to(tl.float32)
    b_s = tl.sigmoid(b_x)
    b_dx = b_dy * b_s * (1.0 - b_s)
    tl.store(dx + row * stride_dx_row + col, b_dx.to(dx.dtype.element_ty), mask=mask)


@dispatch('modules')
def sigmoid_fwd(x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    sigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        T=T,
        D=D,
    )
    return y


@dispatch('modules')
def sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    sigmoid_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        dy=dy,
        dx=dx,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
        T=T,
        D=D,
    )
    return dx


class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_fwd(x)

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return sigmoid_bwd(x, dout)


sigmoid = SigmoidFunction.apply


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_fwd_kernel(
    x,
    y,
    stride_x_row,
    stride_y_row,
    temperature,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + row * stride_y_row + col, b_y.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_bwd_kernel(
    x,
    dy,
    dx,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    temperature,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_dy = tl.load(dy + row * stride_dy_row + col, mask=mask, other=0.).to(tl.float32)
    b_dx = b_dy * ((1. - tl.sigmoid(b_x)) / temperature)
    tl.store(dx + row * stride_dx_row + col, b_dx.to(dx.dtype.element_ty), mask=mask)


@dispatch('modules')
def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1., output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        temperature=temperature,
        T=T,
        D=D,
    )
    return y


@dispatch('modules')
def logsigmoid_bwd(
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
        dy=dy,
        dx=dx,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
        temperature=temperature,
        T=T,
        D=D,
    )
    return dx


class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x)
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None


def logsigmoid(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    return LogSigmoidFunction.apply(x, temperature)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swish_fwd_kernel(
    x, y,
    stride_x_row,
    stride_y_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = b_x * tl.sigmoid(b_x)
    tl.store(y + row * stride_y_row + col, b_y.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swish_bwd_kernel(
    x, dy, dx,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_dy = tl.load(dy + row * stride_dy_row + col, mask=mask, other=0.).to(tl.float32)
    b_s = tl.sigmoid(b_x)
    b_dx = b_dy * b_s * (1.0 + b_x * (1.0 - b_s))
    tl.store(dx + row * stride_dx_row + col, b_dx.to(dx.dtype.element_ty), mask=mask)


@dispatch('modules')
def swish_fwd(x: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    swish_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        T=T,
        D=D,
    )
    return y


@dispatch('modules')
def swish_bwd(x: torch.Tensor, dy: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    dy = _ensure_inner_contiguous(dy)
    T, D = x.numel(), x.shape[-1]
    dx = _alloc_output(x, output_contiguous)
    swish_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        dy=dy,
        dx=dx,
        stride_x_row=_get_stride(x),
        stride_dy_row=_get_stride(dy),
        stride_dx_row=_get_stride(dx),
        T=T,
        D=D,
    )
    return dx


class SwishFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return swish_bwd(x, dout)


swish = SwishFunction.apply

# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.compile
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=y.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D=D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    grad_y = ff * g
    return grad_y.to(dtype=y.dtype), grad_y.sum(dim=(0), dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return bias_gelu_bwd(grad_output, input, bias)


bias_gelu_impl = GeLUFunction.apply


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@dispatch('modules')
@torch.compile
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=x.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@dispatch('modules')
@torch.compile
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return (ff * g).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


@torch.compile
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@dispatch('modules')
@torch.compile
def sqrelu_fwd(x):
    r = F.relu(x.float())
    return (r * r).to(dtype=x.dtype)


@dispatch('modules')
@torch.compile
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x.float())).to(dtype=x.dtype)


class SquaredReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return sqrelu_bwd(grad_output, input)


sqrelu = SquaredReLUFunction.apply


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwd_kernel(
    x, y, z,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_z = b_x * tl.sigmoid(b_x) * b_y
    tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwdbwd_kernel(
    x, y, g, dx, dy, z,
    stride_x_row,
    stride_y_row,
    stride_g_row,
    stride_dx_row,
    stride_dy_row,
    stride_z_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_g = tl.load(g + row * stride_g_row + col, mask=mask, other=0.).to(tl.float32)

    b_s = tl.sigmoid(b_x)
    b_xs = b_x * b_s
    b_dx = b_g * b_s * (1.0 + b_x * (1.0 - b_s)) * b_y
    b_dy = b_g * b_xs

    tl.store(dx + row * stride_dx_row + col, b_dx.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + row * stride_dy_row + col, b_dy.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        b_z = b_xs * b_y
        tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


@dispatch('modules')
def swiglu_fwd(x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    assert x.shape == y.shape, f"swiglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    T, D = x.numel(), x.shape[-1]
    z = _alloc_output(x, output_contiguous)
    swiglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        z=z,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_z_row=_get_stride(z),
        T=T,
        D=D,
    )
    return z


@dispatch('modules')
def swiglu_fwdbwd(
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
        T=T,
        D=D,
    )
    if use_weight:
        return dx, dy, z
    return dx, dy


class SwiGLUFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function.

    .. math::
        \text{SwiGLU}(x, y) = swish(x) * y = \frac{x}{1 + \exp(-x)} * y
    """

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_fwdbwd(x, y, dout)


class SwiGLULinearFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function followed by a linear transformation.

    .. math::
        \text{SwiGLULinear}(x, y, W, b) = (swish(x) * y) W + b

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd(x, y, output_contiguous=True)
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
        dx, dy, z = swiglu_fwdbwd(x, y, dz, use_weight=True, output_contiguous=True)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


swiglu = SwiGLUFunction.apply


@dispatch('modules')
def swiglu_linear(x, y, weight, bias):
    return SwiGLULinearFunction.apply(x, y, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def sigmoidglu_fwd_kernel(
    x, y, z,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_z = tl.sigmoid(b_x) * b_y
    tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def sigmoidglu_fwdbwd_kernel(
    x, y, g, dx, dy, z,
    stride_x_row,
    stride_y_row,
    stride_g_row,
    stride_dx_row,
    stride_dy_row,
    stride_z_row,
    T,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_g = tl.load(g + row * stride_g_row + col, mask=mask, other=0.).to(tl.float32)

    b_s = tl.sigmoid(b_x)
    b_dx = b_g * b_s * (1.0 - b_s) * b_y
    b_dy = b_g * b_s

    tl.store(dx + row * stride_dx_row + col, b_dx.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + row * stride_dy_row + col, b_dy.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        b_z = b_s * b_y
        tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


@torch.compiler.disable
def sigmoidglu_fwd(x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = False) -> torch.Tensor:
    assert x.shape == y.shape, f"sigmoidglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    T, D = x.numel(), x.shape[-1]
    z = _alloc_output(x, output_contiguous)
    sigmoidglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        z=z,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_z_row=_get_stride(z),
        T=T,
        D=D,
    )
    return z


@torch.compiler.disable
def sigmoidglu_fwdbwd(
    x: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    use_weight: bool = False,
    output_contiguous: bool = False,
):
    assert x.shape == y.shape == g.shape, f"sigmoidglu_fwdbwd: shape mismatch x={x.shape} y={y.shape} g={g.shape}"
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
    sigmoidglu_fwdbwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
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
        T=T,
        D=D,
    )
    if use_weight:
        return dx, dy, z
    return dx, dy


class SigmoidGLUFunction(torch.autograd.Function):
    r"""
    Sigmoid-Gated Linear Unit (SigmoidGLU) function.

    .. math::
        \text{SigmoidGLU}(x, y) = sigmoid(x) * y = \frac{1}{1 + \exp(-x)} * y
    """

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return sigmoidglu_fwd(x, y)

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return sigmoidglu_fwdbwd(x, y, dout)


class SigmoidGLULinearFunction(torch.autograd.Function):
    r"""
    Sigmoid-Gated Linear Unit (SigmoidGLU) function followed by a linear transformation.

    .. math::
        \text{SigmoidGLULinear}(x, y, W, b) = (sigmoid(x) * y) W + b

    This simple wrap discards the intermediate results of SigmoidGLU(x, y) to save memory.
    """

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        z = sigmoidglu_fwd(x, y, output_contiguous=True)
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
        dx, dy, z = sigmoidglu_fwdbwd(x, y, dz, use_weight=True, output_contiguous=True)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


sigmoidglu = SigmoidGLUFunction.apply


sigmoidglu_linear = SigmoidGLULinearFunction.apply


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
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
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_s = tl.sigmoid(b_x)
    b_pos = b_x > 0
    # feed only positive lanes to log/sqrt; masked lanes give x**p = 1 and are dropped by the where
    b_xp = tl.where(b_pos, b_x, 1.0)
    b_sqrt = tl.sqrt(b_xp)
    b_p = m / (b_sqrt + 1.0)
    b_pow = exp(b_p * log(b_xp))
    b_g = tl.where(b_pos, b_pow * b_s, b_x * b_s)
    b_z = b_g * b_y
    tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
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
):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_g = tl.load(g + row * stride_g_row + col, mask=mask, other=0.).to(tl.float32)

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

    tl.store(dx + row * stride_dx_row + col, b_dx.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + row * stride_dy_row + col, b_dy.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        b_z = b_gate * b_y
        tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


@dispatch('modules')
def powglu_fwd(x: torch.Tensor, y: torch.Tensor, power: float = 3.0, output_contiguous: bool = False) -> torch.Tensor:
    assert x.shape == y.shape, f"powglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x = _ensure_inner_contiguous(x)
    y = _ensure_inner_contiguous(y)
    T, D = x.numel(), x.shape[-1]
    z = _alloc_output(x, output_contiguous)
    powglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        z=z,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
        stride_z_row=_get_stride(z),
        m=power,
        T=T,
        D=D,
    )
    return z


@dispatch('modules')
def powglu_fwdbwd(
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
    powglu_fwdbwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
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
    )
    if use_weight:
        return dx, dy, z
    return dx, dy


class PowGLUFunction(torch.autograd.Function):
    r"""
    Power-Gated Linear Unit (PowGLU) function.

    .. math::
        \text{PowGLU}(x, y) = g(x) * y,\quad
        g(x) = \begin{cases} x^{power/(\sqrt{x}+1)}\,\sigma(x) & x > 0 \\ x\,\sigma(x) & x \le 0 \end{cases}

    For ``x <= 0`` the gate reduces to swish, matching SwiGLU; for large ``x > 0`` it saturates instead of
    growing, replacing SwiGLU's quadratic amplification with bounded growth (Power Linear Unit, arXiv:2605.25704).
    """

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def forward(ctx, x, y, power):
        ctx.save_for_backward(x, y)
        ctx.power = power
        return powglu_fwd(x, y, power)

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        dx, dy = powglu_fwdbwd(x, y, dout, ctx.power)
        return dx, dy, None


class PowGLULinearFunction(torch.autograd.Function):
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
        z = powglu_fwd(x, y, power, output_contiguous=True)
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
        dx, dy, z = powglu_fwdbwd(x, y, dz, ctx.power, use_weight=True, output_contiguous=True)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias, None


def powglu(x: torch.Tensor, y: torch.Tensor, power: float = 3.0) -> torch.Tensor:
    return PowGLUFunction.apply(x, y, power)


@dispatch('modules')
def powglu_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    power: float = 3.0,
) -> torch.Tensor:
    return PowGLULinearFunction.apply(x, y, weight, bias, power)


ACT2FN = {
    'relu': F.relu,
    'sigmoid': sigmoid,
    'logsigmoid': logsigmoid,
    'silu': swish,
    'swish': swish,
    'sqrelu': sqrelu,
    'gelu': fast_gelu_impl,
    'bias_gelu': bias_gelu_impl,
}
