# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Tri Dao, Yu Zhang, Songlin Yang.

import torch
import torch.nn.functional as F
from fla.utils import (autocast_custom_bwd, autocast_custom_fwd,
                       contiguous, device)
from fla.modules.activations_triton import (logsigmoid_fwd,
                                            logsigmoid_bwd)

if device == 'cuda':
    from fla.modules.activations_cuda import (sigmoid, swish, swiglu_fwd,
                                              swiglu_bwd, swiglu_fwdbwd)
else:
    sigmoid = F.sigmoid
    swish = F.silu
    from fla.modules.activations_triton import (swiglu_fwd, swiglu_bwd,
                                                swiglu_fwdbwd)


@torch.jit.script
def swiglu_fwd_torch(x, y):
    return (F.silu(x.float()) * y).to(x.dtype)


@torch.jit.script
def swiglu_bwd_torch(x, y, g):
    dtype = x.dtype
    x, y, g = x.float(), y.float(), g.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    return dx.to(dtype), dy.to(dtype)


@torch.jit.script
def swiglu_fwdbwd_torch(x, y, g):
    dtype = x.dtype
    x, y, g = x.float(), y.float(), g.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g
    z = x_swish * y
    return dx.to(dtype), dy.to(dtype), z.to(dtype)


class SwiGLUFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function.

    .. math::
        \text{SwiGLU}(x, y) = swish(x) * y = \frac{x}{1 + \exp(-x)} * y
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
            return swiglu_fwd_torch(x, y)
        else:
            return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
            return swiglu_bwd_torch(x, y, dout)
        else:
            return swiglu_bwd(x, y, dout)


class SwiGLULinearFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function followed by a linear transformation.

    .. math::
        \text{SwiGLULinear}(x, y, W, b) = (swish(x) * y) W + b

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
                z = swiglu_fwd_torch(x, y)
            else:
                z = swiglu_fwd(x, y)
        out = F.linear(z, weight, bias)
        # We don't store z, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
                dx, dy, z = swiglu_fwdbwd_torch(x, y, dz)
            else:
                dx, dy, z = swiglu_fwdbwd(x, y, dz)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x,)
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @contiguous
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None


# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=y.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D) and bias has shape (D)"""
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
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=x.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
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


@torch.jit.script
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)


@torch.jit.script
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x)).to(dtype=x.dtype)


class SquaredReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return sqrelu_bwd(grad_output, input)


swiglu = SwiGLUFunction.apply
logsigmoid = LogSigmoidFunction.apply
swiglu_linear = SwiGLULinearFunction.apply
bias_gelu_impl = GeLUFunction.apply
fast_gelu_impl = FastGeLUFunction.apply
sqrelu = SquaredReLUFunction.apply


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
