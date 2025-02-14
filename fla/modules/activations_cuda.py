# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Tri Dao, Yu Zhang, Songlin Yang.

import torch

sigmoid_fwd_codestring = """
template <typename T> T sigmoid_fwd(T x) {
    return 1.0f / (1.0f + ::exp(-float(x)));
}
"""
sigmoid_bwd_codestring = """
template <typename T> T sigmoid_bwd(T x, T g) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(g) * x_sigmoid * (1.0f - x_sigmoid);
}
"""

sigmoid_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(sigmoid_fwd_codestring)
sigmoid_bwd_jit_fn = torch.cuda.jiterator._create_jit_fn(sigmoid_bwd_codestring)


@torch.compiler.disable
def sigmoid_fwd(x):
    return sigmoid_fwd_jit_fn(x)


@torch.compiler.disable
def sigmoid_bwd(x, g):
    return sigmoid_bwd_jit_fn(x, g)


class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return sigmoid_bwd(x, dout)


sigmoid = SigmoidFunction.apply


swish_fwd_codestring = """
template <typename T> T swish_fwd(T x) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(x) * x_sigmoid;
}
"""
swish_bwd_codestring = """
template <typename T> T swish_bwd(T x, T g) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    return float(g) * x_sigmoid * (1.0f - float(x) * x_sigmoid + float(x));
}
"""

swish_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swish_fwd_codestring)
swish_bwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swish_bwd_codestring)


@torch.compiler.disable
def swish_fwd(x):
    return swish_fwd_jit_fn(x)


@torch.compiler.disable
def swish_bwd(x, g):
    return swish_bwd_jit_fn(x, g)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return swish_bwd(x, dout)


swish = SwishFunction.apply


swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""

swiglu_fwdbwd_codestring = """
template <typename T> T swiglu_fwdbwd(T x, T y, T g, T& dx, T& dy, T& z) {(T x, T y, T g, T& dx, T& dy, T& z) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    float x_swish = float(x) * x_sigmoid;
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = x_swish * float(g);
    z = x_swish * float(y);
}
"""

swiglu_fwd_jit_fn = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd_jit_fn = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)
swiglu_fwdbwd_jit_fn = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_fwdbwd_codestring, num_outputs=3)


@torch.compiler.disable
def swiglu_fwd(x, y):
    return swiglu_fwd_jit_fn(x, y)


@torch.compiler.disable
def swiglu_bwd(x, y, g):
    return swiglu_bwd_jit_fn(x, y, g)


@torch.compiler.disable
def swiglu_fwdbwd(x, y, g):
    return swiglu_fwdbwd_jit_fn(x, y, g)
