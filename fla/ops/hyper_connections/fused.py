# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from importlib import import_module

import torch
import triton
from torch.autograd.function import once_differentiable

from fla.backends import dispatch
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


def _validate(x, *others):
    if not IS_NVIDIA or not x.is_cuda:
        raise ValueError('Hyper-connection kernels require NVIDIA CUDA tensors')
    for value in (x, *others):
        if value is not None and (value.device != x.device or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError('Inputs must share a CUDA device and use FP16, BF16, or FP32')
    if x.ndim < 2 or not 1 <= x.shape[-2] <= 8 or x.shape[-1] < 1:
        raise ValueError('Expected [..., N, D] with 1 <= N <= 8 and D >= 1')


def _launch(backend, name, grid, **kwargs):
    if backend == 'tilelang':
        from fla.ops.hyper_connections.backends import tilelang
        tilelang.launch(name=name, **kwargs)
    else:
        path = 'triton' if backend == 'triton' else 'backends.gluon'
        kernels = import_module(f'fla.ops.hyper_connections.{path}')
        getattr(kernels, name)[grid](**kwargs, num_warps=4, enable_fp_fusion=False)


class HyperConnectionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, weight, branch, mixing, backend):
        N, D = x.shape[-2:]
        T = x.numel() // (N * D)
        BD = min(256, triton.next_power_of_2(D))
        CD = triton.cdiv(D, BD)
        out = torch.empty_like(x) if branch is not None else x.new_empty((*x.shape[:-2], D))
        if T:
            _launch(
                backend,
                'hyper_connection_fwd_kernel',
                (T, CD),
                X=x,
                W=weight,
                B=branch,
                M=mixing,
                O=out,
                T=T,
                N=N,
                D=D,
                BN=triton.next_power_of_2(N),
                BD=BD,
                POST=branch is not None,
                MIX=mixing is not None,
            )
        ctx.save_for_backward(x, weight, branch, mixing)
        ctx.backend, ctx.BD = backend, BD
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, dout):
        x, weight, branch, mixing = ctx.saved_tensors
        N, D = x.shape[-2:]
        T, BD = x.numel() // (N * D), ctx.BD
        CD = triton.cdiv(D, BD)
        dx, dw = torch.empty_like(x), torch.empty_like(weight)
        db = torch.empty_like(branch) if branch is not None else None
        dm = torch.empty_like(mixing) if mixing is not None else None
        pw = torch.empty((T, CD, N), device=x.device, dtype=torch.float32)
        pm = torch.empty((T, CD, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
        if T:
            _launch(
                ctx.backend,
                'hyper_connection_bwd_kernel',
                (T, CD),
                X=x,
                W=weight,
                B=branch,
                M=mixing,
                DO=dout,
                DX=dx,
                DB=db,
                PW=pw,
                PM=pm,
                T=T,
                N=N,
                D=D,
                BN=triton.next_power_of_2(N),
                BD=BD,
                CD=CD,
                POST=branch is not None,
                MIX=mixing is not None,
            )
            for partial, grad, size in ((pw, dw, N), (pm, dm, N * N)):
                if grad is not None:
                    _launch(
                        ctx.backend,
                        'hyper_connection_reduce_kernel',
                        (T,),
                        P=partial,
                        G=grad,
                        T=T,
                        C=CD,
                        S=size,
                        BC=triton.next_power_of_2(CD),
                        BS=triton.next_power_of_2(size),
                    )
        return dx, dw, db, dm, None


class SinkhornFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, logits, num_iters, backend, save_history):
        N = logits.shape[-1]
        T = logits.numel() // (N * N)
        out = torch.empty_like(logits, dtype=torch.float32)
        history = torch.empty((2 * num_iters, T, N, N), device=logits.device, dtype=torch.float32) if save_history else None
        if T:
            _launch(
                backend,
                'sinkhorn_fwd_kernel',
                (T,),
                X=logits,
                O=out,
                H=history,
                T=T,
                N=N,
                BN=triton.next_power_of_2(N),
                ITERS=num_iters,
                SAVE=save_history,
            )
        ctx.save_for_backward(out, history)
        ctx.backend, ctx.num_iters, ctx.dtype = backend, num_iters, logits.dtype
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, dout):
        out, history = ctx.saved_tensors
        N, T = out.shape[-1], out.numel() // out.shape[-1] ** 2
        dx = torch.empty_like(out, dtype=ctx.dtype)
        if T:
            _launch(
                ctx.backend,
                'sinkhorn_bwd_kernel',
                (T,),
                O=out,
                H=history,
                DO=dout,
                DX=dx,
                T=T,
                N=N,
                BN=triton.next_power_of_2(N),
                ITERS=ctx.num_iters,
            )
        return dx, None, None, None


def _pre(x, pre, backend):
    _validate(x, pre)
    if pre.shape != x.shape[:-1]:
        raise ValueError('pre must have shape [..., N] matching x')
    return HyperConnectionFunction.apply(x, pre, None, None, backend)


def _post(x, branch, post, mixing=None, *, backend):
    _validate(x, branch, post, mixing)
    if branch.shape != (*x.shape[:-2], x.shape[-1]) or post.shape != x.shape[:-1]:
        raise ValueError('branch and post must have shapes [..., D] and [..., N] matching x')
    if mixing is not None and mixing.shape != (*x.shape[:-1], x.shape[-2]):
        raise ValueError('mixing must have shape [..., N, N] matching x')
    return HyperConnectionFunction.apply(x, post, branch, mixing, backend)


def _sinkhorn(logits, num_iters=20, *, backend):
    _validate(logits)
    if logits.shape[-1] != logits.shape[-2]:
        raise ValueError('logits must have shape [..., N, N]')
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    return SinkhornFunction.apply(logits, num_iters, backend, torch.is_grad_enabled() and logits.requires_grad)


@dispatch
def hyper_connection_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Read [..., N, D] streams with [..., N] weights; return [..., D] in x.dtype."""
    return _pre(x, pre, 'triton')


@dispatch
def hyper_connection_post(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write [..., D] branch values into streams, with optional [..., N, N] residual mixing."""
    return _post(x, branch, post, mixing, backend='triton')


@dispatch
def sinkhorn(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Return FP32 mixing matrices after column/row log normalizations; see the operator README."""
    return _sinkhorn(logits, num_iters, backend='triton')
