# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from fla.backends import dispatch
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit(do_not_specialize=['T'])
def hc_fwd_kernel(
    x,
    weight,
    branch,
    mixing,
    o,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HAS_BRANCH: tl.constexpr,
    HAS_MIXING: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_weight = tl.load(weight + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if HAS_BRANCH:
        b_branch = tl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        if HAS_MIXING:
            b_o = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_mixing = tl.load(mixing + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(tl.float32)
                b_o += b_mixing[:, None] * b_x[None, :]
        else:
            b_o = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_o += b_weight[:, None] * b_branch[None, :]
        tl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = tl.full((BD,), 0, dtype=tl.float32)
        for i_n in tl.static_range(N):
            b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
            b_weight_j = tl.load(weight + i_t * N + i_n).to(tl.float32)
            b_o += b_weight_j * b_x
        tl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@triton.jit(do_not_specialize=['T'])
def hc_bwd_kernel(
    x,
    weight,
    branch,
    mixing,
    do,
    dx,
    dbranch,
    dweight_partial,
    dmixing_partial,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    CD: tl.constexpr,
    HAS_BRANCH: tl.constexpr,
    HAS_MIXING: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    b_weight = tl.load(weight + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if HAS_BRANCH:
        b_do = tl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_branch = tl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        b_dweight = tl.sum(b_do * b_branch[None, :], axis=1)
        b_dbranch = tl.sum(b_do * b_weight[:, None], axis=0)
        tl.store(dbranch + i_t * D + o_d, b_dbranch, mask=o_d < D)
        if HAS_MIXING:
            b_dx = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                b_do_j = tl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_mixing = tl.load(mixing + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(tl.float32)
                b_dx += b_mixing[:, None] * b_do_j[None, :]
                b_x_j = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                b_dmixing = tl.sum(b_do * b_x_j[None, :], axis=1)
                tl.store(dmixing_partial + ((i_t * CD + i_c) * N + o_n) * N + i_n, b_dmixing, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        b_do = tl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        b_x = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_dx = b_weight[:, None] * b_do[None, :]
        b_dweight = tl.sum(b_x * b_do[None, :], axis=1)
    tl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    tl.store(dweight_partial + (i_t * CD + i_c) * N + o_n, b_dweight, mask=o_n < N)


@triton.jit(do_not_specialize=['T'])
def hc_bwd_kernel_reduce(partial, grad, T, C: tl.constexpr, S: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_c = tl.arange(0, BC).to(tl.int64)
    o_s = tl.arange(0, BS).to(tl.int64)
    b_partial = tl.load(
        partial + (i_t * C + o_c[:, None]) * S + o_s[None, :],
        mask=(o_c[:, None] < C) & (o_s[None, :] < S),
        other=0,
    )
    tl.store(grad + i_t * S + o_s, tl.sum(b_partial, axis=0), mask=o_s < S)


@triton.jit(do_not_specialize=['T'])
def sinkhorn_fwd_kernel(
    x,
    o,
    history,
    T,
    N: tl.constexpr,
    BN: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    SAVE_HISTORY: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_i = tl.arange(0, BN).to(tl.int64)
    o_j = tl.arange(0, BN).to(tl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_x = tl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(tl.float32)
    for step in range(NUM_ITERS):
        b_max = tl.max(b_x, axis=0)
        b_x = b_x - b_max[None, :]
        b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=0))[None, :], float('-inf'))
        if SAVE_HISTORY:
            tl.store(history + (step.to(tl.int64) * 2 * T * N * N) + o_ij, tl.exp(b_x), mask=m_ij)
        b_max = tl.max(b_x, axis=1)
        b_x = b_x - b_max[:, None]
        b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=1))[:, None], float('-inf'))
        if SAVE_HISTORY:
            tl.store(history + ((step.to(tl.int64) * 2 + 1) * T * N * N) + o_ij, tl.exp(b_x), mask=m_ij)
    tl.store(o + o_ij, tl.exp(b_x), mask=m_ij)


@triton.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(o, history, do, dx, T, N: tl.constexpr, BN: tl.constexpr, NUM_ITERS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_i = tl.arange(0, BN).to(tl.int64)
    o_j = tl.arange(0, BN).to(tl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_dx = tl.load(do + o_ij, mask=m_ij, other=0).to(tl.float32) * tl.load(o + o_ij, mask=m_ij, other=0)
    for step in range(NUM_ITERS - 1, -1, -1):
        b_probability = tl.load(history + ((step.to(tl.int64) * 2 + 1) * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * tl.sum(b_dx, axis=1)[:, None]
        b_probability = tl.load(history + (step.to(tl.int64) * 2 * T * N * N) + o_ij, mask=m_ij, other=0)
        b_dx = b_dx - b_probability * tl.sum(b_dx, axis=0)[None, :]
    tl.store(dx + o_ij, b_dx, mask=m_ij)


def _validate_inputs(x: torch.Tensor, *others: torch.Tensor | None):
    if not IS_NVIDIA or not x.is_cuda:
        raise ValueError('HC kernels require NVIDIA CUDA tensors')
    for value in (x, *others):
        if value is not None and (
            value.device != x.device or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError('Inputs must share a CUDA device and use FP16, BF16, or FP32')
    if x.ndim < 2 or not 1 <= x.shape[-2] <= 8 or x.shape[-1] < 1:
        raise ValueError('Expected [..., N, D] with 1 <= N <= 8 and D >= 1')


def _launch_kernel(backend: str, kernel: triton.JITFunction, grid: tuple[int, ...], **kwargs):
    if backend == 'tilelang':
        from fla.ops.hc.backends import tilelang
        tilelang._launch_kernel(name=kernel.__name__, **kwargs)
    else:
        if backend == 'gluon':
            from fla.ops.hc.backends import gluon
            kernel = getattr(gluon, kernel.__name__)
        kernel[grid](**kwargs, num_warps=4, enable_fp_fusion=False)


class HCFunction(torch.autograd.Function):

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
            _launch_kernel(
                backend=backend,
                kernel=hc_fwd_kernel,
                grid=(T, CD),
                x=x,
                weight=weight,
                branch=branch,
                mixing=mixing,
                o=out,
                T=T,
                N=N,
                D=D,
                BN=triton.next_power_of_2(N),
                BD=BD,
                HAS_BRANCH=branch is not None,
                HAS_MIXING=mixing is not None,
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
        dx, dweight = torch.empty_like(x), torch.empty_like(weight)
        dbranch = torch.empty_like(branch) if branch is not None else None
        dmixing = torch.empty_like(mixing) if mixing is not None else None
        dweight_partial = torch.empty((T, CD, N), device=x.device, dtype=torch.float32)
        dmixing_partial = torch.empty((T, CD, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
        if T:
            _launch_kernel(
                backend=ctx.backend,
                kernel=hc_bwd_kernel,
                grid=(T, CD),
                x=x,
                weight=weight,
                branch=branch,
                mixing=mixing,
                do=dout,
                dx=dx,
                dbranch=dbranch,
                dweight_partial=dweight_partial,
                dmixing_partial=dmixing_partial,
                T=T,
                N=N,
                D=D,
                BN=triton.next_power_of_2(N),
                BD=BD,
                CD=CD,
                HAS_BRANCH=branch is not None,
                HAS_MIXING=mixing is not None,
            )
            for partial, grad, size in ((dweight_partial, dweight, N), (dmixing_partial, dmixing, N * N)):
                if grad is not None:
                    _launch_kernel(
                        backend=ctx.backend,
                        kernel=hc_bwd_kernel_reduce,
                        grid=(T,),
                        partial=partial,
                        grad=grad,
                        T=T,
                        C=CD,
                        S=size,
                        BC=triton.next_power_of_2(CD),
                        BS=triton.next_power_of_2(size),
                    )
        return dx, dweight, dbranch, dmixing, None


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
            _launch_kernel(
                backend=backend,
                kernel=sinkhorn_fwd_kernel,
                grid=(T,),
                x=logits,
                o=out,
                history=history,
                T=T,
                N=N,
                BN=triton.next_power_of_2(N),
                NUM_ITERS=num_iters,
                SAVE_HISTORY=save_history,
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
            _launch_kernel(
                backend=ctx.backend,
                kernel=sinkhorn_bwd_kernel,
                grid=(T,),
                o=out,
                history=history,
                do=dout,
                dx=dx,
                T=T,
                N=N,
                BN=triton.next_power_of_2(N),
                NUM_ITERS=ctx.num_iters,
            )
        return dx, None, None, None


def _hc_pre(x: torch.Tensor, pre: torch.Tensor, backend: str) -> torch.Tensor:
    _validate_inputs(x, pre)
    if pre.shape != x.shape[:-1]:
        raise ValueError('pre must have shape [..., N] matching x')
    return HCFunction.apply(x, pre, None, None, backend)


def _hc_post(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
    *,
    backend: str,
) -> torch.Tensor:
    _validate_inputs(x, branch, post, mixing)
    if branch.shape != (*x.shape[:-2], x.shape[-1]) or post.shape != x.shape[:-1]:
        raise ValueError('branch and post must have shapes [..., D] and [..., N] matching x')
    if mixing is not None and mixing.shape != (*x.shape[:-1], x.shape[-2]):
        raise ValueError('mixing must have shape [..., N, N] matching x')
    return HCFunction.apply(x, post, branch, mixing, backend)


def _sinkhorn(logits: torch.Tensor, num_iters: int = 20, *, backend: str) -> torch.Tensor:
    _validate_inputs(logits)
    if logits.shape[-1] != logits.shape[-2]:
        raise ValueError('logits must have shape [..., N, N]')
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    return SinkhornFunction.apply(logits, num_iters, backend, torch.is_grad_enabled() and logits.requires_grad)


@dispatch
def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Read out residual streams with activated routing weights.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`.
        pre (torch.Tensor):
            Routing weights of shape `[..., N]` with the same leading dimensions as `x`.

    Returns:
        torch.Tensor:
            Weighted sum of shape `[..., D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    return _hc_pre(x=x, pre=pre, backend='triton')


@dispatch
def hc_post(x: torch.Tensor, branch: torch.Tensor, post: torch.Tensor, mixing: torch.Tensor | None = None) -> torch.Tensor:
    """Write a branch into residual streams using iHC or mHC routing.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`.
        branch (torch.Tensor):
            Branch output of shape `[..., D]` with the same leading dimensions as `x`.
        post (torch.Tensor):
            Activated routing weights of shape `[..., N]`.
        mixing (torch.Tensor, Optional):
            Residual mixing matrices of shape `[..., N, N]` for mHC; `None` uses identity mixing for iHC. Default: `None`.

    Returns:
        torch.Tensor:
            Updated streams of shape `[..., N, D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    return _hc_post(x=x, branch=branch, post=post, mixing=mixing, backend='triton')


@dispatch
def sinkhorn(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Normalize routing logits with alternating column and row log-sum-exp steps.

    Args:
        logits (torch.Tensor):
            Routing logits of shape `[..., N, N]`.
        num_iters (int, Optional):
            Positive number of normalization iterations. Default: 20.

    Returns:
        torch.Tensor:
            Nonnegative FP32 mixing matrices of shape `[..., N, N]`.
            A finite iteration count does not guarantee exact double stochasticity.
    """
    return _sinkhorn(logits=logits, num_iters=num_iters, backend='triton')
