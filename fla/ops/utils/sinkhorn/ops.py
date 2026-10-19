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
def sinkhorn_fwd_kernel(
    x,
    o,
    T,
    N: tl.constexpr,
    BN: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_i = tl.arange(0, BN).to(tl.int64)
    o_j = tl.arange(0, BN).to(tl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_x = tl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(tl.float32)
    for _ in range(NUM_ITERS):
        b_max = tl.max(b_x, axis=0)
        b_x = b_x - b_max[None, :]
        b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=0))[None, :], float('-inf'))
        b_max = tl.max(b_x, axis=1)
        b_x = b_x - b_max[:, None]
        b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=1))[:, None], float('-inf'))
    tl.store(o + o_ij, tl.exp(b_x), mask=m_ij)


@triton.jit(do_not_specialize=['T'])
def sinkhorn_bwd_kernel(x, do, dx, T, N: tl.constexpr, BN: tl.constexpr, NUM_ITERS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_i = tl.arange(0, BN).to(tl.int64)
    o_j = tl.arange(0, BN).to(tl.int64)
    o_ij = (i_t * N + o_i[:, None]) * N + o_j[None, :]
    m_ij = (o_i[:, None] < N) & (o_j[None, :] < N)
    b_input = tl.load(x + o_ij, mask=m_ij, other=float('-inf')).to(tl.float32)
    b_dx = tl.load(do + o_ij, mask=m_ij, other=0).to(tl.float32)
    # Recompute earlier blocks to bound the live probability cache.
    BI: tl.constexpr = NUM_ITERS if NUM_ITERS < 20 else 20
    for i_b in range(triton.cdiv(NUM_ITERS, BI) - 1, -1, -1):
        b_x = b_input
        for _ in range(i_b * BI):
            b_x = b_x - tl.max(b_x, axis=0)[None, :]
            b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=0))[None, :], float('-inf'))
            b_x = b_x - tl.max(b_x, axis=1)[:, None]
            b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=1))[:, None], float('-inf'))
        b_ps = ()
        for step in tl.static_range(BI):
            b_col = tl.full((BN, BN), 0, tl.float32)
            b_row = tl.full((BN, BN), 0, tl.float32)
            if i_b * BI + step < NUM_ITERS:
                b_x = b_x - tl.max(b_x, axis=0)[None, :]
                b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=0))[None, :], float('-inf'))
                b_col = tl.exp(b_x)
                b_x = b_x - tl.max(b_x, axis=1)[:, None]
                b_x = tl.where(m_ij, b_x - tl.log(tl.sum(tl.exp(b_x), axis=1))[:, None], float('-inf'))
                b_row = tl.exp(b_x)
            b_ps += (b_col, b_row)
        if i_b == triton.cdiv(NUM_ITERS, BI) - 1:
            b_dx = b_dx * tl.exp(b_x)
        for step in tl.static_range(BI - 1, -1, -1):
            b_dx = b_dx - b_ps[2 * step + 1] * tl.sum(b_dx, axis=1)[:, None]
            b_dx = b_dx - b_ps[2 * step] * tl.sum(b_dx, axis=0)[None, :]
    tl.store(dx + o_ij, b_dx, mask=m_ij)


@dispatch
def sinkhorn_fwd(logits: torch.Tensor, num_iters: int):
    N = logits.shape[-1]
    T = logits.numel() // (N * N)
    out = torch.empty_like(logits, dtype=torch.float32)
    if T:
        sinkhorn_fwd_kernel[(T,)](
            x=logits,
            o=out,
            T=T,
            N=N,
            BN=triton.next_power_of_2(N),
            NUM_ITERS=num_iters,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, sinkhorn_bwd


def sinkhorn_bwd(logits: torch.Tensor, dout: torch.Tensor, num_iters: int) -> torch.Tensor:
    N = logits.shape[-1]
    T = logits.numel() // (N * N)
    dx = torch.empty_like(logits)
    if T:
        sinkhorn_bwd_kernel[(T,)](
            x=logits,
            do=dout,
            dx=dx,
            T=T,
            N=N,
            BN=triton.next_power_of_2(N),
            NUM_ITERS=num_iters,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dx


class SinkhornFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, logits, num_iters):
        out, ctx.bwd = sinkhorn_fwd(logits=logits, num_iters=num_iters)
        ctx.save_for_backward(logits)
        ctx.num_iters = num_iters
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, dout):
        logits, = ctx.saved_tensors
        dx = ctx.bwd(logits=logits, dout=dout, num_iters=ctx.num_iters)
        return dx, None


def sinkhorn(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Normalize square matrices with alternating column and row log-sum-exp steps.

    Args:
        logits (torch.Tensor):
            Input logits of shape `[..., N, N]`, with `1 <= N <= 8`.
        num_iters (int, Optional):
            Positive number of normalization iterations. Default: 20.

    Returns:
        torch.Tensor:
            Nonnegative FP32 matrices of shape `[..., N, N]`.
            A finite iteration count does not guarantee exact double stochasticity.
    """
    if not IS_NVIDIA or not logits.is_cuda:
        raise ValueError('Sinkhorn kernels require NVIDIA CUDA tensors')
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('logits must use FP16, BF16, or FP32')
    if logits.ndim < 2 or not 1 <= logits.shape[-1] <= 8 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError('logits must have shape [..., N, N] with 1 <= N <= 8')
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    return SinkhornFunction.apply(logits, num_iters)
