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
from triton.testing import do_bench_cudagraph

from fla.backends import dispatch
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['T', 'N', 'D', 'BD', 'IS_POST', 'HAS_MIXING'],
    do_bench=do_bench_cudagraph,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def hc_fwd_kernel(
    x,
    w,
    branch,
    mixing,
    o,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    IS_POST: tl.constexpr,
    HAS_MIXING: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    # [BN]
    b_w = tl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if IS_POST:
        # [BD]
        b_branch = tl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        if HAS_MIXING:
            b_o = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                # [BD]
                b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                # [BN], one input stream's weights across output streams
                b_mixing = tl.load(mixing + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(tl.float32)
                # [BN, 1] * [1, BD] -> [BN, BD]
                b_o += b_mixing[:, None] * b_x[None, :]
        else:
            # [BN, BD]
            b_o = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_o += b_w[:, None] * b_branch[None, :]
        tl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = tl.full((BD,), 0, dtype=tl.float32)
        for i_n in tl.static_range(N):
            # [BD], one residual stream
            b_x = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
            b_wj = tl.load(w + i_t * N + i_n).to(tl.float32)
            b_o += b_wj * b_x
        tl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['T', 'N', 'D', 'BD', 'IS_POST', 'HAS_MIXING'],
    do_bench=do_bench_cudagraph,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def hc_bwd_kernel(
    x,
    w,
    branch,
    mixing,
    do,
    dx,
    dbranch,
    dw_partial,
    dmixing_partial,
    T,
    N: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    ND: tl.constexpr,
    IS_POST: tl.constexpr,
    HAS_MIXING: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    o_n = tl.arange(0, BN).to(tl.int64)
    o_d = i_c * BD + tl.arange(0, BD).to(tl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    # [BN]
    b_w = tl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(tl.float32)
    if IS_POST:
        # [BN, BD]
        b_do = tl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        # [BD]
        b_branch = tl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        # [BN, BD] -> [BN]
        b_dw = tl.sum(b_do * b_branch[None, :], axis=1)
        # [BN, BD] -> [BD]
        b_dbranch = tl.sum(b_do * b_w[:, None], axis=0)
        tl.store(dbranch + i_t * D + o_d, b_dbranch, mask=o_d < D)
        if HAS_MIXING:
            b_dx = tl.full((BN, BD), 0, dtype=tl.float32)
            for i_n in tl.static_range(N):
                # [BD]
                b_doj = tl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                # [BN], one output stream's weights across input streams
                b_mixing = tl.load(mixing + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(tl.float32)
                b_dx += b_mixing[:, None] * b_doj[None, :]
                # [BD]
                b_xj = tl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(tl.float32)
                # [BN], one column of the mixing gradient for this feature tile
                b_dmixing = tl.sum(b_do * b_xj[None, :], axis=1)
                tl.store(dmixing_partial + ((i_t * ND + i_c) * N + o_n) * N + i_n, b_dmixing, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        # [BD]
        b_do = tl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(tl.float32)
        # [BN, BD]
        b_x = tl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(tl.float32)
        b_dx = b_w[:, None] * b_do[None, :]
        # [BN, BD] -> [BN]
        b_dw = tl.sum(b_x * b_do[None, :], axis=1)
    tl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    tl.store(dw_partial + (i_t * ND + i_c) * N + o_n, b_dw, mask=o_n < N)


@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['T', 'C', 'S'],
    do_bench=do_bench_cudagraph,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def hc_bwd_kernel_reduce(partial, grad, T, C: tl.constexpr, S: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_c = tl.arange(0, BC).to(tl.int64)
    o_s = tl.arange(0, BS).to(tl.int64)
    # [BC, BS], feature tiles by routing-weight entries
    b_partial = tl.load(
        partial + (i_t * C + o_c[:, None]) * S + o_s[None, :],
        mask=(o_c[:, None] < C) & (o_s[None, :] < S),
        other=0,
    )
    # [BC, BS] -> [BS]
    tl.store(grad + i_t * S + o_s, tl.sum(b_partial, axis=0), mask=o_s < S)


def hc_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    is_post: bool,
):
    N, D = x.shape[-2:]
    # flatten all leading batch/sequence dimensions into T tokens
    T = x.numel() // (N * D)
    BD = min(256, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    # [..., N, D] for post; [..., D] for pre
    o = torch.empty_like(x) if is_post else x.new_empty((*x.shape[:-2], D))
    hc_fwd_kernel[(T, ND)](
        x=x,
        w=w,
        branch=branch,
        mixing=mixing,
        o=o,
        T=T,
        N=N,
        D=D,
        BN=triton.next_power_of_2(N),
        BD=BD,
        IS_POST=is_post,
        HAS_MIXING=mixing is not None,
    )
    return o


def hc_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    do: torch.Tensor,
    is_post: bool,
):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    BD = min(256, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    dx, dw = torch.empty_like(x), torch.empty_like(w)
    dbranch = torch.empty_like(branch) if is_post else None
    dmixing = torch.empty_like(mixing) if mixing is not None else None
    # [T, ND, N] and [T, ND, N * N], partial gradients along feature tiles
    dw_partial = torch.empty((T, ND, N), device=x.device, dtype=torch.float32)
    dmixing_partial = torch.empty((T, ND, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
    hc_bwd_kernel[(T, ND)](
        x=x,
        w=w,
        branch=branch,
        mixing=mixing,
        do=do,
        dx=dx,
        dbranch=dbranch,
        dw_partial=dw_partial,
        dmixing_partial=dmixing_partial,
        T=T,
        N=N,
        D=D,
        BN=triton.next_power_of_2(N),
        BD=BD,
        ND=ND,
        IS_POST=is_post,
        HAS_MIXING=mixing is not None,
    )
    for partial, grad, size in ((dw_partial, dw, N), (dmixing_partial, dmixing, N * N)):
        if grad is not None:
            hc_bwd_kernel_reduce[(T,)](
                partial=partial,
                grad=grad,
                T=T,
                C=ND,
                S=size,
                BC=triton.next_power_of_2(ND),
                BS=triton.next_power_of_2(size),
            )
    return dx, dw, dbranch, dmixing


class FusedHCPreFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, pre):
        o = hc_fwd(x=x, w=pre, branch=None, mixing=None, is_post=False)
        ctx.save_for_backward(x, pre)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, pre = ctx.saved_tensors
        dx, dpre, _, _ = hc_bwd(x=x, w=pre, branch=None, mixing=None, do=do, is_post=False)
        return dx, dpre


class FusedHCPostFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, branch, post, mixing):
        o = hc_fwd(x=x, w=post, branch=branch, mixing=mixing, is_post=True)
        ctx.save_for_backward(x, branch, post, mixing)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, branch, post, mixing = ctx.saved_tensors
        dx, dpost, dbranch, dmixing = hc_bwd(x=x, w=post, branch=branch, mixing=mixing, do=do, is_post=True)
        return dx, dbranch, dpost, dmixing


@dispatch
def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return FusedHCPreFunction.apply(x, pre)


@dispatch
def hc_post(x: torch.Tensor, branch: torch.Tensor, post: torch.Tensor, mixing: torch.Tensor | None = None) -> torch.Tensor:
    return FusedHCPostFunction.apply(x, branch, post, mixing)


def fused_hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Combine residual streams into the input of an attention or MLP branch.

    Hyper-connections (HC) maintain N residual streams instead of a single residual path.
    The pre operation reads a weighted combination of these streams:
    `o[..., d] = sum_n pre[..., n] * x[..., n, d]`.
    The caller applies its attention or MLP to this readout, then writes the result back with `fused_hc_post`.

    This readout is shared by HC, manifold-constrained HC (mHC), and iHC.
    The post operation controls how the existing residual streams are mixed.
    Routing projections and weight activations belong to the caller; mHC uses `pre = sigmoid(logits)`.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`, with N streams and D features per stream.
        pre (torch.Tensor):
            Routing weights of shape `[..., N]` with the same leading dimensions as `x`.

    Returns:
        torch.Tensor:
            Weighted sum of shape `[..., D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    if x.ndim < 2 or not 1 <= x.shape[-2] <= 8 or x.shape[-1] < 1:
        raise ValueError('Expected [..., N, D] with 1 <= N <= 8 and D >= 1')
    if pre.shape != x.shape[:-1]:
        raise ValueError('pre must have shape [..., N] matching x')
    return hc_pre(x=x, pre=pre)


def fused_hc_post(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update HC, mHC, or iHC residual streams after an attention or MLP branch.

    Hyper-connections (HC) mix the incoming residual streams and add a weighted copy of the branch output to each stream:
    `o[..., i, d] = sum_j mixing[..., i, j] * x[..., j, d] + post[..., i] * branch[..., d]`.
    The branch typically operates on the readout returned by `fused_hc_pre`.

    HC allows unconstrained residual mixing. Manifold-constrained HC (mHC) uses a doubly stochastic mixing matrix,
    usually produced by Sinkhorn normalization, and `post = 2 * sigmoid(logits)`.
    With `mixing=None`, iHC uses identity mixing: each incoming stream is carried forward before adding the branch update.
    The caller supplies the mixing matrix and activated weights; this operation does not normalize them.

    Args:
        x (torch.Tensor):
            Residual streams of shape `[..., N, D]`, with N streams and D features per stream.
        branch (torch.Tensor):
            Branch output of shape `[..., D]` with the same leading dimensions as `x`.
        post (torch.Tensor):
            Activated routing weights of shape `[..., N]`.
        mixing (torch.Tensor, Optional):
            Residual mixing matrices of shape `[..., N, N]`, indexed by output stream then input stream.
            Use a doubly stochastic matrix for mHC or `None` for iHC. Default: `None`.

    Returns:
        torch.Tensor:
            Updated streams of shape `[..., N, D]`, accumulated in FP32 and returned in `x.dtype`.
    """
    if x.ndim < 2 or not 1 <= x.shape[-2] <= 8 or x.shape[-1] < 1:
        raise ValueError('Expected [..., N, D] with 1 <= N <= 8 and D >= 1')
    if branch.shape != (*x.shape[:-2], x.shape[-1]) or post.shape != x.shape[:-1]:
        raise ValueError('branch and post must have shapes [..., D] and [..., N] matching x')
    if mixing is not None and mixing.shape != (*x.shape[:-1], x.shape[-2]):
        raise ValueError('mixing must have shape [..., N, N] matching x')
    return hc_post(x=x, branch=branch, post=post, mixing=mixing)
