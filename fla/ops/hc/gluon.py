# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
from torch.autograd.function import once_differentiable
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.testing import do_bench_cudagraph

from fla.backends import GluonBackend, register
from fla.ops.hc.fused import hc_post, hc_pre
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


def _prune_hc_layouts(configs, named_args, **kwargs):
    N = {**named_args, **kwargs}['N']
    return [config for config in configs if config.kwargs['WN'] <= triton.next_power_of_2(N)]


@fla_cache_autotune(
    configs=[
        triton.Config({'NW': num_warps, 'WN': WN, 'R': R}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
        for WN in [1, num_warps]
        for R in [1, 4]
    ],
    key=['T', 'N', 'D', 'BD', 'IS_POST', 'HAS_MIXING'],
    prune_configs_by={'early_config_prune': _prune_hc_layouts},
    do_bench=do_bench_cudagraph,
    **autotune_cache_kwargs,
)
@gluon.jit(do_not_specialize=['T'])
def hc_fwd_kernel_gluon(
    x,
    w,
    branch,
    mixing,
    o,
    T,
    N: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    IS_POST: gl.constexpr,
    HAS_MIXING: gl.constexpr,
    NW: gl.constexpr,
    WN: gl.constexpr,
    R: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_c = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, R], [1, 32], [WN, NW // WN], [1, 0])
    o_n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_d = i_c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    # [BN]
    b_w = gl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(gl.float32)
    if IS_POST:
        # [BD]
        b_branch = gl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        if HAS_MIXING:
            b_o = gl.full((BN, BD), 0, dtype=gl.float32, layout=layout)
            for i_n in gl.static_range(N):
                # [BD]
                b_x = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                # [BN], one input stream's weights across output streams
                b_mixing = gl.load(mixing + (i_t * N + o_n) * N + i_n, mask=o_n < N, other=0).to(gl.float32)
                # [BN, 1] * [1, BD] -> [BN, BD]
                b_o += b_mixing[:, None] * b_x[None, :]
        else:
            # [BN, BD]
            b_o = gl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_o += b_w[:, None] * b_branch[None, :]
        gl.store(o + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_o, mask=m_nd)
    else:
        # keep the stream reduction in registers to avoid a shared-memory layout conversion
        b_o = gl.full((BD,), 0, dtype=gl.float32, layout=gl.SliceLayout(0, layout))
        for i_n in gl.static_range(N):
            # [BD], one residual stream
            b_x = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
            b_wj = gl.load(w + i_t * N + i_n).to(gl.float32)
            b_o += b_wj * b_x
        gl.store(o + i_t * D + o_d, b_o, mask=o_d < D)


@fla_cache_autotune(
    configs=[
        triton.Config({'NW': num_warps, 'WN': WN, 'R': R}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
        for WN in [1, num_warps]
        for R in [1, 4]
    ],
    key=['T', 'N', 'D', 'BD', 'IS_POST', 'HAS_MIXING'],
    prune_configs_by={'early_config_prune': _prune_hc_layouts},
    do_bench=do_bench_cudagraph,
    **autotune_cache_kwargs,
)
@gluon.jit(do_not_specialize=['T'])
def hc_bwd_kernel_gluon(
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
    N: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    ND: gl.constexpr,
    IS_POST: gl.constexpr,
    HAS_MIXING: gl.constexpr,
    NW: gl.constexpr,
    WN: gl.constexpr,
    R: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_c = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, R], [1, 32], [WN, NW // WN], [1, 0])
    o_n = gl.arange(0, BN, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_d = i_c * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    m_nd = (o_n[:, None] < N) & (o_d[None, :] < D)
    # [BN]
    b_w = gl.load(w + i_t * N + o_n, mask=o_n < N, other=0).to(gl.float32)
    if IS_POST:
        # [BN, BD]
        b_do = gl.load(do + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        # [BD]
        b_branch = gl.load(branch + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        # [BN, BD] -> [BN]
        b_dw = gl.sum(b_do * b_branch[None, :], axis=1)
        # [BN, BD] -> [BD]
        b_dbranch = gl.sum(b_do * b_w[:, None], axis=0)
        gl.store(dbranch + i_t * D + o_d, b_dbranch, mask=o_d < D)
        if HAS_MIXING:
            b_dx = gl.full((BN, BD), 0, dtype=gl.float32, layout=layout)
            for i_n in gl.static_range(N):
                # [BD]
                if WN == 1:
                    # all streams reside in each thread's registers
                    b_doj = gl.sum(gl.where(o_n[:, None] == i_n, b_do, 0), axis=0)
                else:
                    b_doj = gl.load(do + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                # [BN], one output stream's weights across input streams
                b_mixing = gl.load(mixing + (i_t * N + i_n) * N + o_n, mask=o_n < N, other=0).to(gl.float32)
                b_dx += b_mixing[:, None] * b_doj[None, :]
                # [BD]
                b_xj = gl.load(x + (i_t * N + i_n) * D + o_d, mask=o_d < D, other=0).to(gl.float32)
                # [BN], one column of the mixing gradient for this feature tile
                b_dmixing = gl.sum(b_do * b_xj[None, :], axis=1)
                gl.store(dmixing_partial + ((i_t * ND + i_c) * N + o_n) * N + i_n, b_dmixing, mask=o_n < N)
        else:
            b_dx = b_do
    else:
        # [BD]
        b_do = gl.load(do + i_t * D + o_d, mask=o_d < D, other=0).to(gl.float32)
        # [BN, BD]
        b_x = gl.load(x + (i_t * N + o_n[:, None]) * D + o_d[None, :], mask=m_nd, other=0).to(gl.float32)
        b_dx = b_w[:, None] * b_do[None, :]
        # [BN, BD] -> [BN]
        b_dw = gl.sum(b_x * b_do[None, :], axis=1)
    gl.store(dx + (i_t * N + o_n[:, None]) * D + o_d[None, :], b_dx, mask=m_nd)
    gl.store(dw_partial + (i_t * ND + i_c) * N + o_n, b_dw, mask=o_n < N)


@fla_cache_autotune(
    configs=[
        triton.Config({'NW': num_warps}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['T', 'C', 'S'],
    do_bench=do_bench_cudagraph,
    **autotune_cache_kwargs,
)
@gluon.jit(do_not_specialize=['T'])
def hc_bwd_kernel_reduce_gluon(
    partial,
    grad,
    T,
    C: gl.constexpr,
    S: gl.constexpr,
    BC: gl.constexpr,
    BS: gl.constexpr,
    NW: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
    o_c = gl.arange(0, BC, layout=gl.SliceLayout(1, layout)).to(gl.int64)
    o_s = gl.arange(0, BS, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    # [BC, BS], feature tiles by routing-weight entries
    b_partial = gl.load(
        partial + (i_t * C + o_c[:, None]) * S + o_s[None, :],
        mask=(o_c[:, None] < C) & (o_s[None, :] < S),
        other=0,
    )
    # [BC, BS] -> [BS]
    gl.store(grad + i_t * S + o_s, gl.sum(b_partial, axis=0), mask=o_s < S)


def hc_fwd_gluon(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    is_post: bool,
):
    N, D = x.shape[-2:]
    # flatten all leading batch/sequence dimensions into T tokens
    T = x.numel() // (N * D)
    BD = min(256 if is_post else 1024, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    # [..., N, D] for post; [..., D] for pre
    o = torch.empty_like(x) if is_post else x.new_empty((*x.shape[:-2], D))
    hc_fwd_kernel_gluon[(T, ND)](
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


def hc_bwd_gluon(
    x: torch.Tensor,
    w: torch.Tensor,
    branch: torch.Tensor | None,
    mixing: torch.Tensor | None,
    do: torch.Tensor,
    is_post: bool,
):
    N, D = x.shape[-2:]
    T = x.numel() // (N * D)
    if mixing is not None:
        # smaller mixed-post tiles limit live registers for do and dx
        BD = 256 if N > 4 else 512
    elif not is_post and N <= 4:
        BD = 256
    else:
        BD = 1024
    BD = min(BD, triton.next_power_of_2(D))
    ND = triton.cdiv(D, BD)
    dx, dw = torch.empty_like(x), torch.empty_like(w)
    dbranch = torch.empty_like(branch) if is_post else None
    dmixing = torch.empty_like(mixing) if mixing is not None else None
    # [T, ND, N] and [T, ND, N * N], partial gradients along feature tiles
    dw_partial = torch.empty((T, ND, N), device=x.device, dtype=torch.float32)
    dmixing_partial = torch.empty((T, ND, N * N), device=x.device, dtype=torch.float32) if mixing is not None else None
    hc_bwd_kernel_gluon[(T, ND)](
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
            hc_bwd_kernel_reduce_gluon[(T,)](
                partial=partial,
                grad=grad,
                T=T,
                C=ND,
                S=size,
                BC=triton.next_power_of_2(ND),
                BS=triton.next_power_of_2(size),
            )
    return dx, dw, dbranch, dmixing


class FusedHCPreGluonFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, pre):
        o = hc_fwd_gluon(x=x, w=pre, branch=None, mixing=None, is_post=False)
        ctx.save_for_backward(x, pre)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, pre = ctx.saved_tensors
        dx, dpre, _, _ = hc_bwd_gluon(x=x, w=pre, branch=None, mixing=None, do=do, is_post=False)
        return dx, dpre


class FusedHCPostGluonFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, branch, post, mixing):
        o = hc_fwd_gluon(x=x, w=post, branch=branch, mixing=mixing, is_post=True)
        ctx.save_for_backward(x, branch, post, mixing)
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do):
        x, branch, post, mixing = ctx.saved_tensors
        dx, dpost, dbranch, dmixing = hc_bwd_gluon(x=x, w=post, branch=branch, mixing=mixing, do=do, is_post=True)
        return dx, dbranch, dpost, dmixing


@register(hc_pre, backend=GluonBackend)
def hc_pre_gluon(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return FusedHCPreGluonFunction.apply(x, pre)


@register(hc_post, backend=GluonBackend)
def hc_post_gluon(
    x: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mixing: torch.Tensor | None = None,
) -> torch.Tensor:
    return FusedHCPostGluonFunction.apply(x, branch, post, mixing)
