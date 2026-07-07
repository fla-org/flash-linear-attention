# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gluon backend for the fused AttnRes op (see ``fla/ops/attnres/fused.py`` for the Triton reference).

Where the Triton kernel leaves layout, shared memory, and load/compute overlap to the compiler, this
port makes them explicit: residual sources are indexed statically (``L`` is a constexpr, so no
pointer-table gather), V tiles are staged through shared memory with ``cp.async``, and the backward
keeps all ``L`` tiles resident when they fit and streams a 2-deep ring otherwise.

Opt-in and auto-dispatched like the other FLA backends: enable with ``FLA_ATTNRES_GLUON=1`` (off by
default); the verifier then selects it for suitable CUDA calls and falls back to Triton elsewhere.
Numerical parity with Triton is the frozen ``tests/ops/test_attnres.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

from fla.ops.backends import BaseBackend
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)

# tokens per bwd program (BT * KT); each program spills one fp32 dqw/dow partial row
GROUP = 32

# each config sets `num_warps=W` and constexpr `NW=W` together so the static layouts stay in sync
# with the launch warp count. num_stages is meaningless in Gluon (no compiler pipelining).
_FWD_CONFIGS = [
    triton.Config({'BT': BT, 'NW': num_warps}, num_warps=num_warps)
    for BT in [1, 2, 4]
    for num_warps in [4, 8]
]
_BWD_CONFIGS = [
    triton.Config({'BT': BT, 'NW': num_warps}, num_warps=num_warps)
    for BT in [1, 2, 4]
    for num_warps in [4, 8, 16]
]

# leave headroom below the 228KB Hopper/Blackwell carveout for barriers/paddings
_SMEM_BUDGET = 192 * 1024


def _prune_fwd_smem(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    def need(c): return 2 * c.kwargs['BT'] * args['BD'] * args['ES']  # noqa: E731
    keep = [c for c in configs if need(c) <= _SMEM_BUDGET] or configs[:1]
    if args['L'] > 8:
        # the source loop is fully unrolled; cap the sweep so big-L compiles stay sane
        keep = [c for c in keep if c.kwargs['BT'] == 2] or keep[:2]
    return keep


def _prune_bwd_smem(configs, named_args, **kwargs):
    # streaming fallback needs a 2-deep V ring plus the do tile
    args = {**named_args, **kwargs}
    def need(c): return 3 * c.kwargs['BT'] * args['BD'] * args['ES']  # noqa: E731
    keep = [c for c in configs if need(c) <= _SMEM_BUDGET] or configs[:1]
    if args['L'] > 8:
        # both passes unroll L tile bodies; cap the sweep so big-L compiles stay sane
        keep = [c for c in keep if c.kwargs['BT'] == 2 and c.num_warps in (8, 16)] or keep[:2]
    return keep


@gluon.jit
def _sum2_combine(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@gluon.constexpr_function
def _warp_split(BT, NW):
    # spread warps over tokens first, remainder over D
    w0 = min(BT, NW)
    return [w0, NW // w0]


@gluon.constexpr_function
def _lane_split(BD):
    # lanes cover [32 // t1, t1] so that 4-wide lane segments span exactly min(BD, 128) columns
    t1 = min(max(BD // 4, 1), 32)
    return [32 // t1, t1]


@gluon.constexpr_function
def _resident_tiles(L, BT, BD, ES):
    # True when all L v tiles plus the do tile fit in the smem budget
    return (L + 1) * BT * BD * ES <= 192 * 1024


@gluon.jit
def _dv_tile(b_v, b_do, b_qw, b_rstd, b_logit, b_p, b_delta, scale: gl.constexpr, D: gl.constexpr, L: gl.constexpr):
    # softmax bwd with delta already known; returns the dv tile and this tile's dqw contribution.
    # for L == 1 the softmax is constant so dlogit vanishes identically; branching on the constexpr
    # keeps ds exactly zero instead of relying on two reductions cancelling bitwise
    if L == 1:
        b_ds = gl.zeros_like(b_delta)
    else:
        b_dp = gl.sum(b_v * b_do, axis=1)
        b_ds = b_p * (b_dp - b_delta) * scale
    # [BT, BD]
    b_k = b_v * b_rstd[:, None]
    b_dv = b_p[:, None] * b_do + (b_ds * b_rstd)[:, None] * (b_qw[None, :] - b_k * (b_logit / D)[:, None])
    return b_dv, gl.sum(b_ds[:, None] * b_k, axis=0)


@fla_cache_autotune(
    configs=_FWD_CONFIGS,
    key=['L', 'D', 'R', 'ES', 'HAS_ONORM', 'SAVE_OPRE'],
    prune_configs_by={'early_config_prune': _prune_fwd_smem},
    **autotune_cache_kwargs,
)
@gluon.jit
def attnres_fwd_kernel_gluon(
    q,
    res,
    w,
    ow,
    o,
    o_pre,
    rstd,
    logit,
    lse,
    N,
    L: gl.constexpr,
    D: gl.constexpr,
    eps: gl.constexpr,
    scale: gl.constexpr,
    BT: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    R: gl.constexpr,
    ES: gl.constexpr,
    HAS_ONORM: gl.constexpr,
    SAVE_OPRE: gl.constexpr,
):
    W: gl.constexpr = _warp_split(BT, NW)
    # [BT, BD] row-major, R-wide lane segments along D; warps over tokens first, then D
    blk: gl.constexpr = gl.BlockedLayout([1, R], [1, 32], W, [1, 0])
    sl_d: gl.constexpr = gl.SliceLayout(0, blk)
    sl_t: gl.constexpr = gl.SliceLayout(1, blk)
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    NEEDS_DMASK: gl.constexpr = D != BD

    i_n = gl.program_id(0).to(gl.int64) * BT
    # [BT]; OOB rows are clamped to a valid token (never masked at load) and masked only at stores,
    # so no lane ever reads uninitialized smem under NaN poisoning
    o_t = i_n + gl.arange(0, BT, layout=sl_t)
    m_t = o_t < N
    o_tc = gl.minimum(o_t, N - 1)
    # [BD]
    o_d = gl.arange(0, BD, layout=sl_d)
    m_d = o_d < D

    b_qw = gl.load(q + o_d, mask=m_d, other=0.).to(gl.float32) * gl.load(w + o_d, mask=m_d, other=0.).to(gl.float32)

    # [BT, BD] global offsets shared by every source
    o_v = o_tc[:, None] * D + o_d[None, :]

    # double-buffered cp.async over the L sources
    b_vs = gl.allocate_shared_memory(res[0].dtype.element_ty, [2, BT, BD], smem_layout)
    if NEEDS_DMASK:
        cp.async_copy_global_to_shared(b_vs.index(0), res[0] + o_v, mask=m_d[None, :], eviction_policy="evict_first")
    else:
        cp.async_copy_global_to_shared(b_vs.index(0), res[0] + o_v, eviction_policy="evict_first")
    cp.commit_group()

    b_m = gl.full([BT], float('-inf'), gl.float32, layout=sl_t)
    b_acc = gl.zeros([BT], gl.float32, layout=sl_t)
    b_o = gl.zeros([BT, BD], gl.float32, layout=blk)
    for i_l in gl.static_range(L):
        if i_l + 1 < L:
            if NEEDS_DMASK:
                cp.async_copy_global_to_shared(
                    b_vs.index((i_l + 1) % 2),
                    res[i_l + 1] + o_v,
                    mask=m_d[None, :],
                    eviction_policy="evict_first",
                )
            else:
                cp.async_copy_global_to_shared(b_vs.index((i_l + 1) % 2), res[i_l + 1] + o_v, eviction_policy="evict_first")
            cp.commit_group()
            cp.wait_group(1)
        else:
            cp.wait_group(0)

        # [BT, BD]
        b_v = b_vs.index(i_l % 2).load(blk).to(gl.float32)
        if NEEDS_DMASK:
            b_v = gl.where(m_d[None, :], b_v, 0.)

        # one cross-warp pass for both row reductions
        b_v2, b_vq = gl.reduce((b_v * b_v, b_v * b_qw[None, :]), axis=1, combine_fn=_sum2_combine)
        # [BT]
        b_rstd = gl.rsqrt(b_v2 / D + eps)
        b_logit = b_vq * b_rstd
        b_s = b_logit * scale

        b_m, b_mp = gl.maximum(b_m, b_s), b_m
        b_r = gl.exp(b_mp - b_m)
        b_p = gl.exp(b_s - b_m)
        b_acc = b_acc * b_r + b_p
        b_o = b_o * b_r[:, None] + b_p[:, None] * b_v

        gl.store(rstd + i_l * N + o_t, b_rstd.to(rstd.dtype.element_ty), mask=m_t)
        gl.store(logit + i_l * N + o_t, b_logit.to(logit.dtype.element_ty), mask=m_t)
        # cp.async is same-lane staging, but the CTA must sync before the next issue reuses slot i_l % 2
        gl.thread_barrier()

    gl.store(lse + o_t, b_m + gl.log(b_acc), mask=m_t)

    b_o = b_o / b_acc[:, None]
    if SAVE_OPRE:
        gl.store(o_pre + o_v, b_o.to(o_pre.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])
    if HAS_ONORM:
        # masked cols are already zero, so the row reduction needs no extra select
        b_o_rstd = gl.rsqrt(gl.sum(b_o * b_o, axis=1) / D + eps)
        b_ow = gl.load(ow + o_d, mask=m_d, other=0.).to(gl.float32)
        b_o = b_o * b_o_rstd[:, None] * b_ow[None, :]
    gl.store(o + o_v, b_o.to(o.dtype.element_ty), mask=m_t[:, None] & m_d[None, :])


@fla_cache_autotune(
    configs=_BWD_CONFIGS,
    key=['L', 'D', 'R', 'ES', 'HAS_ONORM', 'SAVE_OPRE'],
    prune_configs_by={'early_config_prune': _prune_bwd_smem},
    **autotune_cache_kwargs,
)
@gluon.jit
def attnres_bwd_kernel_dv_gluon(
    q,
    res,
    w,
    ow,
    o_pre,
    rstd,
    logit,
    lse,
    do,
    dres,
    dqw,
    dow_partial,
    N,
    L: gl.constexpr,
    D: gl.constexpr,
    eps: gl.constexpr,
    scale: gl.constexpr,
    G: gl.constexpr,
    BT: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    R: gl.constexpr,
    ES: gl.constexpr,
    HAS_ONORM: gl.constexpr,
    SAVE_OPRE: gl.constexpr,
):
    W: gl.constexpr = _warp_split(BT, NW)
    blk: gl.constexpr = gl.BlockedLayout([1, R], [1, 32], W, [1, 0])
    sl_d: gl.constexpr = gl.SliceLayout(0, blk)
    sl_t: gl.constexpr = gl.SliceLayout(1, blk)
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    NEEDS_DMASK: gl.constexpr = D != BD
    KT: gl.constexpr = G // BT
    # all L v tiles stay resident when they fit (V then hits global exactly once per token);
    # otherwise they stream through a 2-deep ring and each pass re-reads V, like Triton
    RESIDENT: gl.constexpr = _resident_tiles(L, BT, BD, ES)
    NB: gl.constexpr = L if RESIDENT else 2

    i_g = gl.program_id(0).to(gl.int64) * G

    o_d = gl.arange(0, BD, layout=sl_d)
    m_d = o_d < D
    b_qw = gl.load(q + o_d, mask=m_d, other=0.).to(gl.float32) * gl.load(w + o_d, mask=m_d, other=0.).to(gl.float32)
    if HAS_ONORM:
        b_ow = gl.load(ow + o_d, mask=m_d, other=0.).to(gl.float32)

    # [BD] fp32 partials accumulated across all G tokens, spilled once at the end
    b_dqw = gl.zeros([BD], gl.float32, layout=sl_d)
    b_dow = gl.zeros([BD], gl.float32, layout=sl_d)

    b_vs = gl.allocate_shared_memory(res[0].dtype.element_ty, [NB, BT, BD], smem_layout)
    b_dos = gl.allocate_shared_memory(do.dtype.element_ty, [BT, BD], smem_layout)

    for t in range(KT):
        # [BT]; OOB rows are clamped to a valid token and their do rows zeroed below, so every
        # masked contribution vanishes without reading uninitialized smem under NaN poisoning
        o_t = i_g + t * BT + gl.arange(0, BT, layout=sl_t)
        m_t = o_t < N
        o_tc = gl.minimum(o_t, N - 1)
        o_v = o_tc[:, None] * D + o_d[None, :]

        # do goes first so its wait count is independent of the V schedule
        if NEEDS_DMASK:
            cp.async_copy_global_to_shared(b_dos, do + o_v, mask=m_d[None, :])
        else:
            cp.async_copy_global_to_shared(b_dos, do + o_v)
        cp.commit_group()
        if RESIDENT:
            for i_l in gl.static_range(L):
                if NEEDS_DMASK:
                    cp.async_copy_global_to_shared(
                        b_vs.index(i_l),
                        res[i_l] + o_v,
                        mask=m_d[None, :],
                        eviction_policy="evict_first",
                    )
                else:
                    cp.async_copy_global_to_shared(b_vs.index(i_l), res[i_l] + o_v, eviction_policy="evict_first")
                cp.commit_group()
            cp.wait_group(L)
        else:
            if NEEDS_DMASK:
                cp.async_copy_global_to_shared(b_vs.index(0), res[0] + o_v, mask=m_d[None, :], eviction_policy="evict_first")
            else:
                cp.async_copy_global_to_shared(b_vs.index(0), res[0] + o_v, eviction_policy="evict_first")
            cp.commit_group()
            cp.wait_group(1)

        b_lse = gl.load(lse + o_tc).to(gl.float32)
        # [BT, BD]
        b_do = b_dos.load(blk).to(gl.float32)
        b_do = gl.where(m_t[:, None], b_do, 0.)
        if NEEDS_DMASK:
            b_do = gl.where(m_d[None, :], b_do, 0.)

        if SAVE_OPRE:
            b_opre = gl.load(o_pre + o_v, mask=m_d[None, :], other=0.).to(gl.float32)
            if RESIDENT:
                cp.wait_group(0)
        else:
            # level 1: recompute the mix sum_l p_l * v_l
            b_opre = gl.zeros([BT, BD], gl.float32, layout=blk)
            for i_l in gl.static_range(L):
                if RESIDENT:
                    cp.wait_group(L - 1 - i_l)
                    b_v = b_vs.index(i_l).load(blk).to(gl.float32)
                else:
                    if i_l + 1 < L:
                        if NEEDS_DMASK:
                            cp.async_copy_global_to_shared(
                                b_vs.index((i_l + 1) % 2),
                                res[i_l + 1] + o_v,
                                mask=m_d[None, :],
                                eviction_policy="evict_first",
                            )
                        else:
                            cp.async_copy_global_to_shared(
                                b_vs.index((i_l + 1) % 2),
                                res[i_l + 1] + o_v,
                                eviction_policy="evict_first",
                            )
                        cp.commit_group()
                        cp.wait_group(1)
                    else:
                        cp.wait_group(0)
                    b_v = b_vs.index(i_l % 2).load(blk).to(gl.float32)
                if NEEDS_DMASK:
                    b_v = gl.where(m_d[None, :], b_v, 0.)
                b_logit = gl.load(logit + i_l * N + o_tc).to(gl.float32)
                b_p = gl.exp(b_logit * scale - b_lse)
                b_opre += b_p[:, None] * b_v
                if not RESIDENT:
                    # the slot just read is the next issue target
                    gl.thread_barrier()

        # output RMSNorm bwd: turn b_do into the gradient w.r.t. the pre-norm output
        if HAS_ONORM:
            b_o_rstd = gl.rsqrt(gl.sum(b_opre * b_opre, axis=1) / D + eps)
            b_xhat = b_opre * b_o_rstd[:, None]
            b_c1 = gl.sum(b_xhat * b_ow[None, :] * b_do, axis=1) / D
            b_dow += gl.sum(b_xhat * b_do, axis=0)
            b_do = (b_ow[None, :] * b_do - b_xhat * b_c1[:, None]) * b_o_rstd[:, None]
        # [BT] delta = sum_l p*dp = <do_pre, o_pre>
        b_delta = gl.sum(b_do * b_opre, axis=1)

        if not RESIDENT and not SAVE_OPRE:
            # re-prime the ring for the second streaming pass
            if NEEDS_DMASK:
                cp.async_copy_global_to_shared(b_vs.index(0), res[0] + o_v, mask=m_d[None, :], eviction_policy="evict_first")
            else:
                cp.async_copy_global_to_shared(b_vs.index(0), res[0] + o_v, eviction_policy="evict_first")
            cp.commit_group()

        for i_l in gl.static_range(L):
            if RESIDENT:
                # tiles were already waited on above; second use comes straight from smem
                b_v = b_vs.index(i_l).load(blk).to(gl.float32)
            else:
                if i_l + 1 < L:
                    if NEEDS_DMASK:
                        cp.async_copy_global_to_shared(
                            b_vs.index((i_l + 1) % 2),
                            res[i_l + 1] + o_v,
                            mask=m_d[None, :],
                            eviction_policy="evict_first",
                        )
                    else:
                        cp.async_copy_global_to_shared(
                            b_vs.index((i_l + 1) % 2),
                            res[i_l + 1] + o_v,
                            eviction_policy="evict_first",
                        )
                    cp.commit_group()
                    cp.wait_group(1)
                else:
                    cp.wait_group(0)
                b_v = b_vs.index(i_l % 2).load(blk).to(gl.float32)
            if NEEDS_DMASK:
                b_v = gl.where(m_d[None, :], b_v, 0.)
            # [BT]
            b_rstd = gl.load(rstd + i_l * N + o_tc).to(gl.float32)
            b_logit = gl.load(logit + i_l * N + o_tc).to(gl.float32)
            b_p = gl.exp(b_logit * scale - b_lse)

            b_dv, b_inc = _dv_tile(b_v, b_do, b_qw, b_rstd, b_logit, b_p, b_delta, scale, D, L)
            gl.store(dres[i_l] + o_v, b_dv.to(dres[0].dtype.element_ty), mask=m_t[:, None] & m_d[None, :])
            b_dqw += b_inc
            if not RESIDENT:
                gl.thread_barrier()
        # tiles of iteration t are fully consumed before iteration t+1 refills the buffers
        gl.thread_barrier()

    i_p = gl.program_id(0).to(gl.int64)
    gl.store(dqw + i_p * D + o_d, b_dqw, mask=m_d)
    if HAS_ONORM:
        gl.store(dow_partial + i_p * D + o_d, b_dow, mask=m_d)


@fla_cache_autotune(
    configs=[
        triton.Config({'BN': BN, 'BD': BD, 'NW': num_warps}, num_warps=num_warps)
        for BN, BD, num_warps in [(256, 32, 4), (512, 64, 4), (512, 64, 8), (1024, 128, 8)]
    ],
    key=['NP', 'D', 'HAS_ONORM'],
    **autotune_cache_kwargs,
)
@gluon.jit
def attnres_bwd_kernel_dqdw_gluon(
    q,
    w,
    dqw,
    dow_partial,
    dq,
    dw,
    dow,
    NP,
    D: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    HAS_ONORM: gl.constexpr,
):
    i_d = gl.program_id(0).to(gl.int32)

    # [BN, BD] fp32 rows; 4-wide lane segments along D, warps along the partial rows
    T: gl.constexpr = _lane_split(BD)
    blk: gl.constexpr = gl.BlockedLayout([1, 4], T, [NW, 1], [1, 0])
    sl_d: gl.constexpr = gl.SliceLayout(0, blk)
    sl_n: gl.constexpr = gl.SliceLayout(1, blk)

    o_d = i_d * BD + gl.arange(0, BD, layout=sl_d)
    m_d = o_d < D

    # column-sum the [NP, D] partials
    b_dqw = gl.zeros([BD], gl.float32, layout=sl_d)
    b_dow = gl.zeros([BD], gl.float32, layout=sl_d)
    for i_n in range(0, NP, BN):
        o_n = i_n + gl.arange(0, BN, layout=sl_n)
        m = (o_n < NP)[:, None] & m_d[None, :]
        p = o_n[:, None].to(gl.int64) * D + o_d[None, :]
        b_dqw += gl.sum(gl.load(dqw + p, mask=m, other=0.), axis=0)
        if HAS_ONORM:
            b_dow += gl.sum(gl.load(dow_partial + p, mask=m, other=0.), axis=0)

    b_q = gl.load(q + o_d, mask=m_d, other=0.).to(gl.float32)
    b_w = gl.load(w + o_d, mask=m_d, other=0.).to(gl.float32)
    gl.store(dq + o_d, (b_dqw * b_w).to(dq.dtype.element_ty), mask=m_d)
    gl.store(dw + o_d, (b_dqw * b_q).to(dw.dtype.element_ty), mask=m_d)
    if HAS_ONORM:
        gl.store(dow + o_d, b_dow.to(dow.dtype.element_ty), mask=m_d)


def _vec_width(dtype: torch.dtype, d: int, bd: int) -> int:
    # widest lane segment (<= 16B) that divides D without forcing intra-warp broadcast;
    # cp.async needs >= 4B per segment
    r = min(16 // dtype.itemsize, max(1, bd // 32))
    while r > 1 and d % r != 0:
        r //= 2
    if r * dtype.itemsize < 4:
        raise ValueError(f"Gluon attnres requires D * itemsize >= 128 bytes, got D={d} ({dtype})")
    return r


def _check_sources(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    for t in tensors:
        assert t.data_ptr() % 16 == 0, "attnres residual sources must be 16-byte aligned"
    return tuple(tensors)


def _fused_attnres_fwd(
    q: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    res: tuple[torch.Tensor, ...],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not residuals[0].is_cuda:
        raise ValueError("Gluon attnres requires CUDA tensors")

    output_shape = residuals[0].shape
    L, N, D = len(residuals), residuals[0].numel() // output_shape[-1], output_shape[-1]

    dtype = residuals[0].dtype
    stats_shape = (L, *output_shape[:-1])

    has_onorm = ow is not None
    save_opre = checkpoint_level == 0
    o = torch.empty(output_shape, device=residuals[0].device, dtype=dtype)
    o_pre = torch.empty(output_shape, device=residuals[0].device, dtype=dtype) if save_opre else None
    lse = torch.empty(output_shape[:-1], device=residuals[0].device, dtype=torch.float32)
    rstd = torch.empty(stats_shape, device=residuals[0].device, dtype=torch.float32)
    logit = torch.empty_like(rstd)

    def grid(meta): return (triton.cdiv(N, meta['BT']),)
    attnres_fwd_kernel_gluon[grid](
        q=q,
        res=res,
        w=w,
        ow=ow,
        o=o,
        o_pre=o_pre,
        rstd=rstd,
        logit=logit,
        lse=lse,
        N=N,
        L=L,
        D=D,
        eps=eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        R=_vec_width(dtype, D, triton.next_power_of_2(D)),
        ES=dtype.itemsize,
        HAS_ONORM=has_onorm,
        SAVE_OPRE=save_opre,
    )

    return o, o_pre, rstd, logit, lse


def _fused_attnres_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    res: tuple[torch.Tensor, ...],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    o_pre: torch.Tensor | None,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    L, N, D = len(residuals), do.numel() // do.shape[-1], do.shape[-1]
    NP = triton.cdiv(N, GROUP)

    # bwd_dv produces dvs + per-program dqw/dow partials [NP, D]; bwd_dqdw reduces them into dq / dw / dow
    has_onorm = ow is not None
    if has_onorm:
        dow_partial = torch.empty(NP, D, device=do.device, dtype=torch.float32)
        dow = torch.empty_like(ow)
    else:
        dow_partial = dow = None

    dvs = [torch.empty_like(r) for r in residuals]
    dres = _check_sources(dvs)
    dqw = torch.empty(NP, D, device=do.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)

    attnres_bwd_kernel_dv_gluon[(NP,)](
        q=q,
        res=res,
        w=w,
        ow=ow,
        o_pre=o_pre,
        rstd=rstd,
        logit=logit,
        lse=lse,
        do=do,
        dres=dres,
        dqw=dqw,
        dow_partial=dow_partial,
        N=N,
        L=L,
        D=D,
        eps=eps,
        scale=scale,
        G=GROUP,
        BD=triton.next_power_of_2(D),
        R=_vec_width(do.dtype, D, triton.next_power_of_2(D)),
        ES=do.dtype.itemsize,
        HAS_ONORM=has_onorm,
        SAVE_OPRE=checkpoint_level == 0,
    )

    def grid(meta): return (triton.cdiv(D, meta['BD']),)
    attnres_bwd_kernel_dqdw_gluon[grid](
        q=q,
        w=w,
        dqw=dqw,
        dow_partial=dow_partial,
        dq=dq,
        dw=dw,
        dow=dow,
        NP=NP,
        D=D,
        HAS_ONORM=has_onorm,
    )

    return dvs, dq, dw, dow


class FusedAttnresGluonFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        query: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None,
        rms_eps: float,
        scale: float,
        return_weights: bool,
        checkpoint_level: int,
        *residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        res = _check_sources(residuals)
        o, o_pre, rstd, logit, lse = _fused_attnres_fwd(
            q=query,
            residuals=residuals,
            res=res,
            w=rms_weight,
            ow=output_rms_weight,
            eps=rms_eps,
            scale=scale,
            checkpoint_level=checkpoint_level,
        )
        ctx.save_for_backward(query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, *residuals)
        ctx.eps = rms_eps
        ctx.scale = scale
        ctx.checkpoint_level = checkpoint_level
        ctx.res = res
        # probs are materialized only when requested; bwd recomputes them from logit + lse
        p = (logit * scale - lse).exp() if return_weights else o.new_empty(0)
        ctx.mark_non_differentiable(p)
        return o, p

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dp: torch.Tensor | None = None,
    ):
        del dp
        query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, *residuals = ctx.saved_tensors
        dvs, dq, dw, dow = _fused_attnres_bwd(
            do=do,
            q=query,
            residuals=residuals,
            res=ctx.res,
            w=rms_weight,
            ow=output_rms_weight,
            o_pre=o_pre,
            rstd=rstd,
            logit=logit,
            lse=lse,
            eps=ctx.eps,
            scale=ctx.scale,
            checkpoint_level=ctx.checkpoint_level,
        )
        return (dq, dw, dow, None, None, None, None, *dvs)


def _run(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    checkpoint_level: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    output_shape = residuals[0].shape
    D = output_shape[-1]
    flat_residuals = tuple(r.reshape(-1, D).contiguous() for r in residuals)

    o, p = FusedAttnresGluonFunction.apply(
        query, rms_weight, output_rms_weight, rms_eps, scale, return_weights, checkpoint_level, *flat_residuals,
    )
    o = o.view(output_shape)

    if return_weights:
        p = p.view(len(residuals), *output_shape[:-1])
        return o, p
    return o


class AttnResGluonBackend(BaseBackend):
    """Dispatch entry for the Gluon AttnRes kernels (see the module docstring for the design).

    Off by default; enable with ``FLA_ATTNRES_GLUON=1``. The verifier then accepts any CUDA call
    on SM80+ with ``D * itemsize >= 128`` bytes and otherwise defers to the Triton path.
    """

    backend_type = "gluon"
    package_name = "triton.experimental.gluon"  # ships with Triton 3.5+; absent on older builds
    env_var = "FLA_ATTNRES_GLUON"
    default_enable = False
    priority = 3

    def fused_attnres_verifier(
        self,
        query: torch.Tensor,
        residuals: Sequence[torch.Tensor],
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor | None = None,
        rms_eps: float = 1e-6,
        scale: float = 1.0,
        return_weights: bool = False,
        checkpoint_level: int = 1,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if not residuals or not residuals[0].is_cuda:
            return False, "gluon AttnRes backend requires CUDA tensors"
        v = residuals[0]
        if torch.cuda.get_device_capability(v.device)[0] < 8:
            return False, "gluon AttnRes backend requires NVIDIA SM80+ (cp.async)"
        if v.shape[-1] * v.element_size() < 128:
            return False, "gluon AttnRes backend requires D * itemsize >= 128 bytes"
        return True, None

    def fused_attnres(self, *args, **kwargs):
        return _run(*args, **kwargs)


__all__ = ["AttnResGluonBackend"]
