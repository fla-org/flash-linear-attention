# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused AttnRes kernels for triton-ascend on Ascend NPU.

GPU passes a padded tuple of per-source pointers (``res[i]``). Triton-Ascend
cannot bind tuple arguments, so sources are stacked in L-chunks as
``[L_chunk, N, D]`` (capped at 768 MiB per chunk). Cross-chunk softmax
stats are merged on the host; the weighted sum accumulates in a single
fp32 ``o_mix`` buffer to match GPU register precision.

A 2D gather along L compiles to scalar, so ``fwd_p2`` / ``bwd_dv`` use 1D
``[BD]`` tiles when ``BN==1``, and contiguous ``[BN, BD]`` tiles along N
otherwise. ``BN>1`` is only launched when ``SINGLE_D`` (``BD >= D``); the
host falls back to ``BN==1`` for the D-loop path. ``SINGLE_D`` keeps
``q*w`` in UB; no-onorm ``SINGLE_D`` fuses ``bwd_prep`` (``FUSE_PREP``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    iter_axis_launch_chunks,
)

_FWD_MEM_MULT = 3.0
_BWD_DV_MEM_MULT = 6.0
_FWD_BN_MEM_MULT = 3.0
# bwd_dv holds more live tiles under Ascend multi-buffering than the host UB model.
_BWD_BN_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.85
_FALLBACK_BD = 256
_MAX_BN = 8
_MAX_L_CHUNK_BYTES = 768 << 20


def _l_chunk_size(L: int, N: int, D: int, elem_size: int) -> int:
    layer_bytes = N * D * elem_size
    if layer_bytes == 0:
        return L
    return max(1, min(L, _MAX_L_CHUNK_BYTES // layer_bytes))


def _flat_residual(r: torch.Tensor, D: int) -> torch.Tensor:
    """View ``[..., D]`` as ``[N, D]`` for the autograd path."""
    return r.reshape(-1, D).contiguous()


def _stack_sources(sources: Sequence[torch.Tensor], l0: int, l1: int) -> torch.Tensor:
    if l1 - l0 == 1:
        return sources[l0].unsqueeze(0)
    return torch.stack(sources[l0:l1], dim=0)


def _iter_l_chunks(
    sources: Sequence[torch.Tensor],
) -> Iterator[tuple[int, int, torch.Tensor, int, int, int]]:
    L, N, D = len(sources), *sources[0].shape
    stride_ln, l_chunk = N * D, _l_chunk_size(L, N, D, sources[0].element_size())
    for l0 in range(0, L, l_chunk):
        l1 = min(l0 + l_chunk, L)
        yield l0, l1 - l0, _stack_sources(sources, l0, l1), stride_ln, N, D


def _merge_online_softmax(
    m: torch.Tensor,
    acc: torch.Tensor,
    m_chunk: torch.Tensor,
    acc_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m_new = torch.maximum(m, m_chunk)
    acc_new = acc * torch.exp(m - m_new) + acc_chunk * torch.exp(m_chunk - m_new)
    return m_new, acc_new


def _get_bd(row_dim: int, col_dim: int, *, memory_multiplier: float) -> int:
    return compute_row_tile_block_size(
        row_dim,
        col_dim,
        memory_multiplier,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=_FALLBACK_BD,
        min_block=64,
        max_block=8192,
    )


def _get_bl_bd(ll: int, D: int, *, mem_mult: float) -> tuple[int, int]:
    bl = min(8, max(1, triton.next_power_of_2(ll)))
    return bl, _get_bd(bl, D, memory_multiplier=mem_mult)


def _get_bn(N: int, BD: int, *, mem_mult: float) -> int:
    return compute_row_tile_block_size(
        N,
        BD,
        memory_multiplier=mem_mult,
        tiling_row=True,
        safety_margin=_SAFETY_MARGIN,
        dtype_size=4,
        fallback=1,
        min_block=1,
        max_block=_MAX_BN,
    )


def _n_tile_bn(N: int, BD: int, D: int, *, mem_mult: float) -> int:
    """N-tile only when D fits in one BD tile (``SINGLE_D``).

    The ``BN>1 && BD < D`` D-loop is untested (A2 never reaches it; bwd
    ``mem_mult=8`` keeps ``BN=1`` whenever ``BD < D``). ``BN==1`` already
    covers that case.
    """
    if BD < D:
        return 1
    return _get_bn(N, BD, mem_mult=mem_mult)


def _launch_n(
    kernel: Callable,
    N: int,
    /,
    **kwargs,
) -> None:
    bn = int(kwargs.get('BN', 1))
    n_tiles = triton.cdiv(N, max(bn, 1))
    for n_off, n_len in iter_axis_launch_chunks(n_tiles, 1, max_grid=ASCEND_MAX_GRID_DIM):
        kernel[(n_len,)](N_OFFSET=n_off * bn, **kwargs)


@triton.jit(do_not_specialize=['L', 'N', 'D', 'L_OFFSET'])
def attnres_fwd_p1_chunk_kernel(
    q,
    res,
    w,
    rstd,
    logit,
    chunk_m,
    chunk_acc,
    L,
    N,
    D,
    stride_ln,
    eps,
    scale,
    BL: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    L_OFFSET,
    N_OFFSET: tl.constexpr,
):
    # BN==1: [BL, BD] along L. BN>1 implies SINGLE_D: [BN, BD] along N, q*w hoisted.
    i_n0 = tl.program_id(0).to(tl.int64) * BN + N_OFFSET
    if BN == 1:
        i_n = i_n0
        b_m = tl.full([], float('-inf'), dtype=tl.float32)
        b_acc = tl.zeros([], dtype=tl.float32)
        for i_l in range(tl.cdiv(L, BL)):
            o_l = (i_l * BL + tl.arange(0, BL)).to(tl.int64)
            m_l = o_l < L
            b_v_sq = tl.zeros([BL], dtype=tl.float32)
            b_v_qw = tl.zeros([BL], dtype=tl.float32)
            for i_d in range(tl.cdiv(D, BD)):
                o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
                m_d = o_d < D
                b_qw = (
                    tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
                    * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
                )
                b_v = tl.load(
                    res + o_l[:, None] * stride_ln + i_n * D + o_d[None, :],
                    mask=m_l[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                b_v_sq += tl.sum(b_v * b_v, axis=1)
                b_v_qw += tl.sum(b_v * b_qw[None, :], axis=1)
            b_rstd = tl.rsqrt(b_v_sq / D + eps)
            b_logit = b_v_qw * b_rstd
            b_s = tl.where(m_l, b_logit * scale, float('-inf'))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
            b_acc = b_acc * exp(b_mp - b_m) + tl.sum(exp(b_s - b_m), axis=0)
            g_l = L_OFFSET + o_l
            tl.store(rstd + g_l * N + i_n, b_rstd.to(rstd.dtype.element_ty), mask=m_l)
            tl.store(logit + g_l * N + i_n, b_logit.to(logit.dtype.element_ty), mask=m_l)
        tl.store(chunk_m + i_n, b_m)
        tl.store(chunk_acc + i_n, b_acc)
    else:
        o_n = i_n0 + tl.arange(0, BN).to(tl.int64)
        m_n = o_n < N
        b_m = tl.full([BN], float('-inf'), dtype=tl.float32)
        b_acc = tl.zeros([BN], dtype=tl.float32)
        o_d = tl.arange(0, BD).to(tl.int64)
        m_d = o_d < D
        m_nd = m_n[:, None] & m_d[None, :]
        b_qw = (
            tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
            * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        )
        for i_l in range(L):
            i_l64 = tl.cast(i_l, tl.int64)
            g_l = L_OFFSET + i_l64
            b_v = tl.load(
                res + i_l64 * stride_ln + o_n[:, None] * D + o_d[None, :],
                mask=m_nd,
                other=0.0,
            ).to(tl.float32)
            b_v_sq = tl.sum(tl.where(m_nd, b_v * b_v, 0.0), axis=1)
            b_v_qw = tl.sum(tl.where(m_nd, b_v * b_qw[None, :], 0.0), axis=1)
            b_rstd = tl.rsqrt(b_v_sq / D + eps)
            b_logit = b_v_qw * b_rstd
            b_s = tl.where(m_n, b_logit * scale, float('-inf'))
            b_mp = b_m
            b_m = tl.maximum(b_m, b_s)
            b_acc = b_acc * exp(b_mp - b_m) + exp(b_s - b_m)
            tl.store(rstd + g_l * N + o_n, b_rstd.to(rstd.dtype.element_ty), mask=m_n)
            tl.store(logit + g_l * N + o_n, b_logit.to(logit.dtype.element_ty), mask=m_n)
        tl.store(chunk_m + o_n, b_m, mask=m_n)
        tl.store(chunk_acc + o_n, b_acc, mask=m_n)


@triton.jit(do_not_specialize=['L', 'N', 'D', 'L_OFFSET'])
def attnres_fwd_p2_chunk_kernel(
    res,
    logit,
    lse,
    o_mix,
    L,
    N,
    D,
    stride_ln,
    scale,
    BD: tl.constexpr,
    BN: tl.constexpr,
    L_OFFSET,
    N_OFFSET: tl.constexpr,
    LOAD_OMIX: tl.constexpr,
):
    # BN==1: 1D [BD]. BN>1: [BN, BD] along N. D-outer keeps b_o in UB; LOAD_OMIX=0 skips a zero load.
    i_n0 = tl.program_id(0).to(tl.int64) * BN + N_OFFSET
    if BN == 1:
        i_n = i_n0
        b_lse = tl.load(lse + i_n).to(tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            if LOAD_OMIX:
                b_o = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            else:
                b_o = tl.zeros([BD], dtype=tl.float32)
            for i_l in range(L):
                i_l64 = tl.cast(i_l, tl.int64)
                g_l = L_OFFSET + i_l64
                b_logit = tl.load(logit + g_l * N + i_n).to(tl.float32)
                b_p = exp(b_logit * scale - b_lse)
                b_v = tl.load(
                    res + i_l64 * stride_ln + i_n * D + o_d,
                    mask=m_d,
                    other=0.0,
                ).to(tl.float32)
                b_o += b_p * b_v
            tl.store(o_mix + i_n * D + o_d, b_o, mask=m_d)
    else:
        o_n = i_n0 + tl.arange(0, BN).to(tl.int64)
        m_n = o_n < N
        b_lse = tl.load(lse + o_n, mask=m_n, other=0.).to(tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            m_nd = m_n[:, None] & m_d[None, :]
            if LOAD_OMIX:
                b_o = tl.load(
                    o_mix + o_n[:, None] * D + o_d[None, :],
                    mask=m_nd,
                    other=0.,
                ).to(tl.float32)
            else:
                b_o = tl.zeros([BN, BD], dtype=tl.float32)
            for i_l in range(L):
                i_l64 = tl.cast(i_l, tl.int64)
                g_l = L_OFFSET + i_l64
                b_logit = tl.load(logit + g_l * N + o_n, mask=m_n, other=0.).to(tl.float32)
                b_p = exp(b_logit * scale - b_lse)
                b_v = tl.load(
                    res + i_l64 * stride_ln + o_n[:, None] * D + o_d[None, :],
                    mask=m_nd,
                    other=0.0,
                ).to(tl.float32)
                b_o += b_p[:, None] * b_v
            tl.store(o_mix + o_n[:, None] * D + o_d[None, :], b_o, mask=m_nd)


@triton.jit(do_not_specialize=['N', 'D'])
def attnres_fwd_onorm_kernel(
    o,
    o_mix,
    ow,
    N,
    D,
    eps,
    BD: tl.constexpr,
    N_OFFSET: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64) + N_OFFSET
    b_o_sq = tl.zeros([], dtype=tl.float32)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
        m_d = o_d < D
        b_o = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_o_sq += tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0)
    b_o_rstd = tl.rsqrt(b_o_sq / D + eps)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
        m_d = o_d < D
        b_o = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
        tl.store(o + i_n * D + o_d, (b_o * b_o_rstd * b_ow).to(o.dtype.element_ty), mask=m_d)


@triton.jit(do_not_specialize=['N', 'D'])
def attnres_bwd_prep_kernel(
    ow,
    o_mix,
    do,
    do_eff,
    dow_partial,
    b_delta,
    N,
    D,
    eps,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    N_OFFSET: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64) + N_OFFSET
    if HAS_ONORM:
        b_o_sq = tl.zeros([], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_sq += tl.sum(tl.where(m_d, b_o_pre * b_o_pre, 0.0), axis=0)
        b_o_rstd = tl.rsqrt(b_o_sq / D + eps)
        b_c1 = tl.zeros([], dtype=tl.float32)
        for i_d in range(tl.cdiv(D, BD)):
            o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
            m_d = o_d < D
            b_do = tl.load(do + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            b_c1 += tl.sum(tl.where(m_d, b_o_pre * b_o_rstd * b_ow * b_do, 0.0), axis=0)
        b_c1 /= D

    b_delta_acc = tl.zeros([], dtype=tl.float32)
    for i_d in range(tl.cdiv(D, BD)):
        o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
        m_d = o_d < D
        b_do = tl.load(do + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        b_o_pre = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
        if HAS_ONORM:
            b_ow = tl.load(ow + o_d, mask=m_d, other=0.).to(tl.float32)
            b_xhat = b_o_pre * b_o_rstd
            tl.store(dow_partial + i_n * D + o_d, (b_xhat * b_do).to(dow_partial.dtype.element_ty), mask=m_d)
            b_do = (b_ow * b_do - b_xhat * b_c1) * b_o_rstd
        tl.store(do_eff + i_n * D + o_d, b_do, mask=m_d)
        b_delta_acc += tl.sum(tl.where(m_d, b_do * b_o_pre, 0.0), axis=0)
    tl.store(b_delta + i_n, b_delta_acc)


@triton.jit(do_not_specialize=['L', 'N', 'D', 'L_OFFSET'])
def attnres_bwd_dv_chunk_kernel(
    q,
    res,
    w,
    rstd,
    logit,
    lse,
    do_eff,
    o_mix,
    dres,
    dqw,
    b_delta,
    L,
    N,
    D,
    stride_ln,
    scale,
    BD: tl.constexpr,
    BN: tl.constexpr,
    SINGLE_D: tl.constexpr,
    LOAD_DQW: tl.constexpr,
    FUSE_PREP: tl.constexpr,
    L_OFFSET,
    N_OFFSET: tl.constexpr,
):
    # BN==1: 1D [BD]. BN>1 implies SINGLE_D: [BN, BD] along N. FUSE_PREP inlines delta from o_mix.
    i_n0 = tl.program_id(0).to(tl.int64) * BN + N_OFFSET
    if BN == 1:
        i_n = i_n0
        b_lse = tl.load(lse + i_n).to(tl.float32)
        if SINGLE_D:
            o_d = tl.arange(0, BD).to(tl.int64)
            m_d = o_d < D
            b_qw = (
                tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
                * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
            )
            b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
            if FUSE_PREP:
                b_dqw = tl.load(o_mix + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
                b_delta_n = tl.sum(tl.where(m_d, b_do * b_dqw, 0.0), axis=0)
                if LOAD_DQW:
                    b_dqw = tl.load(dqw + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
                else:
                    b_dqw = b_dqw * 0.0
            else:
                b_delta_n = tl.load(b_delta + i_n).to(tl.float32)
                if LOAD_DQW:
                    b_dqw = tl.load(dqw + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
                else:
                    b_dqw = tl.zeros([BD], dtype=tl.float32)
            for i_l in range(L):
                i_l64 = tl.cast(i_l, tl.int64)
                g_l = L_OFFSET + i_l64
                b_rstd = tl.load(rstd + g_l * N + i_n).to(tl.float32)
                b_logit = tl.load(logit + g_l * N + i_n).to(tl.float32)
                b_p = exp(b_logit * scale - b_lse)
                b_v = tl.load(
                    res + i_l64 * stride_ln + i_n * D + o_d,
                    mask=m_d,
                    other=0.0,
                ).to(tl.float32)
                b_dp = tl.sum(tl.where(m_d, b_v * b_do, 0.0), axis=0)
                b_ds = b_p * (b_dp - b_delta_n) * scale
                b_v = b_v * b_rstd
                b_dqw += b_ds * b_v
                b_v = b_v * (b_logit / D)
                b_v = b_qw - b_v
                b_v = (b_ds * b_rstd) * b_v
                b_v = b_p * b_do + b_v
                tl.store(
                    dres + i_l64 * stride_ln + i_n * D + o_d,
                    b_v.to(dres.dtype.element_ty),
                    mask=m_d,
                )
            tl.store(dqw + i_n * D + o_d, b_dqw, mask=m_d)
        else:
            b_delta_n = tl.load(b_delta + i_n).to(tl.float32)
            for i_l in range(L):
                i_l64 = tl.cast(i_l, tl.int64)
                g_l = L_OFFSET + i_l64
                b_rstd = tl.load(rstd + g_l * N + i_n).to(tl.float32)
                b_logit = tl.load(logit + g_l * N + i_n).to(tl.float32)
                b_p = exp(b_logit * scale - b_lse)
                b_dp = tl.zeros([], dtype=tl.float32)
                for i_d in range(tl.cdiv(D, BD)):
                    o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
                    m_d = o_d < D
                    b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
                    b_v = tl.load(
                        res + i_l64 * stride_ln + i_n * D + o_d,
                        mask=m_d,
                        other=0.0,
                    ).to(tl.float32)
                    b_dp += tl.sum(tl.where(m_d, b_v * b_do, 0.0), axis=0)
                b_ds = b_p * (b_dp - b_delta_n) * scale
                for i_d in range(tl.cdiv(D, BD)):
                    o_d = (i_d * BD + tl.arange(0, BD)).to(tl.int64)
                    m_d = o_d < D
                    b_qw = (
                        tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
                        * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
                    )
                    b_do = tl.load(do_eff + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
                    b_v = tl.load(
                        res + i_l64 * stride_ln + i_n * D + o_d,
                        mask=m_d,
                        other=0.0,
                    ).to(tl.float32)
                    b_k = b_v * b_rstd
                    b_dv = b_p * b_do + (b_ds * b_rstd) * (b_qw - b_k * (b_logit / D))
                    tl.store(
                        dres + i_l64 * stride_ln + i_n * D + o_d,
                        b_dv.to(dres.dtype.element_ty),
                        mask=m_d,
                    )
                    b_dqw = tl.load(dqw + i_n * D + o_d, mask=m_d, other=0.).to(tl.float32)
                    b_dqw += b_ds * b_k
                    tl.store(dqw + i_n * D + o_d, b_dqw, mask=m_d)
    else:
        o_n = i_n0 + tl.arange(0, BN).to(tl.int64)
        m_n = o_n < N
        b_lse = tl.load(lse + o_n, mask=m_n, other=0.).to(tl.float32)
        o_d = tl.arange(0, BD).to(tl.int64)
        m_d = o_d < D
        m_nd = m_n[:, None] & m_d[None, :]
        b_qw = (
            tl.load(q + o_d, mask=m_d, other=0.).to(tl.float32)
            * tl.load(w + o_d, mask=m_d, other=0.).to(tl.float32)
        )
        b_do = tl.load(
            do_eff + o_n[:, None] * D + o_d[None, :],
            mask=m_nd,
            other=0.,
        ).to(tl.float32)
        if FUSE_PREP:
            b_dqw = tl.load(
                o_mix + o_n[:, None] * D + o_d[None, :],
                mask=m_nd,
                other=0.,
            ).to(tl.float32)
            b_delta_n = tl.sum(tl.where(m_nd, b_do * b_dqw, 0.0), axis=1)
            if LOAD_DQW:
                b_dqw = tl.load(
                    dqw + o_n[:, None] * D + o_d[None, :],
                    mask=m_nd,
                    other=0.,
                ).to(tl.float32)
            else:
                b_dqw = b_dqw * 0.0
        else:
            b_delta_n = tl.load(b_delta + o_n, mask=m_n, other=0.).to(tl.float32)
            if LOAD_DQW:
                b_dqw = tl.load(
                    dqw + o_n[:, None] * D + o_d[None, :],
                    mask=m_nd,
                    other=0.,
                ).to(tl.float32)
            else:
                b_dqw = tl.zeros([BN, BD], dtype=tl.float32)
        for i_l in range(L):
            i_l64 = tl.cast(i_l, tl.int64)
            g_l = L_OFFSET + i_l64
            b_rstd = tl.load(rstd + g_l * N + o_n, mask=m_n, other=0.).to(tl.float32)
            b_logit = tl.load(logit + g_l * N + o_n, mask=m_n, other=0.).to(tl.float32)
            b_p = exp(b_logit * scale - b_lse)
            b_v = tl.load(
                res + i_l64 * stride_ln + o_n[:, None] * D + o_d[None, :],
                mask=m_nd,
                other=0.0,
            ).to(tl.float32)
            b_dp = tl.sum(tl.where(m_nd, b_v * b_do, 0.0), axis=1)
            b_ds = b_p * (b_dp - b_delta_n) * scale
            b_v = b_v * b_rstd[:, None]
            b_dqw += b_ds[:, None] * b_v
            b_v = b_v * (b_logit / D)[:, None]
            b_v = b_qw[None, :] - b_v
            b_v = (b_ds * b_rstd)[:, None] * b_v
            b_v = b_p[:, None] * b_do + b_v
            tl.store(
                dres + i_l64 * stride_ln + o_n[:, None] * D + o_d[None, :],
                b_v.to(dres.dtype.element_ty),
                mask=m_nd,
            )
        tl.store(dqw + o_n[:, None] * D + o_d[None, :], b_dqw, mask=m_nd)


def _get_o_mix(
    sources: Sequence[torch.Tensor],
    logit: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    device: torch.device,
    chunks: Iterable[tuple[int, int, torch.Tensor, int, int, int]] | None = None,
) -> torch.Tensor:
    o_mix = torch.zeros(*sources[0].shape, device=device, dtype=torch.float32)
    load_omix = 0
    stream = chunks if chunks is not None else _iter_l_chunks(sources)
    for l0, ll, chunk, stride_ln, N, D in stream:
        bd = _get_bd(1, D, memory_multiplier=_FWD_MEM_MULT)
        bn = _get_bn(N, bd, mem_mult=_FWD_BN_MEM_MULT)
        _launch_n(
            attnres_fwd_p2_chunk_kernel,
            N,
            BN=bn,
            res=chunk,
            logit=logit,
            lse=lse,
            o_mix=o_mix,
            L=ll,
            N=N,
            D=D,
            stride_ln=stride_ln,
            scale=scale,
            BD=bd,
            L_OFFSET=l0,
            LOAD_OMIX=load_omix,
        )
        load_omix = 1
        del chunk
    return o_mix


def _dres_chunk(
    flat_dvs: list[torch.Tensor | None],
    sources: Sequence[torch.Tensor],
    l0: int,
    ll: int,
    chunk: torch.Tensor,
) -> torch.Tensor:
    if ll == 1:
        if flat_dvs[l0] is None:
            flat_dvs[l0] = torch.empty_like(sources[l0])
        return flat_dvs[l0].unsqueeze(0)
    # views into the stacked buffer; avoid a per-layer copy_
    dres = torch.empty_like(chunk)
    for i in range(ll):
        flat_dvs[l0 + i] = dres[i]
    return dres


def fused_attnres_fwd_npu(
    q: torch.Tensor,
    sources: Sequence[torch.Tensor],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    eps: float,
    scale: float,
    checkpoint_level: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    L, N, D = len(sources), *sources[0].shape
    dtype = sources[0].dtype
    save_opre = checkpoint_level == 0

    o = torch.empty((N, D), device=sources[0].device, dtype=dtype)
    o_pre = torch.empty((N, D), device=sources[0].device, dtype=dtype) if save_opre else None
    lse = torch.empty(N, device=sources[0].device, dtype=torch.float32)
    rstd = torch.empty((L, N), device=sources[0].device, dtype=torch.float32)
    logit = torch.empty_like(rstd)

    m = torch.full((N,), float('-inf'), device=sources[0].device, dtype=torch.float32)
    acc = torch.zeros(N, device=sources[0].device, dtype=torch.float32)
    reuse_chunks = _l_chunk_size(L, N, D, sources[0].element_size()) >= L
    held: list[tuple[int, int, torch.Tensor, int, int, int]] = []
    for l0, ll, chunk, stride_ln, N, D in _iter_l_chunks(sources):
        chunk_m = torch.empty(N, device=chunk.device, dtype=torch.float32)
        chunk_acc = torch.empty(N, device=chunk.device, dtype=torch.float32)
        bd = _get_bd(1, D, memory_multiplier=_FWD_MEM_MULT)
        bn = _n_tile_bn(N, bd, D, mem_mult=_FWD_BN_MEM_MULT)
        if bn == 1:
            bl, bd = _get_bl_bd(ll, D, mem_mult=_FWD_MEM_MULT)
        else:
            bl = 1
        _launch_n(
            attnres_fwd_p1_chunk_kernel,
            N,
            BN=bn,
            q=q,
            res=chunk,
            w=w,
            rstd=rstd,
            logit=logit,
            chunk_m=chunk_m,
            chunk_acc=chunk_acc,
            L=ll,
            N=N,
            D=D,
            stride_ln=stride_ln,
            eps=eps,
            scale=scale,
            BL=bl,
            BD=bd,
            L_OFFSET=l0,
        )
        m, acc = _merge_online_softmax(m, acc, chunk_m, chunk_acc)
        del chunk_m, chunk_acc
        if reuse_chunks:
            held.append((l0, ll, chunk, stride_ln, N, D))
        else:
            del chunk
    lse.copy_(m + torch.log(acc))
    o_mix = _get_o_mix(
        sources, logit, lse, scale, sources[0].device,
        chunks=held if reuse_chunks else None,
    )

    if save_opre:
        o_pre.copy_(o_mix.to(dtype))
        if ow is None:
            o.copy_(o_pre)
    elif ow is None:
        o.copy_(o_mix.to(dtype))
    if ow is not None:
        _launch_n(
            attnres_fwd_onorm_kernel,
            N,
            o=o,
            o_mix=o_mix,
            ow=ow,
            N=N,
            D=D,
            eps=eps,
            BD=_get_bd(1, D, memory_multiplier=2.0),
        )
    stacked = held[0][2].detach() if held else None
    return o, o_pre, rstd, logit, lse, stacked


def fused_attnres_bwd_npu(
    do: torch.Tensor,
    q: torch.Tensor,
    sources: Sequence[torch.Tensor],
    w: torch.Tensor,
    ow: torch.Tensor | None,
    o_pre: torch.Tensor | None,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    eps: float,
    scale: float,
    stacked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
    has_onorm = ow is not None
    N, D = do.shape
    flat_dvs: list[torch.Tensor | None] = [None] * len(sources)

    dqw = torch.zeros_like(do, dtype=torch.float32)
    dq, dw = torch.empty_like(q), torch.empty_like(w)
    dow = torch.empty_like(ow) if has_onorm else None
    dow_partial = torch.empty_like(do, dtype=torch.float32) if has_onorm else None

    if stacked is not None:
        ll, N, D = stacked.shape
        held = [(0, ll, stacked, N * D, N, D)]
    else:
        held = None
    # ckpt=1: drop o_mix in fwd (same as CUDA) and recompute here.
    o_mix = o_pre if o_pre is not None else _get_o_mix(
        sources, logit, lse, scale, do.device, chunks=held,
    )
    bd0 = _get_bd(1, D, memory_multiplier=_BWD_DV_MEM_MULT)
    fuse_prep = (not has_onorm) and bd0 >= D
    if fuse_prep:
        do_eff = do
        b_delta = lse
    else:
        do_eff = torch.empty_like(do, dtype=torch.float32)
        b_delta = torch.empty(N, device=do.device, dtype=torch.float32)
        if dow_partial is None:
            dow_partial = do_eff
        _launch_n(
            attnres_bwd_prep_kernel,
            N,
            ow=ow if ow is not None else w,
            o_mix=o_mix,
            do=do,
            do_eff=do_eff,
            dow_partial=dow_partial,
            b_delta=b_delta,
            N=N,
            D=D,
            eps=eps,
            BD=bd0,
            HAS_ONORM=has_onorm,
        )

    load_dqw = 0
    stream = held if held is not None else _iter_l_chunks(sources)
    for l0, ll, chunk, stride_ln, N, D in stream:
        dres = _dres_chunk(flat_dvs, sources, l0, ll, chunk)
        bd = _get_bd(1, D, memory_multiplier=_BWD_DV_MEM_MULT)
        single_d = 1 if bd >= D else 0
        bn = _n_tile_bn(N, bd, D, mem_mult=_BWD_BN_MEM_MULT)
        _launch_n(
            attnres_bwd_dv_chunk_kernel,
            N,
            BN=bn,
            q=q,
            res=chunk,
            w=w,
            rstd=rstd,
            logit=logit,
            lse=lse,
            do_eff=do_eff,
            o_mix=o_mix,
            dres=dres,
            dqw=dqw,
            b_delta=b_delta,
            L=ll,
            N=N,
            D=D,
            stride_ln=stride_ln,
            scale=scale,
            BD=bd,
            SINGLE_D=single_d,
            LOAD_DQW=load_dqw,
            FUSE_PREP=1 if fuse_prep else 0,
            L_OFFSET=l0,
        )
        load_dqw = 1
        del chunk

    dqw_sum = dqw.sum(dim=0)
    dq.copy_((dqw_sum * w.float()).to(dq.dtype))
    dw.copy_((dqw_sum * q.float()).to(dw.dtype))
    if has_onorm:
        dow.copy_(dow_partial.sum(dim=0).to(dow.dtype))
    assert all(t is not None for t in flat_dvs)
    return dq, dw, dow, flat_dvs  # type: ignore[return-value]


class FusedAttnresNpuFunction(torch.autograd.Function):

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
        o, o_pre, rstd, logit, lse, stacked = fused_attnres_fwd_npu(
            query, residuals, rms_weight, output_rms_weight,
            rms_eps, scale, checkpoint_level,
        )
        if stacked is None:
            ctx.save_for_backward(
                query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, *residuals,
            )
            ctx.has_stack = False
        else:
            ctx.save_for_backward(
                query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse, stacked, *residuals,
            )
            ctx.has_stack = True
        ctx.eps = rms_eps
        ctx.scale = scale
        p = (logit * scale - lse).exp() if return_weights else o.new_empty(0)
        ctx.mark_non_differentiable(p)
        return o, p

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dp: torch.Tensor | None = None):
        del dp
        saved = ctx.saved_tensors
        query, rms_weight, output_rms_weight, o_pre, rstd, logit, lse = saved[:7]
        if ctx.has_stack:
            stacked, residuals = saved[7], saved[8:]
        else:
            stacked, residuals = None, saved[7:]
        dq, dw, dow, flat_dvs = fused_attnres_bwd_npu(
            do, query, residuals, rms_weight, output_rms_weight,
            o_pre, rstd, logit, lse, ctx.eps, ctx.scale,
            stacked=stacked,
        )
        return (dq, dw, dow, None, None, None, None, *flat_dvs)


def fused_attnres_npu(
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
    flat = tuple(_flat_residual(r, D) for r in residuals)
    o, p = FusedAttnresNpuFunction.apply(
        query, rms_weight, output_rms_weight, rms_eps, scale,
        return_weights, checkpoint_level, *flat,
    )
    o = o.view(output_shape)
    if return_weights:
        return o, p.view(len(residuals), *output_shape[:-1])
    return o


__all__ = [
    'FusedAttnresNpuFunction',
    'fused_attnres_bwd_npu',
    'fused_attnres_fwd_npu',
    'fused_attnres_npu',
]
