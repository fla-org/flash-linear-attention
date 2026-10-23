# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CP chunk_delta_h pre-process kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    get_ub_manager,
    max_grid_axis_chunks,
)

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext

# peak live fp32: b_h[64,BV] + b_w[64,64] + b_v[64,BV] + b_k[64,64] during recurrence.
_PREPROCESS_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BLOCK = 16
# larger tiles amortize the per-iteration scalar/address-gen cost of the
# sequential chunk recurrence (these kernels are scalar-bound on NPU).
# 64 is the UB-safe ceiling for D128 (mem_mult model); D64 caps at pow2(V)=64.
_MAX_BLOCK = 64


def _m_tiles_per_program(K: int, block_size: int) -> int:
    """Pair K-column tiles so one program forms k^T w once.

    K=256 keeps a single tile: a second [256, 64] fp32 slab does not fit with the KxK factor.
    """
    nm = triton.cdiv(K, block_size)
    if triton.next_power_of_2(K) <= 128 and nm >= 2 and nm % 2 == 0:
        return 2
    return 1


def _h_block_size(K: int, V: int) -> int:
    """V tile for the h recurrence. One 128-wide tile covers D=128 when two K slabs fit."""
    narrow = _get_block_size(K, V)
    wide = 128
    if wide > triton.next_power_of_2(V) or wide <= narrow:
        return narrow
    strips = triton.cdiv(min(K, 256), 64)
    # fp32 slabs: strips of [64, wide], plus one w, v, and k tile.
    peak = strips * 64 * wide * 4 + 64 * 64 * 4 + 64 * wide * 4 + 64 * 64 * 4
    if peak <= int(get_ub_manager().ub_capacity_bytes * _SAFETY_MARGIN):
        return wide
    return narrow


def _get_block_size(K: int, V: int) -> int:
    """Compute UB-safe block size for pre-process kernels."""
    return compute_row_tile_block_size(
        min(K, 64),
        V,
        _PREPROCESS_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BLOCK,
        min_block=16,
        max_block=min(_MAX_BLOCK, triton.next_power_of_2(V)),
    )


def _contig_gate(g: torch.Tensor | None, hv: int) -> torch.Tensor | None:
    """Transpose g to [B, HV, T] so the NPU kernels load stride-1 along T."""
    if g is not None and hv > 1:
        return g.transpose(1, 2).contiguous()
    return g


def _launch_preprocess_kernel(kernel, *, nv_chunks: int, nh_total: int, kernel_kwargs: dict) -> None:
    """Launch pre-process kernel with grid-axis chunking for Ascend."""
    max_nv = max_grid_axis_chunks(nv_chunks, nh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for v_off in range(0, nv_chunks, max_nv):
        v_len = min(max_nv, nv_chunks - v_off)
        kernel_kwargs['V_OFFSET'] = v_off
        max_nh = max_grid_axis_chunks(nh_total, v_len, max_grid=ASCEND_MAX_GRID_DIM)
        for nh_off in range(0, nh_total, max_nh):
            nh_len = min(max_nh, nh_total - nh_off)
            kernel_kwargs['NH_OFFSET'] = nh_off
            kernel[(v_len, nh_len)](**kernel_kwargs)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_merged_npu(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
    V_OFFSET: tl.constexpr = 0,
    NH_OFFSET: tl.constexpr = 0,
    M_GROUP: tl.constexpr = 1,
    # 0: h and m share one grid. 1: h tiles only. 2: m tiles only.
    PART: tl.constexpr = 0,
    AFFINE_CHAIN_PRECISION: tl.constexpr = None,
):
    """Forward pre-process kernel for CP on NPU.

    Computes h (K x V) and m (K x K) matrices for cross-rank state passing.
    Grid: (V/BLOCK_SIZE + K-groups, HV) with host-side chunking via V_OFFSET/NH_OFFSET.
    """
    i_col = (tl.program_id(0) + V_OFFSET).to(tl.int64)
    i_h = (tl.program_id(1) + NH_OFFSET).to(tl.int64)

    if MULTI_SEQS:
        i_n = tl.program_id(2).to(tl.int64)
        hm += i_n * HV * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)

    # save the packed length before varlen overwrites T (used for g strides).
    T_seq = T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    if PART == 1:
        is_h_part = True
    elif PART == 2:
        is_h_part = False
    else:
        is_h_part = i_col * BLOCK_SIZE < V

    # USE_BG shares head dim H with k; otherwise w uses HV, same as v.
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        # g is host-transposed to [B, HV, T] (G_T_CONTIG): stride-1 loads along T.
        if IS_VARLEN:
            g += (bos + i_h.to(tl.int64) * T_seq).to(tl.int64)
        else:
            g += ((i_n * HV + i_h).to(tl.int64) * T_seq)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_h_part:
        v += ((bos * HV + i_h) * V).to(tl.int64)
        if USE_BG:
            u += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_v = v + o_t[:, None] * stride_v + o_vb[None, :]
            if USE_BG:
                b_v_orig = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0)
                p_u = u + o_t[:, None] * stride_v + o_vb[None, :]
                b_v = b_v_decay + tl.load(p_u, mask=m_t[:, None] & m_vb[None, :], other=0.0)
            else:
                b_v = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                b_g = tl.load(g + o_t, mask=m_t, other=0.0).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                b_h1 *= exp2(b_gk_last1)[:, None]
                if K > 64:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    b_h2 *= exp2(b_gk_last2)[:, None]
                if K > 128:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    b_h3 *= exp2(b_gk_last3)[:, None]
                if K > 192:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    b_h4 *= exp2(b_gk_last4)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            if USE_BG:
                p_bg = bg + o_k1[:, None] + o_t[None, :] * stride_k
                b_bg = tl.load(p_bg, mask=m_k1[:, None] & m_t[None, :], other=0.0)
                b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k2[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k3[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k4[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            else:
                b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 += tl.dot(b_k, b_v)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 += tl.dot(b_k, b_v)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 += tl.dot(b_k, b_v)

        stride_hm_kv = K + V
        p_h1 = hm + o_k1[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_h2 = hm + o_k2[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_h3 = hm + o_k3[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_h4 = hm + o_k4[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        # columns past the V tiles index the K block of m.
        # Adjacent tiles share one k^T w; a tile used to rebuild that KxK product.
        if PART == 2:
            i_k_group = i_col
        else:
            i_k_group = i_col - tl.cdiv(V, BLOCK_SIZE)
        i_k_col = i_k_group * M_GROUP

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE
        col2 = col + BLOCK_SIZE

        b_m = tl.zeros([BK1, BLOCK_SIZE], dtype=tl.float32)
        b_m += tl.where(row[:, None] == col[None, :], 1.0, 0.0)
        if M_GROUP > 1:
            b_m2 = tl.zeros([BK1, BLOCK_SIZE], dtype=tl.float32)
            b_m2 += tl.where(row[:, None] == col2[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            if USE_BG:
                p_k = bg + o_t[:, None] * stride_k + row[None, :]
            else:
                p_k = k + o_t[:, None] * stride_k + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                b_g = tl.load(g + o_t, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            if USE_BG:
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag + b_kw
            else:
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag - b_kw
            # += keeps the accumulator in one address space on NPU.
            b_m += tl.dot(
                b_m_i.to(tl.float32), b_m.to(tl.float32), input_precision=AFFINE_CHAIN_PRECISION,
            ) - b_m
            if M_GROUP > 1:
                b_m2 += tl.dot(
                    b_m_i.to(tl.float32), b_m2.to(tl.float32), input_precision=AFFINE_CHAIN_PRECISION,
                ) - b_m2

        stride_hm_kv = K + V
        p_m = hm + V + row[:, None] * stride_hm_kv + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])
        if M_GROUP > 1:
            p_m2 = hm + V + row[:, None] * stride_hm_kv + col2[None, :]
            tl.store(p_m2, b_m2.to(p_m2.dtype.element_ty), mask=(row < K)[:, None] & (col2 < K)[None, :])


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
    'HAS_H_SEQ_IDX': lambda args: args['h_seq_idx'] is not None,
})
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def merge_fwd_bwd_kernel_npu(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    h_seq_idx,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    INTRACARD_MODE: tl.constexpr,
    NUM_SEQ_ENTRIES,
    HAS_H0: tl.constexpr,
    HAS_H_SEQ_IDX: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
    AFFINE_CHAIN_PRECISION: tl.constexpr = None,
    NUM_RANKS_ON_DEVICE: tl.constexpr = False,
):
    """Merge kernel for CP on NPU.

    Merges h/m matrices across ranks or sub-sequences.
    Grid: (V/BV, HV) for CP mode, (V/BV, NUM_SEQ_ENTRIES, HV) for intracard mode.
    """
    i_v = tl.program_id(0).to(tl.int64)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if INTRACARD_MODE:
        i_seq = tl.program_id(1).to(tl.int64)
        i_h = tl.program_id(2).to(tl.int64)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        ss_start = tl.load(seq_offsets + i_seq).to(tl.int64)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int64)
        init_base = tl.load(init_offsets + i_seq).to(tl.int64)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int64)
            if STATE_V_FIRST:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            else:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                b_h = tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        else:
            if STATE_V_FIRST:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            p_he = ag_hm + base + o_k[:, None] * (V + K) + o_v[None, :]
            b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            p_m = ag_hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                # a new accumulator keeps b_h in one address space on NPU.
                b_h_new = tl.dot(
                    b_h.to(tl.float32), tl.trans(b_m), input_precision=AFFINE_CHAIN_PRECISION,
                ) + tl.trans(b_he)
                b_h = b_h_new
            else:
                b_h_new = tl.dot(
                    b_m.to(tl.float32), b_h.to(tl.float32), input_precision=AFFINE_CHAIN_PRECISION,
                ) + b_he.to(tl.float32)
                b_h = b_h_new

            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = HV * K * V
                if STATE_V_FIRST:
                    p_out = h + init_idx * stride_init + i_h * V * K + o_v[:, None] * K + o_k[None, :]
                    m_out = m_v[:, None] & m_k[None, :]
                else:
                    p_out = h + init_idx * stride_init + i_h * K * V + o_k[:, None] * V + o_v[None, :]
                    m_out = m_k[:, None] & m_v[None, :]
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=m_out)
    else:
        i_h = tl.program_id(1).to(tl.int64)
        if HAS_H_SEQ_IDX:
            h += tl.load(h_seq_idx).to(tl.int64) * HV * K * V
        if NUM_RANKS_ON_DEVICE:
            num_ranks = tl.load(pre_or_post_num_ranks).to(tl.int64)
        else:
            num_ranks = pre_or_post_num_ranks.to(tl.int64)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = HV * K * (K + V)
        if STATE_V_FIRST:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = ag_hm + cur_rank * stride + o_k[:, None] * (K + V) + o_v[None, :]
            b_ag_h = tl.load(p_ag_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
            p_ag_m = ag_hm + cur_rank * stride + V + o_k[:, None] * (K + V) + o_k[None, :]
            b_ag_m = tl.load(p_ag_m, mask=m_k[:, None] & m_k[None, :], other=0.0)
            if STATE_V_FIRST:
                # a new accumulator keeps b_h in one address space on NPU.
                b_h_new = tl.dot(
                    b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32), input_precision=AFFINE_CHAIN_PRECISION,
                ) + tl.trans(b_ag_h).to(tl.float32)
                b_h = b_h_new
            else:
                b_h_new = tl.dot(
                    b_ag_m.to(tl.float32), b_h.to(tl.float32), input_precision=AFFINE_CHAIN_PRECISION,
                ) + b_ag_h.to(tl.float32)
                b_h = b_h_new
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_merged_npu(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    V_OFFSET: tl.constexpr = 0,
    NH_OFFSET: tl.constexpr = 0,
    AFFINE_CHAIN_PRECISION: tl.constexpr = None,
):
    """Backward pre-process kernel for CP on NPU.

    Computes dh (K x V) and dm (K x K) matrices for cross-rank gradient passing.
    Grid: (V/BLOCK_SIZE + K/BLOCK_SIZE, HV) with host-side chunking via V_OFFSET/NH_OFFSET.
    """
    i_col = (tl.program_id(0) + V_OFFSET).to(tl.int64)
    i_h = (tl.program_id(1) + NH_OFFSET).to(tl.int64)
    i_n = 0

    # save the packed length before varlen overwrites T (used for g strides).
    T_seq = T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    is_dh_part = i_col * BLOCK_SIZE < V

    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        # g is host-transposed to [B, HV, T] (G_T_CONTIG): stride-1 loads along T.
        # bwd always has i_n == 0, so bos already encodes the batch offset.
        g += (bos + i_h.to(tl.int64) * T_seq).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_dh_part:
        do += ((bos * HV + i_h) * V).to(tl.int64)
        dv += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T

            if USE_G:
                bg_last = tl.load(g + last_idx).to(tl.float32)
                b_g = tl.load(g + o_t, mask=m_t, other=0.0).to(tl.float32)
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)

            p_dv = dv + o_t[:, None] * stride_v + o_vb[None, :]
            p_do = do + o_t[:, None] * stride_v + o_vb[None, :]
            b_do = tl.load(p_do, mask=m_t[:, None] & m_vb[None, :], other=0.0)
            # fold gate*exp and scale into do once so K-slabs skip per-slab q gating.
            # dot(q * g_exp, do) * scale == dot(q, do * g_exp * scale).
            if USE_G:
                if USE_BG:
                    b_do = b_do * b_g_exp[:, None]
                else:
                    b_do = b_do * (b_g_exp * scale)[:, None]
            elif not USE_BG:
                b_do = b_do * scale

            p_k = k + o_t[:, None] * stride_qk + o_k1[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=m_k1, other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = k + o_t[:, None] * stride_qk + o_k2[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                if USE_GK:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=m_k2, other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = k + o_t[:, None] * stride_qk + o_k3[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                if USE_GK:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=m_k3, other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = k + o_t[:, None] * stride_qk + o_k4[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                if USE_GK:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=m_k4, other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, mask=m_t[:, None] & m_vb[None, :], other=0.0)

            p_w = w + o_k1[:, None] + o_t[None, :] * stride_w
            p_q = q + o_k1[:, None] + o_t[None, :] * stride_qk
            b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh1 *= bg_last_exp
            if USE_GK:
                b_dh1 *= exp2(b_gk_last1[:, None])
            if USE_BG:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
            else:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = q + o_k2[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k2[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh2 *= bg_last_exp
                if USE_GK:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                if USE_BG:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = q + o_k3[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k3[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh3 *= bg_last_exp
                if USE_GK:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                if USE_BG:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = q + o_k4[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k4[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh4 *= bg_last_exp
                if USE_GK:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                if USE_BG:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

        stride_dhm = V + K
        p_dh1 = dhm + o_k1[:, None] * stride_dhm + o_vb[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_dh2 = dhm + o_k2[:, None] * stride_dhm + o_vb[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_dh3 = dhm + o_k3[:, None] * stride_dhm + o_vb[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_dh4 = dhm + o_k4[:, None] * stride_dhm + o_vb[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        # columns past the V tiles index the K block of m.
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        b_m = tl.zeros([BK1, BLOCK_SIZE], dtype=tl.float32)
        b_m += tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            i_t = NT - 1 - _i_t
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T

            p_k = k + o_t[:, None] * stride_qk + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                b_g = tl.load(g + o_t, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            if USE_BG:
                b_m_i = b_diag + b_kw
            else:
                b_m_i = b_diag - b_kw
            # += keeps the accumulator in one address space on NPU.
            b_m += tl.dot(
                b_m_i.to(tl.float32), b_m.to(tl.float32), input_precision=AFFINE_CHAIN_PRECISION,
            ) - b_m

        p_m = dhm + V + row[:, None] * (V + K) + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


@input_guard
def chunk_gated_delta_rule_fwd_h_pre_process_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pre-process for CP on NPU."""
    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    # Ascend affine chains stay in IEEE FP32, matching CP_DOT_PRECISION.
    affine_precision = "ieee"
    rank = dist.get_rank(group=context.group)

    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    g = _contig_gate(g, HV)
    block_size = _get_block_size(K, V)

    def launch_fwd(hm_part, cu_win):
        h_block = _h_block_size(K, V)
        base = {
            'k': k,
            'v': u if v is None else v,
            'w': w,
            'g': g,
            'gk': gk,
            'bg': bg,
            'u': u,
            'hm': hm_part,
            'cu_seqlens': cu_win,
            'T': T,
            'H': H,
            'HV': HV,
            'K': K,
            'V': V,
            'BT': BT,
            'BK1': BK,
            'MULTI_SEQS': False,
            'V_OFFSET': 0,
            'NH_OFFSET': 0,
            'AFFINE_CHAIN_PRECISION': affine_precision,
        }
        m_group = _m_tiles_per_program(K, block_size)
        if h_block == block_size:
            nv_chunks = triton.cdiv(V, block_size) + triton.cdiv(triton.cdiv(K, block_size), m_group)
            _launch_preprocess_kernel(
                pre_process_fwd_kernel_merged_npu,
                nv_chunks=nv_chunks,
                nh_total=HV,
                kernel_kwargs={**base, 'BLOCK_SIZE': block_size, 'M_GROUP': m_group, 'PART': 0},
            )
            return
        _launch_preprocess_kernel(
            pre_process_fwd_kernel_merged_npu,
            nv_chunks=triton.cdiv(V, h_block),
            nh_total=HV,
            kernel_kwargs={**base, 'BLOCK_SIZE': h_block, 'M_GROUP': 1, 'PART': 1},
        )
        _launch_preprocess_kernel(
            pre_process_fwd_kernel_merged_npu,
            nv_chunks=triton.cdiv(triton.cdiv(K, block_size), m_group),
            nh_total=HV,
            kernel_kwargs={**base, 'BLOCK_SIZE': block_size, 'M_GROUP': m_group, 'PART': 2},
        )

    def merge_fwd(h, ag, num_ranks, merge_rank, num_ranks_on_device):
        def grid(meta):
            return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel_npu[grid](
            h=h,
            ag_hm=ag,
            pre_or_post_num_ranks=num_ranks,
            rank=merge_rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            h_seq_idx=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=block_size,
            FORWARD=True,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
            AFFINE_CHAIN_PRECISION=affine_precision,
            NUM_RANKS_ON_DEVICE=num_ranks_on_device,
        )

    if context.layout == 'zigzag':
        if use_graph:
            raise NotImplementedError("use_graph is not supported with zigzag CP layout")
        fns = context.front_num_seqs
        hm = k.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
        # each part exports the affine chain of its last segment, skipped when chain-last
        for part, (cu_win, is_last) in enumerate(zip(
                (cu_seqlens[fns - 1: fns + 1], cu_seqlens[-2:]), context.is_last_by_part)):
            if not is_last:
                launch_fwd(hm[part], cu_win)
        ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
        # chain order: front parts of ranks 0..W-1, then back parts of ranks W-1..0
        slots = torch.cat([ag_hm[:, 0], ag_hm[:, 1].flip(0)])
        world_size = dist.get_world_size(context.group)
        for part in range(2):
            if not context.is_first_by_part[part]:
                merge_fwd(
                    initial_state[0] if part == 0 else initial_state[fns],
                    slots,
                    context.pre_num_ranks_by_part[part],
                    rank if part == 0 else 2 * world_size - 1 - rank,
                    False,
                )
        return initial_state

    hm = k.new_zeros(HV, K, (V + K), dtype=torch.float32)
    if state_v_first:
        initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
    cu_last = cu_seqlens[-2:]
    if use_graph:
        assert context.pre_num_ranks_dev is not None, "use_graph with CP requires context.pre_num_ranks_dev"
        # last non-empty sequence window, on-device; zero-length tail padding would
        # otherwise make cu_seqlens[-2:] point at a padding sequence
        i_last = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1
        cu_last = cu_seqlens.index_select(0, torch.stack((i_last, i_last + 1)))
    # graph mode always launches: a last rank's hm is never read, but the all-gather
    # needs every rank, and host-side flags are frozen at capture
    if use_graph or not context.is_last_rank:
        launch_fwd(hm, cu_last)
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    # a zero num_ranks merge just re-stores the zeros initial_state[0] already holds
    if use_graph or not context.is_first_rank:
        merge_fwd(
            initial_state[0],
            ag_hm,
            context.pre_num_ranks_dev if use_graph else context.pre_num_ranks,
            rank,
            use_graph,
        )
    return initial_state


@input_guard
def chunk_gated_delta_rule_bwd_dhu_pre_process_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    chunk_size: int = 64,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pre-process for CP on NPU."""
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    # Ascend affine chains stay in IEEE FP32, matching CP_DOT_PRECISION.
    affine_precision = "ieee"
    rank = dist.get_rank(context.group)

    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    g = _contig_gate(g, HV)
    block_size = _get_block_size(K, V)

    def launch_bwd(dhm_part, cu_win):
        nv_chunks = triton.cdiv(V, block_size) + triton.cdiv(K, block_size)
        _launch_preprocess_kernel(
            pre_process_bwd_kernel_merged_npu,
            nv_chunks=nv_chunks,
            nh_total=HV,
            kernel_kwargs={
                'q': q,
                'k': k if bg is None else bg,
                'w': w,
                'g': g,
                'gk': gk,
                'do': do,
                'dhm': dhm_part,
                'dv': dv,
                'cu_seqlens': cu_win,
                'scale': scale,
                'T': T,
                'H': H,
                'HV': HV,
                'K': K,
                'V': V,
                'BT': BT,
                'BK1': BK,
                'BLOCK_SIZE': block_size,
                'USE_BG': bg is not None,
                'V_OFFSET': 0,
                'NH_OFFSET': 0,
                'AFFINE_CHAIN_PRECISION': affine_precision,
            },
        )

    def merge_bwd(h, ag, num_ranks, merge_rank, h_seq_idx, num_ranks_on_device):
        def grid(meta):
            return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel_npu[grid](
            h=h,
            ag_hm=ag,
            pre_or_post_num_ranks=num_ranks,
            rank=merge_rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            h_seq_idx=h_seq_idx,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=block_size,
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
            AFFINE_CHAIN_PRECISION=affine_precision,
            NUM_RANKS_ON_DEVICE=num_ranks_on_device,
        )

    if context.layout == 'zigzag':
        if use_graph:
            raise NotImplementedError("use_graph is not supported with zigzag CP layout")
        fns = context.front_num_seqs
        dhm = q.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)
        # each part exports the backward affine chain of its first segment, skipped when chain-first
        for part, (cu_win, is_first) in enumerate(zip(
                (cu_seqlens[:2], cu_seqlens[fns: fns + 2]), context.is_first_by_part)):
            if not is_first:
                launch_bwd(dhm[part], cu_win)
        ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
        # chain order: front parts of ranks 0..W-1, then back parts of ranks W-1..0
        slots = torch.cat([ag_dhm[:, 0], ag_dhm[:, 1].flip(0)])
        world_size = dist.get_world_size(context.group)
        for part in range(2):
            if not context.is_last_by_part[part]:
                merge_bwd(
                    dht[fns - 1] if part == 0 else dht[-1],
                    slots,
                    context.post_num_ranks_by_part[part],
                    rank if part == 0 else 2 * world_size - 1 - rank,
                    None,
                    False,
                )
        return dht, None

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    # graph mode always launches: a first rank's dhm is never read, but the all-gather
    # needs every rank, and host-side flags are frozen at capture
    if use_graph or not context.is_first_rank:
        launch_bwd(dhm, cu_seqlens[:2])

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    # a zero num_ranks merge just re-stores the zeros dht already holds
    if use_graph or not context.is_last_rank:
        h_seq_idx = None
        h = dht[-1]
        if use_graph:
            assert context.post_num_ranks_dev is not None, "use_graph with CP requires context.post_num_ranks_dev"
            # merge target is the last non-empty sequence; zero-length tail padding
            # would otherwise make dht[-1] a padding row
            h_seq_idx = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1
            h = dht
        merge_bwd(
            h,
            ag_dhm,
            context.post_num_ranks_dev if use_graph else context.post_num_ranks,
            rank,
            h_seq_idx,
            use_graph,
        )

    return dht, None
