# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.cp.context import _check_cp_device
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


def _prune_cp_configs(configs, nargs, **kwargs):
    args = {**nargs, **kwargs}
    # wide TF32x3 states need single-stage candidates to fit shared memory
    if args['K'] <= 128 and any(config.num_stages > 1 for config in configs):
        return [config for config in configs if config.num_stages > 1]
    return configs


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'USE_G', 'USE_GK', 'USE_BG', 'IS_VARLEN', 'MULTI_SEQS'],
    prune_configs_by={'early_config_prune': _prune_cp_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_merged(
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
):
    i_col, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    if MULTI_SEQS:
        i_n = tl.program_id(2).to(tl.int64)
        hm += i_n * HV * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    is_h_part = i_col * BLOCK_SIZE < V
    # for DPLR, w/bg use H heads; GDN/KDA w uses HV heads
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_h_part:
        v += ((bos * HV + i_h) * V).to(tl.int64)
        if USE_BG:
            # for DPLR, u and v are distinct; GDN/KDA aliases u to v in the wrapper
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
                b_v_decay = tl.dot(b_w, b_h2.to(b_w.dtype), b_v_decay)
            if K > 128:
                p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v_decay = tl.dot(b_w, b_h3.to(b_w.dtype), b_v_decay)
            if K > 192:
                p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v_decay = tl.dot(b_w, b_h4.to(b_w.dtype), b_v_decay)

            p_v = v + o_t[:, None] * stride_v + o_vb[None, :]
            if USE_BG:
                b_v_orig = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0)
                p_u = u + o_t[:, None] * stride_v + o_vb[None, :]
                b_v = b_v_decay + tl.load(p_u, mask=m_t[:, None] & m_vb[None, :], other=0.0)
            else:
                b_v = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
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
                o_k1 = tl.arange(0, 64)
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                b_h1 *= exp2(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    b_h2 *= exp2(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    b_h3 *= exp2(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
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
                b_h1 = tl.dot(b_k, b_v, b_h1)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 = tl.dot(b_k, b_v, b_h2)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 = tl.dot(b_k, b_v, b_h3)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 = tl.dot(b_k, b_v, b_h4)

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
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            if USE_BG:
                # use bg for the DPLR transition matrix
                p_k = bg + o_t[:, None] * stride_k + row[None, :]
            else:
                p_k = k + o_t[:, None] * stride_k + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
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
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32), input_precision="tf32x3")

        stride_hm_kv = K + V
        p_m = hm + V + row[:, None] * stride_hm_kv + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


@triton.jit
def chunk_cp_merge_state(
    hm,
    b_h,
    o_k,
    o_v,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    m_k, m_v = o_k < K, o_v < V
    p_he = hm + o_k[:, None] * (V + K) + o_v[None, :]
    b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.).to(tl.float32)
    if BK <= 128:
        p_m = hm + V + o_k[:, None] * (V + K) + o_k[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.).to(tl.float32)
        if STATE_V_FIRST:
            b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m), input_precision="tf32x3")
        else:
            b_h = tl.dot(b_m, b_h.to(tl.float32), input_precision="tf32x3")
    else:
        # tile wide reductions to keep TF32x3 operands within shared memory
        b_acc = tl.full(b_h.shape, 0, tl.float32)
        for i in tl.static_range(BK // 128):
            o_r = i * 128 + tl.arange(0, 128)
            p_m = hm + V + o_k[:, None] * (V + K) + o_r[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & (o_r < K)[None, :], other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_hi = tl.gather(b_h, tl.broadcast_to(o_r[None, :], (BV, 128)), axis=1)
                b_acc = tl.dot(b_hi.to(tl.float32), tl.trans(b_m), b_acc, input_precision="tf32x3")
            else:
                b_hi = tl.gather(b_h, tl.broadcast_to(o_r[:, None], (128, BV)), axis=0)
                b_acc = tl.dot(b_m, b_hi.to(tl.float32), b_acc, input_precision="tf32x3")
        b_h = b_acc
    return b_h + (tl.trans(b_he) if STATE_V_FIRST else b_he)


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
    'HAS_H_SEQ_IDX': lambda args: args['h_seq_idx'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V', 'STATE_V_FIRST', 'INTRACARD_MODE'],
    prune_configs_by={'early_config_prune': _prune_cp_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def merge_fwd_bwd_kernel(
    h,                   # [HV, K, V] or [num_non_first, HV, K, V]; K/V swapped for STATE_V_FIRST
    ag_hm,               # [num_ranks, HV, K, V+K] or [S_split, HV, K, V+K]
    pre_or_post_num_ranks,
    rank,
    seq_offsets,         # [num_split_seqs + 1] or None
    init_offsets,        # [num_split_seqs + 1] or None
    h0_seq_ids,          # [num_split_seqs] or None
    h0,                  # [N_orig, HV, K, V] or None; K/V swapped for STATE_V_FIRST
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
    NUM_RANKS_ON_DEVICE: tl.constexpr = False,
):
    """Merge affine summaries across ranks or intra-card subsequences.

    Summaries use [K, V+K]; STATE_V_FIRST selects [V, K] input and output states.
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

        ss_start = tl.load(seq_offsets + i_seq).to(tl.int32)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int32)
        init_base = tl.load(init_offsets + i_seq).to(tl.int32)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int32)
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

            b_h = chunk_cp_merge_state(
                hm=ag_hm + base,
                b_h=b_h,
                o_k=o_k,
                o_v=o_v,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                STATE_V_FIRST=STATE_V_FIRST,
            )

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
            num_ranks = tl.load(pre_or_post_num_ranks).to(tl.int32)
        else:
            num_ranks = pre_or_post_num_ranks.to(tl.int32)
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
            b_h = chunk_cp_merge_state(
                hm=ag_hm + cur_rank * stride,
                b_h=b_h,
                o_k=o_k,
                o_v=o_v,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                STATE_V_FIRST=STATE_V_FIRST,
            )
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
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2, 1] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'USE_G', 'USE_GK', 'USE_BG', 'IS_VARLEN'],
    prune_configs_by={'early_config_prune': _prune_cp_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_merged(
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
):
    """Compute backward affine summaries for context-parallel state propagation."""
    i_col, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    is_dh_part = i_col * BLOCK_SIZE < V

    # for DPLR, w uses H heads; GDN/KDA w uses HV heads
    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
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
                bg_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)

            p_dv = dv + o_t[:, None] * stride_v + o_vb[None, :]
            p_do = do + o_t[:, None] * stride_v + o_vb[None, :]
            b_do = tl.load(p_do, mask=m_t[:, None] & m_vb[None, :], other=0.0)

            p_k = k + o_t[:, None] * stride_qk + o_k1[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = k + o_t[:, None] * stride_qk + o_k2[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                if USE_GK:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), b_dv)

            if K > 128:
                p_k = k + o_t[:, None] * stride_qk + o_k3[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                if USE_GK:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), b_dv)

            if K > 192:
                p_k = k + o_t[:, None] * stride_qk + o_k4[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                if USE_GK:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), b_dv)

            if USE_G:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, mask=m_t[:, None] & m_vb[None, :], other=0.0)

            p_w = w + o_k1[:, None] + o_t[None, :] * stride_w
            p_q = q + o_k1[:, None] + o_t[None, :] * stride_qk
            b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh1 *= exp2(b_gk_last1[:, None])
            if USE_BG:
                # scale is already applied upstream for DPLR
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
            else:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = q + o_k2[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k2[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                if USE_BG:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = q + o_k3[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k3[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                if USE_BG:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = q + o_k4[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k4[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                if USE_BG:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        p_dh1 = dhm + o_k1[:, None] * (V + K) + o_vb[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_dh2 = dhm + o_k2[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_dh3 = dhm + o_k3[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_dh4 = dhm + o_k4[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

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
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
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
            # keep the affine chain in fp32 to avoid repeated bf16 rounding
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32), input_precision="tf32x3")

        p_m = dhm + V + row[:, None] * (V + K) + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


def chunk_gated_delta_rule_fwd_h_pre_process(
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
    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    _check_cp_device(k.device)
    rank = dist.get_rank(group=context.group)

    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if context.layout == 'zigzag':
        if use_graph:
            raise NotImplementedError("use_graph is not supported with zigzag CP layout")
        fns = context.front_num_seqs
        hm = k.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid_hm = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
        # each part exports its last segment unless it is chain-last
        for part, (cu_win, is_last) in enumerate(zip(
            (cu_seqlens[fns - 1: fns + 1], cu_seqlens[-2:]),
            context.is_last_by_part,
        )):
            if not is_last:
                pre_process_fwd_kernel_merged[grid_hm](
                    k=k,
                    v=u if v is None else v,
                    w=w,
                    g=g,
                    gk=gk,
                    bg=bg,
                    u=u,
                    hm=hm[part],
                    cu_seqlens=cu_win,
                    T=T,
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    BT=BT,
                    BK1=BK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    MULTI_SEQS=False,
                )
        ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
        # chain order: rank fronts in ascending order, then rank backs in descending order
        slots = torch.cat([ag_hm[:, 0], ag_hm[:, 1].flip(0)])
        world_size = dist.get_world_size(context.group)

        def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
        for part in range(2):
            if not context.is_first_by_part[part]:
                merge_fwd_bwd_kernel[grid](
                    h=initial_state[0] if part == 0 else initial_state[fns],
                    ag_hm=slots,
                    pre_or_post_num_ranks=context.pre_num_ranks_by_part[part],
                    rank=rank if part == 0 else 2 * world_size - 1 - rank,
                    seq_offsets=None,
                    init_offsets=None,
                    h0_seq_ids=None,
                    h0=None,
                    h_seq_idx=None,
                    HV=HV,
                    K=K,
                    V=V,
                    BK=BK,
                    FORWARD=True,
                    INTRACARD_MODE=False,
                    NUM_SEQ_ENTRIES=0,
                    STATE_V_FIRST=state_v_first,
                    NUM_RANKS_ON_DEVICE=False,
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
        # zero-length tail padding must not select a padding sequence
        i_last = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1
        cu_last = cu_seqlens.index_select(0, torch.stack((i_last, i_last + 1)))
    # graph replay freezes host flags; every rank must still join the all-gather
    if use_graph or not context.is_last_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
        # for DPLR, v holds original values; u holds WY-processed values for the recurrent update
        pre_process_fwd_kernel_merged[grid](
            k=k,
            v=u if v is None else v,
            w=w,
            g=g,
            gk=gk,
            bg=bg,
            u=u,
            hm=hm,
            cu_seqlens=cu_last,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            BLOCK_SIZE=BLOCK_SIZE,
            MULTI_SEQS=False,
        )
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    # a zero-rank merge would only re-store the existing zeros
    if use_graph or not context.is_first_rank:
        def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel[grid](
            h=initial_state[0],
            ag_hm=ag_hm,
            pre_or_post_num_ranks=context.pre_num_ranks_dev if use_graph else context.pre_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            h_seq_idx=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            FORWARD=True,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
            NUM_RANKS_ON_DEVICE=use_graph,
        )
    return initial_state


def chunk_gated_delta_rule_bwd_dhu_pre_process(
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
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    _check_cp_device(k.device)
    rank = dist.get_rank(context.group)

    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    if context.layout == 'zigzag':
        if use_graph:
            raise NotImplementedError("use_graph is not supported with zigzag CP layout")
        fns = context.front_num_seqs
        dhm = q.new_zeros(2, HV, K, V + K, dtype=torch.float32)
        if state_v_first:
            dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid_hm = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
        # each part exports its first segment unless it is chain-first
        for part, (cu_win, is_first) in enumerate(zip((cu_seqlens[:2], cu_seqlens[fns: fns + 2]), context.is_first_by_part)):
            if not is_first:
                pre_process_bwd_kernel_merged[grid_hm](
                    q=q,
                    k=k if bg is None else bg,
                    w=w,
                    g=g,
                    gk=gk,
                    do=do,
                    dhm=dhm[part],
                    dv=dv,
                    cu_seqlens=cu_win,
                    scale=scale,
                    T=T,
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    BT=BT,
                    BK1=BK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    USE_BG=bg is not None,
                )
        ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
        # chain order: rank fronts in ascending order, then rank backs in descending order
        slots = torch.cat([ag_dhm[:, 0], ag_dhm[:, 1].flip(0)])
        world_size = dist.get_world_size(context.group)

        def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
        for part in range(2):
            if not context.is_last_by_part[part]:
                merge_fwd_bwd_kernel[grid](
                    h=dht[fns - 1] if part == 0 else dht[-1],
                    ag_hm=slots,
                    pre_or_post_num_ranks=context.post_num_ranks_by_part[part],
                    rank=rank if part == 0 else 2 * world_size - 1 - rank,
                    seq_offsets=None,
                    init_offsets=None,
                    h0_seq_ids=None,
                    h0=None,
                    h_seq_idx=None,
                    HV=HV,
                    K=K,
                    V=V,
                    BK=BK,
                    FORWARD=False,
                    INTRACARD_MODE=False,
                    NUM_SEQ_ENTRIES=0,
                    STATE_V_FIRST=state_v_first,
                    NUM_RANKS_ON_DEVICE=False,
                )
        return dht, None

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    # graph replay freezes host flags; every rank must still join the all-gather
    if use_graph or not context.is_first_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
        pre_process_bwd_kernel_merged[grid](
            q=q,
            k=k if bg is None else bg,
            w=w,
            g=g,
            gk=gk,
            do=do,
            dhm=dhm,
            dv=dv,
            cu_seqlens=cu_seqlens[:2],
            scale=scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_BG=bg is not None,
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    # a zero-rank merge would only re-store the existing zeros
    if use_graph or not context.is_last_rank:
        h_seq_idx = None
        h = dht[-1]
        if use_graph:
            assert context.post_num_ranks_dev is not None, "use_graph with CP requires context.post_num_ranks_dev"
            # zero-length tail padding must not select a padding row
            h_seq_idx = (cu_seqlens[1:] > cu_seqlens[:-1]).sum() - 1
            h = dht

        def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel[grid](
            h=h,
            ag_hm=ag_dhm,
            pre_or_post_num_ranks=context.post_num_ranks_dev if use_graph else context.post_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            h_seq_idx=h_seq_idx,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
            NUM_RANKS_ON_DEVICE=use_graph,
        )

    return dht, None


def compress_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    if context.layout == 'zigzag':
        if len(context.cu_seqlens) == 3:
            # both parts can carry a cross-rank state
            return h0
        # only the first segment of each part can carry a cross-rank state
        return h0[[0, context.front_num_seqs]]
    # clone to avoid retaining the full state tensor for backward
    return h0[:1].clone()


def expand_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    B = len(context.cu_seqlens) - 1
    if context.layout == 'zigzag':
        if B == 2:
            return h0
        expand_h0 = h0.new_zeros(B, *h0.shape[1:])
        expand_h0[[0, context.front_num_seqs]] = h0
        return expand_h0
    expand_h0 = h0.new_zeros(B, *h0.shape[1:])
    expand_h0[:1] = h0
    return expand_h0
