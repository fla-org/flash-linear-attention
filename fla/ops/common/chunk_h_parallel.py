# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Fully parallelized state passing.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV']
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_parallel(
    k,
    v,
    h,
    g,
    gk,
    gv,
    h0,
    ht,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NV = tl.cdiv(V, BV)
    # i_b: batch index
    # i_h: head index
    # i_n: sequence index
    # i_t: chunk index within current sequence
    # i_tg: (global) chunk index across all sequences
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * H + i_h

    if HEAD_FIRST:
        p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
    else:
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

    if i_t == 0:
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

    # [BK, BT]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))

    last_idx = min(i_t * BT + BT, T) - 1
    # scalar decay
    if USE_G:
        if HEAD_FIRST:
            b_g_last = tl.load(g + i_bh * T + last_idx)
            p_g = g + i_bh * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
        else:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

    # vector decay, h = Diag(gk) @ h
    if USE_GK:
        if HEAD_FIRST:
            p_gk = tl.make_block_ptr(gk + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + i_bh * T*K + last_idx * K + i_k * BK + tl.arange(0, BK)
            p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
        else:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)

        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_k = (b_k * tl.exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

    # vector decay, h = h @ Diag(gv)
    if USE_GV:
        if HEAD_FIRST:
            p_gv = tl.make_block_ptr(gv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + i_bh * T*V + last_idx * V + i_v * BV + tl.arange(0, BV)
            p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
        else:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

        b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)

        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_v = (b_v * tl.exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

    b_h = tl.dot(b_k, b_v)
    if i_t < NT - 1:
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(h + (i_bh * NT + i_t + 1) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + ((i_tg + 1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
    elif STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV']
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_reduction(
    h,
    g,
    gk,
    gv,
    kvt,
    ht,
    offsets,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(h + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        if i_t > 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min(i_t * BT + BT, T) - 1
        # scalar decay
        if USE_G:
            if HEAD_FIRST:
                b_g_last = tl.load(g + i_nh * T + last_idx)
            else:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_h *= tl.exp(b_g_last)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            if HEAD_FIRST:
                p_gk_last = gk + i_nh * T*K + last_idx * K + i_k * BK + tl.arange(0, BK)
                p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
            else:
                p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= tl.exp(b_gk_last)[:, None]

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            if HEAD_FIRST:
                p_gv_last = gv + i_nh * T*V + last_idx * V + i_v * BV + tl.arange(0, BV)
                p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
            else:
                p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= tl.exp(b_gv_last)[None, :]

    if STORE_FINAL_STATE:
        p_kvt = tl.make_block_ptr(kvt + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_kvt, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV']
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh_parallel(
    q,
    g,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    offsets,
    indices,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NV = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_hq, i_bg = i_bh // HQ, i_bh % HQ, i_bh // NG
    i_h = i_hq // NG
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * HQ + i_hq

    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + (i_bh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

    if i_t == NT - 1:
        if USE_FINAL_STATE_GRADIENT:
            p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh = tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_dh = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

    # [BK, BT]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))

    if USE_G:
        if HEAD_FIRST:
            p_g = g + i_bg * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
        else:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_q = (b_q * tl.exp(b_g)[None, :]).to(b_q.dtype)

    if USE_GK:
        if HEAD_FIRST:
            p_gk = tl.make_block_ptr(gk + i_bg * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        else:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_q = (b_q * tl.exp(b_gk)).to(b_q.dtype)

    if USE_GV:
        if HEAD_FIRST:
            p_gv = tl.make_block_ptr(gv + i_bg * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_gv)).to(b_do.dtype)

    b_dh = tl.dot(b_q, b_do)
    if i_t > 0:
        if HEAD_FIRST:
            p_dh = tl.make_block_ptr(dh + (i_bh * NT + i_t - 1) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_dh = tl.make_block_ptr(dh + ((i_tg - 1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
    elif STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV']
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh_reduction(
    g,
    gk,
    gv,
    dh,
    doq0,
    dh0,
    offsets,
    chunk_offsets,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_nh // NG
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        if HEAD_FIRST:
            p_dh = tl.make_block_ptr(dh + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dh, boundary_check=(0, 1)).to(tl.float32)
        if i_t < NT - 1:
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min(i_t * BT + BT, T) - 1
        if USE_G:
            if HEAD_FIRST:
                b_g_last = tl.load(g + i_bg * T + last_idx)
            else:
                b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_dh *= tl.exp(b_g_last)

        if USE_GK:
            if HEAD_FIRST:
                p_gk_last = gk + (i_bg * T + last_idx) * K + i_k * BK + tl.arange(0, BK)
                p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
            else:
                p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_dh *= tl.exp(b_gk_last)[:, None]

        if USE_GV:
            if HEAD_FIRST:
                p_gv_last = gv + (i_bg * T + last_idx) * V + i_v * BV + tl.arange(0, BV)
                p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
            else:
                p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_dh *= tl.exp(b_gv_last)[None, :]

    if STORE_INITIAL_STATE_GRADIENT:
        p_doq0 = tl.make_block_ptr(doq0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_doq0, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    states_in_fp32: bool = False,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        N, NT = len(offsets) - 1, len(indices)
        chunk_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)

    h = k.new_empty(B, H, NT, K, V, dtype=torch.float) if head_first else k.new_empty(B, NT, H, K, V, dtype=torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']) * triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_fwd_kernel_h_parallel[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first
    )
    kvt, ht = ht, (torch.empty_like(ht) if output_final_state else None)
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h_reduction[grid](
        h=h,
        g=g,
        gk=gk,
        gv=gv,
        kvt=kvt,
        ht=ht,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first
    )
    h = h.to(k.dtype) if not states_in_fp32 else h
    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    states_in_fp32: bool = False,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        N, NT = len(offsets) - 1, len(indices)
        chunk_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
    NG = HQ // H

    if head_first:
        dh = k.new_empty(B, HQ, NT, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    else:
        dh = k.new_empty(B, NT, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (triton.cdiv(K, meta['BK']) * triton.cdiv(V, meta['BV']), NT, B * HQ)
    chunk_bwd_kernel_dh_parallel[grid](
        q=q,
        g=g,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first
    )

    doq0, dh0 = dh0, (torch.empty_like(dh0) if dh0 is not None else None)
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * HQ)
    chunk_bwd_kernel_dh_reduction[grid](
        g=g,
        gk=gk,
        gv=gv,
        dh=dh,
        doq0=doq0,
        dh0=dh0,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first
    )
    dh = dh.to(q.dtype) if not states_in_fp32 else dh
    return dh, dh0
