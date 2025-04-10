# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_split(
    k,
    v,
    g,
    gk,
    gv,
    hs,
    hr,
    h0,
    ht,
    cu_seqlens,
    split_indices,
    T,
    S: tl.constexpr,
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
    IS_VARLEN: tl.constexpr,
):
    # handle one split at a time
    # i_h: head index
    # i_n: sequence index
    # i_s: local split index inside a sequence
    i_k, i_v, i_sh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_ss, i_h = i_sh // H, i_sh % H
    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T
    i_nh = i_n * H + i_h

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # for the first split, we directly store the state as the final result
    if i_s == 0:
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        p_hr = tl.make_block_ptr(hr + i_sh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_hr, b_h.to(p_hr.dtype.element_ty), boundary_check=(0, 1))
    for i_t in range(tl.cdiv(i_s * S, BT), tl.cdiv(min(i_s * S + S, T), BT)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_h *= exp(b_g_last)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= exp(b_gk_last)[:, None]

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= exp(b_gv_last)[None, :]

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    # if there are more than one splits, we store the result to (unreduced) hs
    # otherwise, we store the result to ht as the final state
    if NS > 1:
        p_hs = tl.make_block_ptr(hs + i_sh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_hs, b_h.to(p_hs.dtype.element_ty), boundary_check=(0, 1))
    elif STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_reduction(
    g,
    gk,
    gv,
    hs,
    hr,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    S: tl.constexpr,
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
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NS = tl.cdiv(T, S)
        boh = i_n * NS

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # skip the first split
    for i_s in range(1, NS):
        p_hs = tl.make_block_ptr(hs + ((boh + i_s-1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_hr = tl.make_block_ptr(hr + ((boh + i_s) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_hr, b_h.to(p_hr.dtype.element_ty), boundary_check=(0, 1))

        for i_t in range(tl.cdiv(i_s * S, BT), tl.cdiv(min(i_s * S + S, T), BT)):
            last_idx = min(i_t * BT + BT, T) - 1
            # scalar decay
            if USE_G:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
                b_h *= exp(b_g_last)

            # vector decay, h = Diag(gk) @ h
            if USE_GK:
                p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
                b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
                b_h *= exp(b_gk_last)[:, None]

            # vector decay, h = h @ Diag(gv)
            if USE_GV:
                p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
                b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
                b_h *= exp(b_gv_last)[None, :]

    if NS > 1:
        if STORE_FINAL_STATE:
            p_hs = tl.make_block_ptr(hs + ((boh + NS-1) * H + i_h)*K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h += tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh_split(
    q,
    g,
    gk,
    gv,
    do,
    dht,
    dhs,
    dhr,
    dh0,
    cu_seqlens,
    split_indices,
    scale,
    T,
    S: tl.constexpr,
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
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # handle one split at a time
    # i_h: head index
    # i_n: sequence index
    # i_s: local split index inside a sequence
    i_k, i_v, i_sh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_ss, i_hq = i_sh // HQ, i_sh % HQ
    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T
    i_nh = i_n * HQ + i_hq
    i_h = i_hq // NG

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if i_s == NS - 1:
        if USE_FINAL_STATE_GRADIENT:
            p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
        p_dhr = tl.make_block_ptr(dhr + i_sh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dhr, b_dh.to(p_dhr.dtype.element_ty), boundary_check=(0, 1))

    for i_t in range(tl.cdiv(min(i_s * S + S, T), BT) - 1, tl.cdiv(i_s * S, BT) - 1, -1):
        p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        last_idx = min(i_t * BT + BT, T) - 1
        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp(b_g_last)

        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_q = (b_q * exp(b_gk)).to(b_q.dtype)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_dh *= exp(b_gk_last)[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = (b_do * exp(b_gv)).to(b_do.dtype)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_dh *= exp(b_gv_last)[None, :]

        b_dh += tl.dot(b_q, b_do)

    if NS > 1:
        p_dhs = tl.make_block_ptr(dhs + i_sh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dhs, b_dh.to(p_dhs.dtype.element_ty), boundary_check=(0, 1))
    elif STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh_reduction(
    g,
    gk,
    gv,
    dhs,
    dhr,
    dh0,
    cu_seqlens,
    split_offsets,
    T,
    S: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
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
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NS = tl.cdiv(T, S)
        boh = i_n * NS

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_s in range(NS - 2, -1, -1):
        p_dhs = tl.make_block_ptr(dhs + ((boh+i_s+1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dhr = tl.make_block_ptr(dhr + ((boh+i_s) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dhs, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_dhr, b_dh.to(p_dhr.dtype.element_ty), boundary_check=(0, 1))

        for i_t in range(tl.cdiv(min(i_s * S + S, T), BT) - 1, tl.cdiv(i_s * S, BT) - 1, -1):
            last_idx = min(i_t * BT + BT, T) - 1
            # scalar decay
            if USE_G:
                b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
                b_dh *= exp(b_g_last)

            if USE_GK:
                p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
                b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
                b_dh *= exp(b_gk_last)[:, None]

            if USE_GV:
                p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
                b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
                b_dh *= exp(b_gv_last)[None, :]

    if NS > 1:
        if STORE_INITIAL_STATE_GRADIENT:
            p_dhs = tl.make_block_ptr(dhs + (boh * H + i_h)*K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh += tl.load(p_dhs, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    split_offsets: Optional[torch.LongTensor] = None,
    split_indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    split_size: int = 256,
    states_in_fp32: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    # B: batch size
    # N: the actual number of sequences in the batch
    # H: number of heads
    # T: sequence length, can be variable across sequences
    # S: split size, a multiple of chunk size
    # BT: chunk size
    S, BT = split_size, chunk_size
    assert S % BT == 0, f"The `split_size` (got {S}) must be a multiple of `chunk_size` {BT}"
    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1]

    # unreduced kv states per split
    hs = k.new_empty(NS, H, K, V, dtype=torch.float)
    # reduced states per split
    hr = k.new_empty(NS, H, K, V, dtype=torch.float if states_in_fp32 else k.dtype)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    # parallelized over splits
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), NS * H)
    chunk_fwd_kernel_h_split[grid](
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        hs=hs,
        hr=hr,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices,
        T=T,
        S=S,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h_reduction[grid](
        g=g,
        gk=gk,
        gv=gv,
        hs=hs,
        hr=hr,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        S=S,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return hr, ht


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
    cu_seqlens: Optional[torch.Tensor] = None,
    split_offsets: Optional[torch.Tensor] = None,
    split_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    split_size: int = 256,
    states_in_fp32: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    # B: batch size
    # N: the actual number of sequences in the batch
    # H: number of heads
    # T: sequence length, can be variable across sequences
    # S: split size, a multiple of chunk size
    # BT: chunk size
    S, BT = max(chunk_size, min(split_size, triton.next_power_of_2(T))), chunk_size
    assert S % BT == 0, f"The `split_size` (got {S}) must be a multiple of `chunk_size` {BT}"
    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1]
    # number of groups in GQA
    NG = HQ // H

    dhs = q.new_empty(NS, HQ, K, V, dtype=torch.float)
    dhr = q.new_empty(NS, HQ, K, V, dtype=torch.float if states_in_fp32 else k.dtype)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    # parallelized over splits
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), NS * HQ)
    chunk_bwd_kernel_dh_split[grid](
        q=q,
        g=g,
        gk=gk,
        gv=gv,
        do=do,
        dht=dht,
        dhs=dhs,
        dhr=dhr,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices,
        scale=scale,
        T=T,
        S=S,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )

    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * HQ)
    chunk_bwd_kernel_dh_reduction[grid](
        g=g,
        gk=gk,
        gv=gv,
        dhs=dhs,
        dhr=dhr,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        S=S,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return dhr, dh0
