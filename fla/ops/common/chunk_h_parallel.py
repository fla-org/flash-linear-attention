# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Fully parallelized state passing.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
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
    cu_seqlens,
    chunk_indices,
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
    IS_VARLEN: tl.constexpr
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
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * H + i_h

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
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

    # vector decay, h = Diag(gk) @ h
    if USE_GK:
        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)

        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

    # vector decay, h = h @ Diag(gv)
    if USE_GV:
        p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

        b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

    b_h = tl.dot(b_k, b_v)
    if i_t < NT - 1:
        p_h = tl.make_block_ptr(h + ((i_tg + 1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
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
    cu_seqlens,
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
    IS_VARLEN: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        if i_t > 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

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

    if STORE_FINAL_STATE:
        p_kvt = tl.make_block_ptr(kvt + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_kvt, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
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
    cu_seqlens,
    chunk_indices,
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
    IS_VARLEN: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NV = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * HQ + i_hq

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
        p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)

    if USE_GK:
        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_q = (b_q * exp(b_gk)).to(b_q.dtype)

    if USE_GV:
        p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_do = (b_do * exp(b_gv)).to(b_do.dtype)

    b_dh = tl.dot(b_q, b_do)
    if i_t > 0:
        p_dh = tl.make_block_ptr(dh + ((i_tg - 1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
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
    cu_seqlens,
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
    IS_VARLEN: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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
        p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dh, boundary_check=(0, 1)).to(tl.float32)
        if i_t < NT - 1:
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min(i_t * BT + BT, T) - 1
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
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, K, V, dtype=torch.float)
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
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None
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
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None
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
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    NG = HQ // H

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
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
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
        USE_GV=gv is not None
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
        cu_seqlens=cu_seqlens,
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
        USE_GV=gv is not None
    )
    dh = dh.to(q.dtype) if not states_in_fp32 else dh
    return dh, dh0
