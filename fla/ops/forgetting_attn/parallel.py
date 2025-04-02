# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl
from einops import rearrange, reduce

from fla.ops.common.utils import prepare_chunk_indices
from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, input_guard


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['B', 'H', 'G', 'K', 'V', 'BK', 'BV'],
)
@triton.jit
def parallel_forgetting_attn_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    lse,
    scale,
    offsets,
    indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT,]
    b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gk = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_gk = tl.load(p_gk, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) + b_gq[:, None] - b_gk[None, :]
        b_s = tl.where(o_q[:, None] >= o_k[None, :], b_s, float('-inf'))

        # [BT]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        # [BT, BS]
        b_p = tl.exp(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m

    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gk = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)

        b_gn = tl.load(g + (bos + min(i_s + BS, T) - 1) * HQ + i_hq).to(tl.float32)
        b_gp = tl.load(g + (bos + i_s - 1) * HQ + i_hq).to(tl.float32) if i_s % BT > 0 else 0.
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) + b_gq[:, None] + (b_gn - b_gk)[None, :]

        b_gq += b_gn - b_gp
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        # [BT, BS]
        b_p = tl.exp(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m

    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_forgetting_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([8] if check_shared_mem('hopper') else [])
        for num_stages in [2, 3, 4]
    ],
    key=['B', 'H', 'G', 'K', 'V', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_forgetting_attn_bwd_kernel_dq(
    q,
    k,
    v,
    g,
    lse,
    delta,
    do,
    dq,
    dg,
    scale,
    offsets,
    indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos * HQ + i_hq), (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT]
    b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT]
    b_dg = tl.zeros([BT,], dtype=tl.float32)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_gk = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_k = i_s + tl.arange(0, BS)
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_gk = tl.load(p_gk, boundary_check=(0,))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) + (b_gq - b_lse)[:, None] - b_gk[None, :]
        b_p = tl.exp(tl.where(o_q[:, None] >= o_k[None, :], b_s, float('-inf')))

        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        # [BT]
        b_dg += tl.sum(b_ds, 1)

    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_gk = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)

        b_gn = tl.load(g + (bos + min(i_s + BS, T) - 1) * HQ + i_hq).to(tl.float32)
        b_gp = tl.load(g + (bos + i_s - 1) * HQ + i_hq).to(tl.float32) if i_s % BT > 0 else 0.
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) + (b_gq - b_lse)[:, None] + (b_gn - b_gk)[None, :]
        b_p = tl.exp(b_s)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        # [BT]
        b_dg += tl.sum(b_ds, 1)

        b_gq += b_gn - b_gp

    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['B', 'H', 'G', 'K', 'V', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_forgetting_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g,
    lse,
    delta,
    do,
    dk,
    dv,
    dg,
    offsets,
    indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_gk = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos * HQ + i_hq), (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    # [BT]
    b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_gq = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)

        m_q = o_q < T
        m_s = (o_k[:, None] <= o_q[None, :]) & m_k[:, None] & m_q[None, :]
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q)) - b_gk[:, None] + (b_gq - b_lse)[None, :]
        b_p = tl.where(m_s, tl.exp(b_s), 0)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        # [BT]
        b_dg -= tl.sum(b_ds, 1)

    b_gk -= tl.load(g + (bos + min((i_t + 1) * BT, T) - 1) * HQ + i_hq).to(tl.float32)
    for i_s in range((i_t + 1) * BT, T, BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_gq = tl.make_block_ptr(g + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        # [BS]
        o_q = i_s + tl.arange(0, BS)
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)

        b_gn = tl.load(g + (bos + min(i_s + BS, T) - 1) * HQ + i_hq).to(tl.float32)
        b_gp = tl.load(g + (bos + i_s - 1) * HQ + i_hq).to(tl.float32) if i_s % BT > 0 else 0.
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q)) + (b_gp - b_gk)[:, None] + (b_gq - b_lse)[None, :]
        b_p = tl.exp(b_s)
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        # [BT]
        b_dg -= tl.sum(b_ds, 1)

        b_gk -= b_gn - b_gp

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def parallel_forgetting_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int = 128,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    BK = max(16, triton.next_power_of_2(K))
    assert V <= 256, "V must be less than or equal to 256"
    if check_shared_mem('hopper'):
        BS = min(64, max(16, triton.next_power_of_2(T)))
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
    BV = min(256, max(16, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    grid = (NV, NT, B * HQ)
    parallel_forgetting_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        o=o,
        lse=lse,
        scale=scale,
        offsets=offsets,
        indices=indices,
        B=B,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_forgetting_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_forgetting_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_forgetting_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    scale: float = None,
    chunk_size: int = 128,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = chunk_size
    BS = min(32, max(16, triton.next_power_of_2(T)))
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    delta = parallel_forgetting_attn_bwd_preprocess(o, do)
    dq = q.new_empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float)
    dk = q.new_empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float)
    dv = q.new_empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float)
    dg = q.new_empty(g.shape, dtype=torch.float)
    # NOTE: the original `dg` can be destroyed during autotuning
    # this is [a known triton issue](https://github.com/triton-lang/triton/issues/5082), which will be fixed in 3.3 (?)
    # so we need to make a copy of `dg`
    dg2 = q.new_empty(g.shape, dtype=torch.float)
    grid = (NV, NT, B * HQ)
    parallel_forgetting_attn_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg=dg,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    parallel_forgetting_attn_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg=dg2,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV
    )
    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    dg = dg.add_(dg2)
    return dq, dk, dv, dg


@torch.compile
class ParallelFoxFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, offsets):
        ctx.dtype = q.dtype
        if check_shared_mem('hopper'):
            chunk_size = min(128, max(16, triton.next_power_of_2(q.shape[1])))
        else:
            chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))
        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None

        g = chunk_local_cumsum(g, chunk_size, offsets=offsets, indices=indices, head_first=False)
        o, lse = parallel_forgetting_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
            offsets=offsets,
            indices=indices
        )
        ctx.save_for_backward(q, k, v, g, o, lse)
        ctx.chunk_size = chunk_size
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, g, o, lse = ctx.saved_tensors
        dq, dk, dv, dg = parallel_forgetting_attn_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            o=o,
            lse=lse,
            do=do,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
            offsets=ctx.offsets,
            indices=ctx.indices
        )
        dg = chunk_global_cumsum(dg, reverse=True, head_first=False, offsets=ctx.offsets)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), None, None, None, None, None, None, None, None


def parallel_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            Forget gates (in **log space**) of shape `[B, T, HQ]` if `head_first=False` else `[B, HQ, T]`.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if g is not None:
        g = g.float()
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v))
        g = rearrange(g, 'b h t -> b t h')
    o = ParallelFoxFunction.apply(q, k, v, g, scale, cu_seqlens)
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o
