# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum
from fla.ops.utils.exp import safe_exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_G': lambda args: args['g'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
    ],
    key=["BT", "BS", "BK", "BV", "USE_G"],
)
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    attn,
    scale,
    offsets,
    indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_G: tl.constexpr
):
    tl.static_assert(not (USE_OFFSETS and HEAD_FIRST), "USE_OFFSETS and HEAD_FIRST cannot be True at the same time")
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    o += i_k * B * T * H * V

    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    k += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    v += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    o += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    if USE_G:
        g += i_bh * T if HEAD_FIRST else bos * H + i_h
    if OUTPUT_ATTENTIONS:
        attn += (bos * H + i_h * T) * T + i_k * B * H * T * T
    stride_qk = K if HEAD_FIRST else H * K
    stride_vo = V if HEAD_FIRST else H * V
    stride_g = 1 if HEAD_FIRST else H

    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap.
    # masks required
    if USE_G:
        p_gq = tl.make_block_ptr(g, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))
        # [BT,]
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
        # rescale interchunk output
    else:
        b_gq = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            p_gk = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,))
            b_s *= safe_exp(b_gq[:, None] - b_gk[None, :])
            b_s = tl.where(m_s, b_s, 0)
        else:
            b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        o_k += BS

    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            p_g = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * stride_g)
            b_gp = tl.load(g + (i_s-1) * stride_g) if i_s % BT > 0 else 0.
            # No concrete meaning. Just to avoid some layout bugs.
            b_s *= safe_exp(b_gq[:, None] + (b_gn - b_g)[None, :])
            b_gq += (b_gn - b_gp)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_simple_gla_bwd_kernel_dq(
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dq,
    dg,
    stride_qk,
    stride_vo,
    stride_g,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr
):
    p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, stride_vo), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV] @ [BV, BS] = [BT, BS]
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            p_g = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * stride_g)
            b_gp = tl.load(g + (i_s - 1) * stride_g) if i_s % BT > 0 else 0.
            b_ds *= safe_exp(b_gn - b_g)[None, :]
            if i_s > 0:
                b_dq *= safe_exp(b_gn - b_gp)
        # [BT, BS] @ [BS, BK] = [BT, BK]
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k)

    if USE_G:
        p_gq = tl.make_block_ptr(g, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))
        # [BT,]
        b_gq = tl.load(p_gq, boundary_check=(0,))
        # [BT, BK]
        b_dq *= safe_exp(b_gq)[:, None]

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap. masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, stride_vo), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV] @ [BV, BS] = [BT, BS]
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            p_gk = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,))
            b_ds *= safe_exp(b_gq[:, None] - b_gk[None, :])
        b_ds = tl.where(o_q[:, None] >= o_k[None, :], b_ds, 0)
        # [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
        o_k += BS

    b_dq *= scale
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_dg = tl.sum(b_dq * b_q, 1)
        p_dg = tl.make_block_ptr(dg, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_simple_gla_bwd_kernel_dkv(
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dk,
    dv,
    dg,
    scale,
    stride_qk,
    stride_vo,
    stride_g,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr
):
    # [BT, BK]
    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_G:
        p_gk = tl.make_block_ptr(g, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,))
    NTS = tl.cdiv(T, BS)
    # [BT, BK]
    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            p_gq = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,))
            b_gp = tl.load(g + (min(i_s + BS, T) - 1) * stride_g)
            b_gn = tl.load(g + (i_s - 1) * stride_g) if i_s % BT > 0 else 0.
            if i_s >= 0:
                tmp = safe_exp(b_gp - b_gn)
                b_dk *= tmp
                b_dv *= tmp
                tmp2 = safe_exp(b_gq - b_gn)
                b_ds *= tmp2[None, :]
                b_s *= tmp2[None, :]
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        # [BT, BV]
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)

    if USE_G:
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * stride_g)
        if i_t >= 0:
            tmp2 = safe_exp(b_g_last - b_gk)[:, None]
            b_dk *= tmp2
            b_dv *= tmp2

    o_q = i_t * BT + tl.arange(0, BS)
    o_k = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_ds = tl.dot(b_v, tl.trans(b_do))
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            p_gq = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,))
            if i_s >= 0:
                tmp = safe_exp(-b_gk[:, None] + b_gq[None, :])
                b_ds *= tmp
                b_s *= tmp
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.where(m_s, b_s, 0)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
        o_q += BS
    b_dk *= scale
    b_dv *= scale
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))
        b_dg = tl.load(p_dg, boundary_check=(0,))
        b_dg -= tl.sum(b_dk * b_k, 1)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_G': lambda args: args['g'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "BS", "BK", "BV", "USE_G"],
)
@triton.jit
def parallel_simple_gla_bwd_kernel(
    q,
    k,
    v,
    g,
    do,
    dq,
    dk,
    dv,
    dg,
    scale,
    offsets,
    indices,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_G: tl.constexpr
):
    tl.static_assert(not (USE_OFFSETS and HEAD_FIRST), "USE_OFFSETS and HEAD_FIRST cannot be True at the same time")
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    dq += i_v * B * H * T * K
    dk += i_v * B * H * T * K
    dv += i_k * B * H * T * V
    if USE_G:
        dg += i_kv * B * H * T

    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (i_bh * T * K) if HEAD_FIRST else (bos * H + i_h) * K
    k += (i_bh * T * K) if HEAD_FIRST else (bos * H + i_h) * K
    v += (i_bh * T * V) if HEAD_FIRST else (bos * H + i_h) * V
    do += (i_bh * T * V) if HEAD_FIRST else (bos * H + i_h) * V
    dq += (i_bh * T * K) if HEAD_FIRST else (bos * H + i_h) * K
    dk += (i_bh * T * K) if HEAD_FIRST else (bos * H + i_h) * K
    dv += (i_bh * T * V) if HEAD_FIRST else (bos * H + i_h) * V
    if USE_G:
        g += (i_bh * T) if HEAD_FIRST else (bos * H + i_h)
        dg += (i_bh * T) if HEAD_FIRST else (bos * H + i_h)
    stride_qk = K if HEAD_FIRST else H * K
    stride_vo = V if HEAD_FIRST else H * V
    stride_g = 1 if HEAD_FIRST else H

    parallel_simple_gla_bwd_kernel_dq(
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dq=dq,
        dg=dg,
        scale=scale,
        stride_qk=stride_qk,
        stride_vo=stride_vo,
        stride_g=stride_g,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        USE_G=USE_G
    )
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dk=dk,
        dv=dv,
        dg=dg,
        scale=scale,
        stride_qk=stride_qk,
        stride_vo=stride_vo,
        stride_g=stride_g,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        USE_G=USE_G
    )


def parallel_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    output_attentions: bool = False,
    chunk_size: int = 128,
    head_first: bool = True,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)

    # local cumulative decay in log space
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first)
    grid = (NK * NV, NT, B * H)
    o = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None

    parallel_simple_gla_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        o=o,
        attn=attn,
        scale=scale,
        offsets=offsets,
        indices=indices,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
    )
    o = o.sum(0)

    if output_attentions:
        attn = attn.sum(0)
    return o, g, attn


def parallel_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int = 128,
    head_first: bool = True,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    BK = min(128, triton.next_power_of_2(k.shape[-1]))
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    dq = torch.empty(NV, * q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    dk = torch.empty(NV, * k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(NK, * v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    dg = torch.empty(NK*NV, *g.shape, dtype=torch.float, device=q.device) if g is not None else None

    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)

    grid = (NK * NV, NT, B * H)
    parallel_simple_gla_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        offsets=offsets,
        indices=indices,
        scale=scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = chunk_global_cumsum(dg.sum(0), reverse=True, head_first=head_first, offsets=offsets) if g is not None else None
    return dq, dk, dv, dg


class ParallelSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, output_attentions, head_first, offsets):
        chunk_size = 128
        ctx.dtype = q.dtype

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        o, g, attn = parallel_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            output_attentions=output_attentions,
            head_first=head_first,
            offsets=offsets,
            indices=indices,
            chunk_size=chunk_size)
        ctx.save_for_backward(q, k, v, g, offsets, indices)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.head_first = head_first
        return o.to(q.dtype), attn

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, da=None):
        q, k, v, g, offsets, indices = ctx.saved_tensors
        dq, dk, dv, dg = parallel_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            do=do,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
            offsets=offsets,
            indices=indices,
            head_first=ctx.head_first)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.dtype) if dg is not None else None, None, None, None, None


def parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    output_attentions: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        g (torch.Tensor):
            Forget gates of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        assert not head_first, "head_first must be False when cu_seqlens are provided"
    if g is not None:
        g = g.float()
    if output_attentions:
        assert cu_seqlens is None, "output_attentions=True is not supported with variable-length sequences"
    o, attn = ParallelSimpleGLAFunction.apply(q, k, v, g, scale, output_attentions, head_first, cu_seqlens)
    return o, attn
