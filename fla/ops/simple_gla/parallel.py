# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_global_cumsum, chunk_local_cumsum
from fla.ops.utils.op import exp
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    check_shared_mem,
    input_guard,
    is_intel_alchemist,
    is_nvidia_hopper
)

# https://github.com/intel/intel-xpu-backend-for-triton/issues/3449
triton_config = {'grf_mode': 'large'} if is_intel_alchemist else {}
NUM_WARPS = [2, 4, 8] if is_nvidia_hopper else [2, 4, 8, 16]


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
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
    cu_seqlens,
    chunk_indices,
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
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += ((i_k * all + bos) * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
    if OUTPUT_ATTENTIONS:
        attn += i_k * B * H * T * T + (bos * H + i_h * T) * T

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    # Q block and K block have overlap.
    # masks required
    if USE_G:
        # [BT,]
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf')).to(tl.float32)
        # rescale interchunk output
    else:
        b_gq = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_s *= exp(b_gq[:, None] - b_gk[None, :])
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = m_q[:, None] & m_k[None, :]
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s-1) * H) if i_s % BT > 0 else 0.
            # No concrete meaning. Just to avoid some layout bugs.
            b_s *= exp(b_gq[:, None] + (b_gn - b_g)[None, :])
            b_gq += b_gn - b_gp
        b_s = tl.where(m_s, b_s, 0)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
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
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr
):
    p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV] @ [BV, BS] = [BT, BS]
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.
            b_ds *= tl.where(m_k, exp(b_gn - b_g), 0)[None, :]
            if i_s > 0:
                b_dq *= exp(b_gn - b_gp)
        # [BT, BS] @ [BS, BK] = [BT, BK]
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k)

    if USE_G:
        # [BT,]
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
        # [BT, BK]
        b_dq *= exp(b_gq)[:, None]

    # Q block and K block have overlap. masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV] @ [BV, BS] = [BT, BS]
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_ds *= exp(b_gq[:, None] - b_gk[None, :])
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)

    b_dq *= scale
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_dg = tl.sum(b_dq * b_q, 1)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
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
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr
):
    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T
    # [BT, BK]
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_G:
        b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
    NTS = tl.cdiv(T, BS)
    # [BT, BK]
    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, BS]
        b_ds = tl.dot(b_v, tl.trans(b_do))
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            b_gp = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gn = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.
            if i_s >= 0:
                b_gpn = exp(b_gp - b_gn)
                b_dk *= b_gpn
                b_dv *= b_gpn
                b_gqn = exp(b_gq - b_gn)
                b_ds *= b_gqn[None, :]
                b_s *= b_gqn[None, :]
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        # [BT, BV]
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)

    if USE_G:
        b_gn = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        if i_t >= 0:
            b_gpn = exp(b_gn - b_gk)[:, None]
            b_dk *= b_gpn
            b_dv *= b_gpn

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            if i_s >= 0:
                b_gkq = exp(-b_gk[:, None] + b_gq[None, :])
                b_ds *= b_gkq
                b_s *= b_gkq
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.where(m_s, b_s, 0)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
    b_dk *= scale
    b_dv *= scale
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        b_dg = tl.load(dg + o_k * H, mask=m_k, other=0)
        b_dg -= tl.sum(b_dk * b_k, 1)
        tl.store(dg + o_k * H, b_dg.to(dg.dtype.element_ty), mask=m_k)


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BS', 'BK', 'BV', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
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
    cu_seqlens,
    chunk_indices,
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
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    dq += i_v * B * H * T * K
    dk += i_v * B * H * T * K
    dv += i_k * B * H * T * V
    if USE_G:
        dg += i_kv * B * H * T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
        dg += bos * H + i_h

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
        T=T,
        H=H,
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
        T=T,
        H=H,
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
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if check_shared_mem('hopper', k.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    elif check_shared_mem('ampere', k.device.index):
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    else:
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # local cumulative decay in log space
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)
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
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
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
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if check_shared_mem('hopper', k.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    elif check_shared_mem('ampere', k.device.index):
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    elif check_shared_mem('ada', k.device.index):
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
    else:
        BK = min(32, triton.next_power_of_2(K))
        BV = min(32, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    dq = torch.empty(NV, * q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    dk = torch.empty(NV, * k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(NK, * v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    dg = torch.empty(NK*NV, *g.shape, dtype=torch.float, device=q.device) if g is not None else None

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

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
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = chunk_global_cumsum(dg.sum(0), reverse=True, cu_seqlens=cu_seqlens) if g is not None else None
    return dq, dk, dv, dg


class ParallelSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, output_attentions, cu_seqlens):
        chunk_size = 128
        ctx.dtype = q.dtype

        o, g, attn = parallel_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            output_attentions=output_attentions,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, g, cu_seqlens)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), attn

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, da=None):
        q, k, v, g, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, dg = parallel_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            do=do,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
            cu_seqlens=cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.dtype) if dg is not None else None, None, None, None


def parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    output_attentions: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
    if output_attentions:
        assert cu_seqlens is None, "output_attentions=True is not supported with variable-length sequences"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, attn = ParallelSimpleGLAFunction.apply(
        q,
        k,
        v,
        g,
        scale,
        output_attentions,
        cu_seqlens
    )
    return o, attn
