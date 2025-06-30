# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import is_nvidia_hopper

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dkv(
    q_star,
    k,
    v,
    beta,
    h_kv,
    g,
    do,
    dh_kv,
    dk_beta,
    dg,
    dv,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_dh = tl.make_block_ptr(dh_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_q = tl.make_block_ptr(q_star, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)

    # calculation
    b_dg_last += tl.sum(b_h * b_dh)
    b_dg_last *= exp(b_g_last)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m
    b_ds = tl.dot(b_do, tl.trans(b_v))
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)
    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)
    b_g_exp_k = tl.where(m_t, exp(-b_g + b_g_last), 0)
    b_ds = b_ds * b_m
    b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_dg -= tl.sum(b_dk * b_k, axis=1)
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
    b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dk = tl.make_block_ptr(dk_beta, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dq(
    q_star,
    k,
    v,
    beta,
    h_kv,
    g,
    do,
    dq,
    dg_prev,
    dg,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg_prev += bos * H + i_h
    dg += bos * H + i_h
    dq += (bos * H + i_h) * K

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_q = tl.make_block_ptr(q_star, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg_prev = tl.make_block_ptr(dg_prev, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_ds = tl.dot(b_do, tl.trans(b_v)) * b_m
    b_g_exp_q = exp(b_g)
    b_dq = tl.dot(b_do, b_h.to(b_do.dtype)) * b_g_exp_q[:, None]
    b_dg = tl.sum(b_dq * b_q, axis=1) + tl.load(p_dg_prev, boundary_check=(0,))
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def chunk_mesa_net_h_kv_bwd_intra_separate_fn(
    q_star,
    k,
    v,
    beta,
    h_kv,
    dh_kv,
    g,
    do,
    cu_seqlens,
    chunk_size=64
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    dq = torch.empty_like(q_star, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    grid = (NT, B * H)
    chunk_mesa_net_h_kv_bwd_intra_kernel_dkv[grid](
        q_star=q_star,
        k=k,
        v=v,
        beta=beta,
        h_kv=h_kv,
        g=g,
        do=do,
        dh_kv=dh_kv,
        dk_beta=dk,
        dg=dg,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV
    )
    dg_final = torch.empty_like(dg)
    chunk_mesa_net_h_kv_bwd_intra_kernel_dq[grid](
        q_star=q_star,
        k=k,
        v=v,
        beta=beta,
        h_kv=h_kv,
        g=g,
        do=do,
        dg=dg_final,
        dg_prev=dg,
        dq=dq,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV
    )
    return dq, dk, dv, dg_final
