# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_lens
from fla.ops.utils.op import exp2
from fla.ops.utils.softplus import softplus2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, contiguous


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HQ', 'G', 'K', 'V', 'BT', 'BS', 'BK', 'BV', 'ATTEND_CURRENT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_stickbreaking_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    rem,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_rem = rem + bos * HQ + i_hq + o_q * HQ

    # [BT, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    # [BT], log2 of the stick left over by the keys visited so far
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # keys are visited nearest first, so a key block only needs the stick left over by the blocks after it.
    # the first loop covers the diagonal block, where the causal mask applies
    for i_s in range(0, BT, BS):
        # [BS]
        o_k = i_t * BT + BT - BS - i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)

        if ATTEND_CURRENT:
            m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        else:
            m_s = (o_q[:, None] > o_k[None, :]) & m_k[None, :]
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        # log2(1 - sigmoid(z)), zeroed for masked keys so they take no part of the stick
        b_lb = tl.where(m_s, -softplus2(b_s), 0.)
        # log2(A) = b_s + reverse_cumsum(b_lb) + b_acc: log2(sigmoid(z)), then the stick left by the nearer keys
        b_p = tl.where(m_s, exp2(b_s + tl.cumsum(b_lb, axis=1, reverse=True) + b_acc[:, None]), 0.)
        b_o += tl.dot(b_p.to(b_v.dtype), b_v)
        b_acc += tl.sum(b_lb, 1)

    # the blocks below the diagonal are fully visible, walked from the nearest one down to the first
    for i_s in range(BT, (i_t + 1) * BT, BS):
        # [BS]
        o_k = (i_t + 1) * BT - BS - i_s + tl.arange(0, BS)
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=o_d[:, None] < K, other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=o_v[None, :] < V, other=0.0)

        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_lb = -softplus2(b_s)
        b_p = exp2(b_s + tl.cumsum(b_lb, axis=1, reverse=True) + b_acc[:, None])
        b_o += tl.dot(b_p.to(b_v.dtype), b_v)
        b_acc += tl.sum(b_lb, 1)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_rem, exp2(b_acc).to(p_rem.dtype.element_ty), mask=m_q)


@triton.jit
def _stickbreaking_attn_tile(b_s, b_acc, m_s, AXIS: tl.constexpr):
    # A for one tile whose keys run along AXIS, given b_acc, the log2 stick left by the nearer key blocks
    b_lb = tl.where(m_s, -softplus2(b_s), 0.)
    b_p = tl.where(m_s, exp2(b_s + tl.cumsum(b_lb, axis=AXIS, reverse=True) + b_acc), 0.)
    return b_lb, b_p


@triton.jit
def _stickbreaking_attn_bwd_tile(b_s, b_lb, b_p, b_dp, b_sa, b_c, m_s, AXIS: tl.constexpr):
    # dz = a - beta * (c - sum of a over the nearer keys), with a = A * <dO, v>
    b_a = b_p * b_dp
    b_near = tl.cumsum(b_a, axis=AXIS, reverse=True) - b_a + b_sa
    b_dz = tl.where(m_s, b_a - exp2(b_s + b_lb) * (b_c - b_near), 0.)
    return b_a, b_dz


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=['H', 'HQ', 'G', 'K', 'V', 'BS', 'BK', 'BV', 'ATTEND_CURRENT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'NS'])
def parallel_stickbreaking_attn_bwd_kernel_dq(
    q,
    k,
    v,
    do,
    drem,
    dq,
    acc,
    sa,
    c,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    NS,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    # [BS]
    o_q = i_t * BS + tl.arange(0, BS)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_dq = dq + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_snap = (bos + o_q) * HQ * NS + i_hq * NS

    # [BS, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BS, BV]
    b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
    # [BS]
    b_acc = tl.zeros([BS], dtype=tl.float32)
    b_sa = tl.zeros([BS], dtype=tl.float32)

    # first walk, nearest key block first: the row totals, and the snapshots the dkv pass restarts from
    for i_s in range(0, (i_t + 1) * BS, BS):
        i_k = i_t - i_s // BS
        o_k = i_k * BS + tl.arange(0, BS)
        m_k = o_k < T
        tl.store(acc + p_snap + i_k, b_acc, mask=m_q)
        tl.store(sa + p_snap + i_k, b_sa, mask=m_q)
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)
        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BV, BS]
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)

        if ATTEND_CURRENT:
            m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        else:
            m_s = (o_q[:, None] > o_k[None, :]) & m_k[None, :]
        # [BS, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_lb, b_p = _stickbreaking_attn_tile(b_s, b_acc[:, None], m_s, 1)
        b_acc += tl.sum(b_lb, 1)
        b_sa += tl.sum(b_p * tl.dot(b_do, b_v), 1)

    # c = sum_j A_ij <dO_i, v_j> + drem_i * rem_i, from the backward's own fp32 sums rather than from the stored o
    b_c = b_sa + tl.load(drem + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.).to(tl.float32) * exp2(b_acc)
    tl.store(c + (bos + o_q) * HQ + i_hq, b_c, mask=m_q)

    # second walk: dq
    b_acc = tl.zeros([BS], dtype=tl.float32)
    b_sa = tl.zeros([BS], dtype=tl.float32)
    # [BS, BK]
    b_dq = tl.zeros([BS, BK], dtype=tl.float32)
    for i_s in range(0, (i_t + 1) * BS, BS):
        i_k = i_t - i_s // BS
        o_k = i_k * BS + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)
        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BV, BS]
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)

        if ATTEND_CURRENT:
            m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        else:
            m_s = (o_q[:, None] > o_k[None, :]) & m_k[None, :]
        # [BS, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_dp = tl.dot(b_do, b_v)
        b_lb, b_p = _stickbreaking_attn_tile(b_s, b_acc[:, None], m_s, 1)
        b_a, b_dz = _stickbreaking_attn_bwd_tile(b_s, b_lb, b_p, b_dp, b_sa[:, None], b_c[:, None], m_s, 1)
        b_dq += tl.dot(b_dz.to(b_k.dtype), tl.trans(b_k))
        b_acc += tl.sum(b_lb, 1)
        b_sa += tl.sum(b_a, 1)

    tl.store(p_dq, (b_dq * scale).to(p_dq.dtype.element_ty), mask=m_q[:, None] & (o_d[None, :] < K))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=['H', 'HQ', 'G', 'K', 'V', 'BT', 'BS', 'BK', 'BV', 'ATTEND_CURRENT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'NS'])
def parallel_stickbreaking_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    do,
    dk,
    dv,
    acc,
    sa,
    c,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    NS,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # one program sums dk and dv for its key block over every query that sees it, in a fixed order: no atomics
    i_s, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    # [BS]
    o_k = i_s * BS + tl.arange(0, BS)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_k = o_k < T
    p_k = k + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
    p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
    p_dk = dk + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
    p_dv = dv + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]

    # [BS, BK]
    b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
    # [BS, BV]
    b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i_g in range(G):
        i_hq = i_h * G + i_g
        for i_t in range(i_s * BS, T, BT):
            # [BT]
            o_q = i_t + tl.arange(0, BT)
            m_q = o_q < T
            p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
            p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
            p_snap = (bos + o_q) * HQ * NS + i_hq * NS + i_s
            # [BT, BK]
            b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
            # [BT, BV]
            b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
            # [BT]
            b_acc = tl.load(acc + p_snap, mask=m_q, other=0.0)
            b_sa = tl.load(sa + p_snap, mask=m_q, other=0.0)
            b_c = tl.load(c + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0)

            if ATTEND_CURRENT:
                m_s = (o_k[:, None] <= o_q[None, :]) & m_k[:, None] & m_q[None, :]
            else:
                m_s = (o_k[:, None] < o_q[None, :]) & m_k[:, None] & m_q[None, :]
            # [BS, BT], keys along axis 0 as in parallel_attn_bwd_kernel_dkv
            b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_lb, b_p = _stickbreaking_attn_tile(b_s, b_acc[None, :], m_s, 0)
            b_a, b_dz = _stickbreaking_attn_bwd_tile(b_s, b_lb, b_p, b_dp, b_sa[None, :], b_c[None, :], m_s, 0)
            # [BS, BV]
            b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
            # [BS, BK]
            b_dk = tl.dot(b_dz.to(b_q.dtype), b_q, b_dk)

    tl.store(p_dk, (b_dk * scale).to(p_dk.dtype.element_ty), mask=m_k[:, None] & (o_d[None, :] < K))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_k[:, None] & (o_v[None, :] < V))


def parallel_stickbreaking_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attend_current: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = 64
    BS = 64 if check_shared_mem('hopper', q.device.index) else 32
    BK = max(16, triton.next_power_of_2(K))
    BV = min(128, max(16, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    rem = torch.empty(B, T, HQ, dtype=q.dtype, device=q.device)
    grid = (NV, NT, B * HQ)
    parallel_stickbreaking_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        rem=rem,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
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
        ATTEND_CURRENT=attend_current,
    )
    return o, rem


def parallel_stickbreaking_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    drem: torch.Tensor,
    scale: float,
    attend_current: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    # the dq pass takes one key block per query block, so its query blocks are BS rows as well
    BS = 64 if check_shared_mem('hopper', q.device.index) else 32
    BT = 32
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        NT = len(chunk_indices)
        NS = triton.cdiv(int(prepare_lens(cu_seqlens).max()), BS)
    else:
        chunk_indices = None
        NT = NS = triton.cdiv(T, BS)

    # per (query row, key block), on entry: log2 of the stick left, and the sum of A * <dO, v> over the nearer keys
    acc = q.new_empty(B, T, HQ, NS, dtype=torch.float)
    sa = q.new_empty(B, T, HQ, NS, dtype=torch.float)
    c = q.new_empty(B, T, HQ, dtype=torch.float)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    parallel_stickbreaking_attn_bwd_kernel_dq[(NT, B * HQ)](
        q=q,
        k=k,
        v=v,
        do=do,
        drem=drem,
        dq=dq,
        acc=acc,
        sa=sa,
        c=c,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NS=NS,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BS=BS,
        BK=BK,
        BV=BV,
        ATTEND_CURRENT=attend_current,
    )
    parallel_stickbreaking_attn_bwd_kernel_dkv[(NT, B * H)](
        q=q,
        k=k,
        v=v,
        do=do,
        dk=dk,
        dv=dv,
        acc=acc,
        sa=sa,
        c=c,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NS=NS,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        ATTEND_CURRENT=attend_current,
    )
    return dq, dk, dv


class StickBreakingAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, attend_current, cu_seqlens):
        o, rem = parallel_stickbreaking_attn_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            attend_current=attend_current,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.attend_current = attend_current
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), rem

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, drem):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = parallel_stickbreaking_attn_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            drem=drem,
            scale=ctx.scale,
            attend_current=ctx.attend_current,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None


def parallel_stickbreaking_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attend_current: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Stick-breaking attention (https://arxiv.org/abs/2410.17980).

    Query `i` gives each visible key `j` the weight `A_ij = sigmoid(z_ij) * prod_l (1 - sigmoid(z_il))`,
    where `z_ij = scale * q_i k_j` and `l` runs over the visible keys after `j`,
    that is `j < l < i`, or `j < l <= i` with `attend_current`.
    Walking back from the query, each key takes a sigmoid share of the stick that the nearer keys left over.
    The weights are not normalized, so a query can leave part of its stick unused.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (float, Optional):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        attend_current (bool, Optional):
            Whether a query also attends to the key at its own position.
            If `False`, query `i` only sees keys `j < i`, and the first query outputs zeros. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
        rem (torch.Tensor):
            The stick each query leaves unused, `1 - sum_j A_ij`, of shape `[B, T, HQ]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    HQ, H = q.shape[2], k.shape[2]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    if q.shape[-1] > 256:
        raise ValueError(f"The key dimension ({q.shape[-1]}) can not be larger than 256.")
    if v.shape[-1] > 256:
        raise ValueError(f"The value dimension ({v.shape[-1]}) can not be larger than 256.")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )

    o, rem = StickBreakingAttentionFunction.apply(q, k, v, scale, attend_current, cu_seqlens)
    return o, rem
