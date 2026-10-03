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
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
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
    # [BT]
    b_log_rem = tl.zeros([BT], dtype=tl.float32)

    # visit nearer blocks first to carry their remaining stick into earlier blocks
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
        # masked keys must leave the remaining stick unchanged
        b_log_om_beta = tl.where(m_s, -softplus2(b_s), 0.)
        b_p = tl.where(m_s, exp2(b_s + tl.cumsum(b_log_om_beta, axis=1, reverse=True) + b_log_rem[:, None]), 0.)
        b_o += tl.dot(b_p.to(b_v.dtype), b_v)
        b_log_rem += tl.sum(b_log_om_beta, 1)

    # blocks below the diagonal need no causal mask
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
        b_log_om_beta = -softplus2(b_s)
        b_p = exp2(b_s + tl.cumsum(b_log_om_beta, axis=1, reverse=True) + b_log_rem[:, None])
        b_o += tl.dot(b_p.to(b_v.dtype), b_v)
        b_log_rem += tl.sum(b_log_om_beta, 1)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_rem, exp2(b_log_rem).to(p_rem.dtype.element_ty), mask=m_q)


@triton.jit
def _stickbreaking_attn_weights(b_s, b_log_rem, m_s, AXIS: tl.constexpr):
    # the scan axis follows keys, which are transposed in the dkv kernel
    b_log_om_beta = tl.where(m_s, -softplus2(b_s), 0.)
    b_p = tl.where(m_s, exp2(b_s + tl.cumsum(b_log_om_beta, axis=AXIS, reverse=True) + b_log_rem), 0.)
    return b_log_om_beta, b_p


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
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
    log_rem,
    delta_acc,
    delta,
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
    o_snap = (bos + o_q) * HQ * NS + i_hq * NS

    # [BS, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BS, BV]
    b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
    # [BS]
    b_log_rem = tl.zeros([BS], dtype=tl.float32)
    b_delta_acc = tl.zeros([BS], dtype=tl.float32)

    # snapshots let the dkv kernel reconstruct each tile independently
    for i_s in range(0, (i_t + 1) * BS, BS):
        i_k = i_t - i_s // BS
        o_k = i_k * BS + tl.arange(0, BS)
        m_k = o_k < T
        tl.store(log_rem + o_snap + i_k, b_log_rem, mask=m_q)
        tl.store(delta_acc + o_snap + i_k, b_delta_acc, mask=m_q)
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
        b_log_om_beta, b_p = _stickbreaking_attn_weights(b_s=b_s, b_log_rem=b_log_rem[:, None], m_s=m_s, AXIS=1)
        b_log_rem += tl.sum(b_log_om_beta, 1)
        b_delta_acc += tl.sum(b_p * tl.dot(b_do, b_v), 1)

    # recompute row totals in fp32 to avoid cancellation against rounded outputs
    b_delta = b_delta_acc + tl.load(drem + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.).to(tl.float32) * exp2(b_log_rem)
    tl.store(delta + (bos + o_q) * HQ + i_hq, b_delta, mask=m_q)

    b_log_rem = tl.zeros([BS], dtype=tl.float32)
    b_delta_acc = tl.zeros([BS], dtype=tl.float32)
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
        b_log_om_beta, b_p = _stickbreaking_attn_weights(b_s=b_s, b_log_rem=b_log_rem[:, None], m_s=m_s, AXIS=1)
        # see README.md for the logit gradient formula
        b_pdp = b_p * b_dp
        b_delta_cumsum = tl.cumsum(b_pdp, axis=1, reverse=True) - b_pdp + b_delta_acc[:, None]
        b_ds = tl.where(m_s, b_pdp - exp2(b_s + b_log_om_beta) * (b_delta[:, None] - b_delta_cumsum), 0.)
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        b_log_rem += tl.sum(b_log_om_beta, 1)
        b_delta_acc += tl.sum(b_pdp, 1)

    tl.store(p_dq, (b_dq * scale).to(p_dq.dtype.element_ty), mask=m_q[:, None] & (o_d[None, :] < K))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
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
    log_rem,
    delta_acc,
    delta,
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
    # each key block has one owner so gradient accumulation needs no atomics
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
            o_snap = (bos + o_q) * HQ * NS + i_hq * NS + i_s
            # [BT, BK]
            b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
            # [BT, BV]
            b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
            # [BT]
            b_log_rem = tl.load(log_rem + o_snap, mask=m_q, other=0.0)
            b_delta_acc = tl.load(delta_acc + o_snap, mask=m_q, other=0.0)
            b_delta = tl.load(delta + (bos + o_q) * HQ + i_hq, mask=m_q, other=0.0)

            if ATTEND_CURRENT:
                m_s = (o_k[:, None] <= o_q[None, :]) & m_k[:, None] & m_q[None, :]
            else:
                m_s = (o_k[:, None] < o_q[None, :]) & m_k[:, None] & m_q[None, :]
            # [BS, BT]
            b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_log_om_beta, b_p = _stickbreaking_attn_weights(b_s=b_s, b_log_rem=b_log_rem[None, :], m_s=m_s, AXIS=0)
            b_pdp = b_p * b_dp
            b_delta_cumsum = tl.cumsum(b_pdp, axis=0, reverse=True) - b_pdp + b_delta_acc[None, :]
            b_ds = tl.where(m_s, b_pdp - exp2(b_s + b_log_om_beta) * (b_delta[None, :] - b_delta_cumsum), 0.)
            # [BS, BV]
            b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
            # [BS, BK]
            b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)

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
    BS = 64 if check_shared_mem(arch='hopper', tensor_idx=q.device.index) else 32
    BK = max(16, triton.next_power_of_2(K))
    BV = min(128, max(16, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT) if cu_seqlens is not None else None
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
    # the dq query blocks match the snapshot key blocks
    BS = 64 if check_shared_mem(arch='hopper', tensor_idx=q.device.index) else 32
    BT = 32
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BS)
        NT = len(chunk_indices)
        NS = triton.cdiv(int(prepare_lens(cu_seqlens=cu_seqlens).max()), BS)
    else:
        chunk_indices = None
        NT = NS = triton.cdiv(T, BS)

    # snapshots store the remaining stick log and weighted gradient sum before each key block
    log_rem = q.new_empty(B, T, HQ, NS, dtype=torch.float)
    delta_acc = q.new_empty(B, T, HQ, NS, dtype=torch.float)
    delta = q.new_empty(B, T, HQ, dtype=torch.float)
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
        log_rem=log_rem,
        delta_acc=delta_acc,
        delta=delta,
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
        log_rem=log_rem,
        delta_acc=delta_acc,
        delta=delta,
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


class ParallelStickBreakingAttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
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
    @input_guard
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
            Queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`. GQA is applied if HQ is divisible by H.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        scale (float, Optional):
            Scale factor for attention scores. Default: `1 / sqrt(K)`.
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

    o, rem = ParallelStickBreakingAttentionFunction.apply(q, k, v, scale, attend_current, cu_seqlens)
    return o, rem
