# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
    'USE_GATE': lambda args: args['g_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_intra_chunk_kernel(
    q, k, v, g_cumsum, w1, w2,
    L, D,
    dq, dq_new, dk, dv, dw1, dw2, do, dg_cumsum,
    offsets, indices,
    T, scale,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,  BV: tl.constexpr,
    BT: tl.constexpr, S: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_GATE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    # offset calculations
    k += (bos * H + i_h) * K  # GQA when H!=HQ
    v += (bos * H + i_h) * V  # GQA when H!=HQ
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K

    q += (bos * HQ + i_hq) * K
    dq += (bos * HQ + i_hq) * K
    dq_new += (bos * HQ + i_hq) * K
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * V
    do += (bos * HQ + i_hq) * V
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K
    L += (bos * HQ + i_hq)
    D += (bos * HQ + i_hq)
    if USE_GATE:
        g_cumsum += (bos * HQ + i_hq)
        dg_cumsum += (bos * HQ + i_hq)

    # constants
    sm_scale = scale * 1.44269504

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_k = m_t[:, None] & (o_d[None, :] < K)
    p_do = do + o_t[:, None] * (HQ*V) + o_v[None, :]
    # [BT, BV]
    b_do = tl.load(p_do, mask=m_t[:, None] & (o_v[None, :] < V), other=0.0)

    p_delta = D + o_t * HQ
    b_delta = tl.load(p_delta, mask=m_t, other=0.0)
    p_l = L + o_t * HQ
    b_l = tl.load(p_l, mask=m_t, other=0.0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    p_dq = dq + o_t[:, None] * (HQ*K) + o_d[None, :]
    b_dq += tl.load(p_dq, mask=m_k, other=0.0)
    p_q = q + o_t[:, None] * (HQ*K) + o_d[None, :]
    b_q = tl.load(p_q, mask=m_k, other=0.0)

    if USE_GATE:
        p_gq_cumsum = g_cumsum + o_t * HQ
        b_gq_cumsum = tl.load(p_gq_cumsum, mask=m_t, other=0.0)
        b_dgq = tl.zeros([BT], dtype=tl.float32)
    else:
        b_dgq = None

    curr_start = (tl.floor(i_t * BT / S).to(tl.int32) * S).to(tl.int32)

    for offset in range(curr_start, i_t * BT, BT):
        o_k = (offset + tl.arange(0, BT)).to(tl.int64)
        mask = o_k < T
        m_kk = mask[:, None] & (o_d[None, :] < K)
        p_k = k + o_k[:, None] * (H*K) + o_d[None, :]
        b_k = tl.load(p_k, mask=m_kk, other=0.0)
        b_q_tmp = tl.zeros([BT, BK], dtype=tl.float32)
        b_q_tmp += b_q
        for i_t_small in range(i_t * BT - BT, offset, -BT):
            o_w = i_t_small + tl.arange(0, BT)
            m_w = (o_w[:, None] < T) & (o_d[None, :] < K)
            p_w1 = w1 + o_w[:, None] * (H*K) + o_d[None, :]
            b_w1 = tl.load(p_w1, mask=m_w, other=0.0)
            p_w2 = w2 + o_w[:, None] * (H*K) + o_d[None, :]
            b_w2 = tl.load(p_w2, mask=m_w, other=0.0)
            b_A_tmp = tl.dot(b_q_tmp.to(b_w1.dtype), tl.trans(b_w1))
            b_q_tmp -= tl.dot(b_A_tmp.to(b_w1.dtype), b_w2)
        b_q2 = b_q_tmp.to(b_k.dtype)
        b_A = tl.dot(b_q2, tl.trans(b_k))
        if USE_GATE:
            p_gk_cumsum = g_cumsum + o_k * HQ
            b_gk_cumsum = tl.load(p_gk_cumsum, mask=mask, other=0.0)
            b_A = b_A + b_gq_cumsum[:, None] - b_gk_cumsum[None, :]
            b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan
        b_A_softmax = tl.math.exp2(b_A * sm_scale - b_l[:, None])
        b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
        tl.atomic_add(
            dv + ((offset + tl.arange(0, BT)) * HQ * V)[:, None] + tl.arange(0, BV)[None, :],
            b_dv.to(dv.dtype.element_ty),
            mask=mask[:, None],
            sem='relaxed',
        )
        p_v = v + o_k[:, None] * (V*H) + o_v[None, :]
        b_v = tl.load(p_v, mask=mask[:, None] & (o_v[None, :] < V), other=0.0)
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_dA = ((b_dp - b_delta[:, None]) * b_A_softmax * scale)
        if USE_GATE:
            b_dgk = -tl.sum(b_dA, axis=0)
            tl.atomic_add(dg_cumsum + (offset + tl.arange(0, BT)) * HQ, b_dgk, mask=mask, sem='relaxed')
            b_dgq += tl.sum(b_dA, axis=1)
        b_dA = b_dA.to(b_v.dtype)
        b_dk = tl.dot(tl.trans(b_dA), b_q2)
        tl.atomic_add(dk + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dk, mask=mask[:, None], sem='relaxed')
        p_w1 = w1 + o_k[:, None] * (H*K) + o_d[None, :]
        b_w1 = tl.load(p_w1, mask=m_kk, other=0.0)
        p_w2 = w2 + o_k[:, None] * (H*K) + o_d[None, :]
        b_w2 = tl.load(p_w2, mask=m_kk, other=0.0)
        b_dA2 = tl.dot(b_dq.to(b_w2.dtype), tl.trans(b_w2)).to(b_v.dtype)
        b_A2 = tl.dot(b_q2.to(b_w1.dtype), tl.trans(b_w1)).to(b_v.dtype)
        b_dw2 = -tl.dot(tl.trans(b_A2), b_dq.to(b_v.dtype))
        tl.atomic_add(dw2 + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dw2, mask=mask[:, None], sem='relaxed')
        b_dw1 = -tl.dot(tl.trans(b_dA2), b_q2.to(b_v.dtype))
        tl.atomic_add(dw1 + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dw1, mask=mask[:, None], sem='relaxed')
        b_dq -= tl.dot(b_dA2, b_w1.to(b_v.dtype))
        b_dq += tl.dot(b_dA.to(b_k.dtype), b_k)

    p_dq_new = dq_new + o_t[:, None] * (HQ*K) + o_d[None, :]
    tl.store(p_dq_new, b_dq.to(dq_new.dtype.element_ty), mask=m_k)
    mask = i_t * BT + tl.arange(0, BT) < T
    if USE_GATE:
        tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ, b_dgq, mask=mask, sem='relaxed')


def parallel_path_bwd_intra_chunk_fn(
    q, k, v, g_cumsum, w1, w2,
    dq, dk, dv, dg_cumsum, dw1, dw2, do,
    scale, L, D,
    cu_seqlens,
    S, BT,
    chunk_indices: torch.LongTensor | None = None,
):
    assert dk.dtype == dv.dtype == dw1.dtype == dw2.dtype == torch.float32, 'atomic_add requires float32'
    B, T, HQ, K = q.shape
    assert dk.shape == dq.shape

    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    dq_new = torch.empty_like(dq, dtype=q.dtype)
    parallel_path_bwd_intra_chunk_kernel[(NT, B*HQ)](
        q=q, k=k, v=v, g_cumsum=g_cumsum,
        w1=w1, w2=w2, L=L, D=D,
        dq=dq, dq_new=dq_new, dk=dk, dv=dv, dw1=dw1, dw2=dw2,
        do=do, dg_cumsum=dg_cumsum,
        offsets=cu_seqlens, indices=indices,
        T=T, S=S, BT=BT, scale=scale,
        G=G, HQ=HQ, H=H, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
    )
    return dq_new, dk, dv, dw1, dw2, dg_cumsum
