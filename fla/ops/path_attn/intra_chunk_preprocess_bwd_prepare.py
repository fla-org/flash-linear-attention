# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    "USE_GATE": lambda args: args['g_cumsum'] is not None,
    "IS_VARLEN": lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_bwd_kernel_prepare(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    L,
    D,
    h,
    q_new,
    k_new,
    AT,
    dA_local,
    dv,
    do,
    dg_cumsum,
    scale,
    indices,  # varlen helper
    offsets,  # varlen helper
    chunk_offsets,  # varlen helper
    T,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
    RETURN_H: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)
        boh = (i_n * NT).to(tl.int64)

    sm_scale = scale * 1.44269504
    # offset calculations
    dA_local += (bos*HQ + i_hq) * BT
    AT += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    do += (bos*HQ + i_hq) * V
    dv += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    if RETURN_H:
        h += ((boh + i_t) * H + i_h) * K * K
    else:
        h += (bos*H + i_h) * K
    if USE_GATE:
        g_cumsum += (bos*HQ + i_hq)
        dg_cumsum += (bos*HQ + i_hq)
    L += (bos*HQ + i_hq)
    D += (bos*HQ + i_hq)

    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_r = o_t < T
    m_k = m_r[:, None] & (o_d[None, :] < K)
    m_v = m_r[:, None] & (o_v[None, :] < V)
    m_T = m_r[:, None] & (o_i[None, :] < BT)
    p_q = q + o_t[:, None] * (HQ*K) + o_d[None, :]
    p_k = k + o_d[:, None] + o_t[None, :] * (H*K)
    p_w = w + o_t[:, None] * (H*K) + o_d[None, :]
    p_beta = beta + o_t * H

    b_q = tl.load(p_q, mask=m_k, other=0.0)
    b_kt = tl.load(p_k, mask=(o_d[:, None] < K) & m_r[None, :], other=0.0)
    b_w = tl.load(p_w, mask=m_k, other=0.0)
    b_beta = tl.load(p_beta, mask=m_r, other=0.0)
    p_T = AT + o_t[:, None] * (BT*H) + o_i[None, :]
    b_T = tl.load(p_T, mask=m_T, other=0.0) * b_beta[None, :]

    m_t = o_i[:, None] >= o_i[None, :]
    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_kt.dtype), b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)

    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_qwT.dtype))
    p_q_new = q_new + o_t[:, None] * (K*HQ) + o_d[None, :]
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), mask=m_k)

    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)  # tf32
        p_h = h + o_t[:, None] * (K * H) + o_d[None, :]
        tl.store(p_h, b_Twb.to(p_h.dtype.element_ty), mask=m_k)
        b_T_wbk = tl.dot(b_T.to(b_wbk.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = k_new + o_d[:, None] + o_t[None, :] * (K*H)
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)), b_T_wbk)
                           ).to(p_k_new.dtype.element_ty), mask=(o_d[:, None] < K) & m_r[None, :])

    if USE_GATE:
        p_g_cumsum = g_cumsum + o_t * HQ
        b_g_cumsum = tl.load(p_g_cumsum, mask=m_r, other=0.0)
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

    p_l = L + o_t * HQ
    b_l = tl.load(p_l, mask=m_r, other=0.0)
    p_delta = D + o_t * HQ
    delta = tl.load(p_delta, mask=m_r, other=0.0)

    b_A_softmax = tl.exp2(tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale - b_l[:, None], float("-inf")))
    p_do = do + o_t[:, None] * (HQ*V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_v, other=0.0)
    b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
    p_dv = dv + o_t[:, None] * (HQ*V) + o_v[None, :]
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

    p_v = v + o_v[:, None] + o_t[None, :] * (H*V)
    b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_r[None, :], other=0.0)
    b_dp = tl.dot(b_do, b_v)
    b_dA = ((b_dp - delta[:, None]) * b_A_softmax * scale)
    if USE_GATE:
        b_dgq = tl.sum(b_dA, axis=1) - tl.sum(b_dA, axis=0)
        p_dg = dg_cumsum + o_t * HQ
        tl.store(p_dg, b_dgq.to(p_dg.dtype.element_ty), mask=m_r)
    p_dA = dA_local + o_t[:, None] * (BT*HQ) + o_i[None, :]
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_T)


def intra_chunk_preprocess_bwd_prepare_fn(q, k, v, w, beta, g_cumsum, A, L, D, do, scale, return_h=True, cu_seqlens=None,
                                          chunk_indices: torch.LongTensor | None = None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ//H

    V = v.shape[-1]
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    h = torch.empty_like(w)
    dA_local = torch.empty(B, T, HQ, BT, dtype=q.dtype, device=q.device)
    dv = torch.empty(B, T, HQ, V, device=q.device, dtype=torch.float32)
    dg_cumsum = torch.empty_like(g_cumsum) if g_cumsum is not None else None

    chunk_transform_qk_bwd_kernel_prepare[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        beta=beta,
        g_cumsum=g_cumsum,
        AT=A,
        dA_local=dA_local,
        dv=dv,
        dg_cumsum=dg_cumsum,
        do=do,
        L=L,
        D=D,
        h=h,
        q_new=q_new,
        k_new=k_new,
        scale=scale,
        offsets=cu_seqlens,
        indices=indices,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        G=G,
        HQ=HQ,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BT=BT,
        RETURN_H=return_h,
    )
    return q_new, k_new, h, dA_local, dv, dg_cumsum
