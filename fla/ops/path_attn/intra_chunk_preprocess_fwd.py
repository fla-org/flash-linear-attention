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
    "USE_G": lambda args: args['g_cumsum'] is not None,
    "IS_VARLEN": lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_fwd_kernel(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    o,
    A,
    L,
    M,
    w2,
    q_new,
    k_new,
    scale,
    indices,  # varlen helper
    offsets,  # varlen helper
    T,
    H: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    sm_scale = scale * 1.44269504
    # offset calculations
    A += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w2 += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    o += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    if USE_G:
        g_cumsum += (bos*HQ + i_hq)
    L += (bos*HQ + i_hq)
    M += (bos*HQ + i_hq)

    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_r = o_t < T
    m_k = m_r[:, None] & (o_d[None, :] < K)
    m_v = m_r[:, None] & (o_v[None, :] < V)
    p_q = q + o_t[:, None] * (HQ*K) + o_d[None, :]
    p_k = k + o_d[:, None] + o_t[None, :] * (H*K)
    p_w = w + o_t[:, None] * (H*K) + o_d[None, :]
    p_v = v + o_t[:, None] * (H*V) + o_v[None, :]
    p_beta = beta + o_t * H
    p_T = A + o_t[:, None] * (BT*H) + o_i[None, :]

    b_beta = tl.load(p_beta, mask=m_r, other=0.0)
    b_q = tl.load(p_q, mask=m_k, other=0.0)
    b_kt = tl.load(p_k, mask=(o_d[:, None] < K) & m_r[None, :], other=0.0)
    b_v = tl.load(p_v, mask=m_v, other=0.0)
    b_w = tl.load(p_w, mask=m_k, other=0.0)
    b_T = tl.load(p_T, mask=m_r[:, None] & (o_i[None, :] < BT), other=0.0)
    b_T = b_T * b_beta[None, :]

    m_t = o_i[:, None] >= o_i[None, :]

    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_q.dtype), b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT.to(b_q.dtype), b_wbk), 0)

    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_q.dtype))
    p_q_new = q_new + o_t[:, None] * (K*HQ) + o_d[None, :]
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), mask=m_k)

    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)
        p_w2 = w2 + o_t[:, None] * (K*H) + o_d[None, :]
        tl.store(p_w2, b_Twb.to(p_w2.dtype.element_ty), mask=m_k)
        b_T_wbk = tl.dot(b_T.to(b_kt.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = k_new + o_d[:, None] + o_t[None, :] * (K*H)
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)), b_T_wbk)
                           ).to(p_k_new.dtype.element_ty), mask=(o_d[:, None] < K) & m_r[None, :])

    if USE_G:
        p_g_cumsum = g_cumsum + o_t * HQ
        b_g_cumsum = tl.load(p_g_cumsum, mask=m_r, other=0.0)
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

    b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale, float("-inf"))
    m_i = tl.max(b_qkT_softmax, 1)
    b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
    l_i = tl.sum(b_qkT_softmax, 1)
    b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
    p_o = o + o_t[:, None] * (V*HQ) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
    p_l = L + o_t * HQ
    p_m = M + o_t * HQ
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), mask=m_r)
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), mask=m_r)


def intra_chunk_preprocess_fwd_fn(q, k, v, w, beta, g_cumsum, A, scale, BT, cu_seqlens,
                                  chunk_indices: torch.LongTensor | None = None):
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    V = v.shape[-1]
    q_new = torch.empty_like(q, dtype=torch.float32)  # for stability
    k_new = torch.empty_like(k)
    o = torch.empty(B, T, HQ, V, device=q.device, dtype=torch.float32)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    L = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    M = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    w2 = torch.empty_like(w)
    G = HQ//H

    intra_chunk_preprocess_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        beta=beta,
        g_cumsum=g_cumsum,
        o=o,
        A=A,
        L=L,
        M=M,
        w2=w2,
        q_new=q_new,
        k_new=k_new,
        scale=scale,
        offsets=cu_seqlens,
        indices=indices,
        T=T,
        H=H,
        G=G,
        HQ=HQ,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BT=BT,
        num_warps=4 if BT == 64 else 2,
    )
    return q_new, k_new, w2, o, L, M
