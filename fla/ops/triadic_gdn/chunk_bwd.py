# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.triadic_gdn.chunk_fwd import chunk_triadic_gdn_fwd_h, get_state_block_size, triadic_intra_r
from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp


@triton.heuristics({
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_bwd_kernel_dhu(
    q,
    k,
    k2,
    q2,
    g,
    beta,
    Ai,
    Aqk,
    do,
    dx,
    dh,
    dht,
    dh0,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Runs the state gradient recurrence backwards over the chunks of one value block of one head,
    keeping the value block of all `E` slices in registers.

    `dh` receives the gradient of the state after each chunk, and `dx` the gradient of
    `beta * v - sum_e (beta * k2_e * exp(g_e)) * (k @ S_e)`, the right-hand side of the UT transform.
    Every tile is transposed so that the narrow value block is the M dimension of each `tl.dot`.
    """
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_v[:, None] & m_k[None, :]
    # [BV, BK] tile of a [K, V] state
    p_h = o_k[None, :] * V + o_v[:, None]

    # one transposed [BV, BK] state gradient tile per slice
    b_dhs = ()
    for i_e in tl.static_range(E):
        if USE_FINAL_STATE_GRADIENT:
            b_s = tl.load(dht + (i_nh * E + i_e) * K*V + p_h, mask=m_h, other=0).to(tl.float32)
        else:
            b_s = tl.zeros([BV, BK], dtype=tl.float32)
        b_dhs = b_dhs + (b_s,)

    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + o_i
        m_t = o_t < T
        m_vt = m_v[:, None] & m_t[None, :]
        p_qk = (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        p_kt = (bos + o_t[None, :]) * H*K + i_h * K + o_k[:, None]
        # [BV, BT] tile of a [T, V] tensor
        p_vt = (bos + o_t[None, :]) * H*V + i_h * V + o_v[:, None]
        # [BT, BT] tile of a [T, BT] matrix
        p_A = (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]
        p_e = (bos + o_t) * H*E + i_h * E
        p_last = (bos + tl.minimum(i_t * BT + BT, T) - 1) * H*E + i_h * E
        # [BT, BK] and [BK, BT]
        b_q = tl.load(q + p_qk, mask=m_t[:, None] & m_k[None, :], other=0)
        b_k = tl.load(k + p_qk, mask=m_t[:, None] & m_k[None, :], other=0)
        b_kt = tl.load(k + p_kt, mask=m_k[:, None] & m_t[None, :], other=0)
        # [BV, BT]
        b_do = tl.load(do + p_vt, mask=m_vt, other=0)
        b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)

        # du = Aqk^T @ do + sum_e (k2_e * exp(g_last_e - g_e)) * (k @ dS_e)
        b_du = tl.dot(b_do, tl.load(Aqk + p_A, mask=m_t[:, None], other=0).to(b_do.dtype))
        for i_e in tl.static_range(E):
            b_dhb = b_dhs[i_e].to(b_k.dtype)
            tl.store(dh + (((boh + i_t) * H + i_h) * E + i_e) * K*V + p_h, b_dhb.to(dh.dtype.element_ty), mask=m_h)
            b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g_last = tl.load(g + p_last + i_e).to(tl.float32)
            b_du += (b_k2 * exp(b_g_last - b_g))[None, :] * tl.dot(b_dhb, b_kt)
        b_dx = tl.dot(b_du.to(b_k.dtype), tl.load(Ai + p_A, mask=m_t[:, None], other=0))
        tl.store(dx + p_vt, b_dx.to(dx.dtype.element_ty), mask=m_vt)

        # dS_e <- exp(g_last_e) * dS_e + scale * q^T @ (q2_e * exp(g_e) * do) - k^T @ (beta * k2_e * exp(g_e) * dx)
        b_new = ()
        for i_e in tl.static_range(E):
            b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_q2 = tl.load(q2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g_last = tl.load(g + p_last + i_e).to(tl.float32)
            b_dh = b_dhs[i_e] * exp(b_g_last)
            b_dh += tl.dot(((scale * b_q2 * exp(b_g))[None, :] * b_do).to(b_q.dtype), b_q)
            b_dh -= tl.dot(((b_beta * b_k2 * exp(b_g))[None, :] * b_dx).to(b_k.dtype), b_k)
            b_new = b_new + (b_dh,)
        b_dhs = b_new

    if USE_INITIAL_STATE:
        for i_e in tl.static_range(E):
            tl.store(dh0 + (i_nh * E + i_e) * K*V + p_h, b_dhs[i_e].to(dh0.dtype.element_ty), mask=m_h)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_bwd_kernel_inter(
    q,
    k,
    k2,
    q2,
    g,
    beta,
    u,
    do,
    dx,
    h,
    dh,
    dq,
    dk,
    qs,
    ks,
    kd,
    ss,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    TG,
    NTG,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Gradients that flow through the chunk-start state `S_e` and the chunk-end state gradient `dS_e`.

    For one key block, accumulates into `dq` and `dk` and writes per-slice row reductions over this block:
    `qs = rowsum(q * (do @ S_e^T))`, `ks = rowsum(k * (dx @ S_e^T))`, `kd = rowsum(k * (u @ dS_e^T))`
    and `ss = sum(S_e * dS_e)`, which the gate gradients consume after summing over key blocks.
    """
    i_k, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    p_e = (bos + o_t) * H*E + i_h * E
    p_last = (bos + tl.minimum(i_t * BT + BT, T) - 1) * H*E + i_h * E
    p_qk = (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
    # [BT, BK]
    b_q = tl.load(q + p_qk, mask=m_tk, other=0).to(tl.float32)
    b_k = tl.load(k + p_qk, mask=m_tk, other=0).to(tl.float32)
    b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)

    o_v = tl.arange(0, BV)
    m_v = o_v < V
    p_v = (bos + o_t[:, None]) * H*V + i_h * V + o_v[None, :]
    p_h = (i_tg * H + i_h) * E*K*V + o_k[:, None] * V + o_v[None, :]
    # [BT, BV], shared by all slices
    b_do = tl.load(do + p_v, mask=m_t[:, None] & m_v[None, :], other=0)
    b_dx = tl.load(dx + p_v, mask=m_t[:, None] & m_v[None, :], other=0).to(b_do.dtype)
    b_u = tl.load(u + p_v, mask=m_t[:, None] & m_v[None, :], other=0).to(b_do.dtype)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_e in range(E):
        # [BK, BV]
        b_h = tl.load(h + p_h + i_e * K*V, mask=m_k[:, None] & m_v[None, :], other=0)
        b_dh = tl.load(dh + p_h + i_e * K*V, mask=m_k[:, None] & m_v[None, :], other=0)
        # [BT, BK]
        b_p1 = tl.dot(b_do, tl.trans(b_h))
        b_p2 = tl.dot(b_dx, tl.trans(b_h))
        b_p3 = tl.dot(b_u, tl.trans(b_dh))
        b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_q2 = tl.load(q2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_g_last = tl.load(g + p_last + i_e).to(tl.float32)
        b_dq += (scale * b_q2 * exp(b_g))[:, None] * b_p1
        b_dk += (b_k2 * exp(b_g_last - b_g))[:, None] * b_p3 - (b_beta * b_k2 * exp(b_g))[:, None] * b_p2

        p_r = ((i_k * TG + bos + o_t) * H + i_h) * E + i_e
        tl.store(qs + p_r, tl.sum(b_q * b_p1, 1), mask=m_t)
        tl.store(ks + p_r, tl.sum(b_k * b_p2, 1), mask=m_t)
        tl.store(kd + p_r, tl.sum(b_k * b_p3, 1), mask=m_t)
        tl.store(ss + ((i_k * NTG + i_tg) * H + i_h) * E + i_e, tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32)))

    tl.store(dq + p_qk, b_dq.to(dq.dtype.element_ty), mask=m_tk)
    tl.store(dk + p_qk, b_dk.to(dk.dtype.element_ty), mask=m_tk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_bwd_kernel_intra(
    q,
    k,
    v,
    k2,
    q2,
    g,
    beta,
    u,
    do,
    dx,
    dq,
    dk,
    dv,
    dbeta,
    dR,
    dRq,
    ks,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    TG,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    NK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Gradients through the intra-chunk matrices `A` and `Aqk`.

    Finishes `dq`, `dk`, `dv` and `dbeta`, and writes the gradients of `R` and `Rq` for the per-slice gate kernel.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_s = (o_i[:, None] > o_i[None, :]) & m_t[:, None] & m_t[None, :]
    m_A = (o_i[:, None] >= o_i[None, :]) & m_t[:, None] & m_t[None, :]
    p_e = (bos + o_t) * H*E + i_h * E
    p_A = (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]
    b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)

    # dA = -dx @ u^T (strictly lower), dAqk = do @ u^T (lower), and the direct beta gradient rowsum(dx * v)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dAqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v < V)[None, :]
        p_v = (bos + o_t[:, None]) * H*V + i_h * V + o_v[None, :]
        b_u = tl.load(u + p_v, mask=m_tv, other=0)
        b_dx = tl.load(dx + p_v, mask=m_tv, other=0)
        b_do = tl.load(do + p_v, mask=m_tv, other=0)
        b_v = tl.load(v + p_v, mask=m_tv, other=0)
        b_dA += tl.dot(b_dx.to(b_u.dtype), tl.trans(b_u))
        b_dAqk += tl.dot(b_do.to(b_u.dtype), tl.trans(b_u))
        b_dbeta += tl.sum(b_dx.to(tl.float32) * b_v.to(tl.float32), 1)
        tl.store(dv + p_v, (b_beta[:, None] * b_dx.to(tl.float32)).to(dv.dtype.element_ty), mask=m_tv)
    b_dA = tl.where(m_s, -b_dA, 0)
    b_dAqk = tl.where(m_A, b_dAqk, 0)

    # k @ k^T and q @ k^T
    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    b_qk = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_qk = (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        b_q = tl.load(q + p_qk, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_k = tl.load(k + p_qk, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_kk += tl.dot(b_k, tl.trans(b_k))
        b_qk += tl.dot(b_q, tl.trans(b_k))

    b_R, b_Rq = triadic_intra_r(k2, q2, g, bos, i_t, i_h, T, H, E, BT, BC, BE)
    b_dbeta += tl.sum(b_dA * b_kk * b_R, 1)
    # gradients of k_i . k_j and q_i . k_j
    b_dP = b_dA * b_beta[:, None] * b_R
    b_dPq = b_dAqk * scale * b_Rq
    # gradients of R and Rq
    b_dR = b_dA * b_beta[:, None] * b_kk
    b_dRq = b_dAqk * scale * b_qk

    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        p_qk = (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        b_q = tl.load(q + p_qk, mask=m_tk, other=0)
        b_k = tl.load(k + p_qk, mask=m_tk, other=0)
        b_dq = tl.load(dq + p_qk, mask=m_tk, other=0).to(tl.float32)
        b_dk = tl.load(dk + p_qk, mask=m_tk, other=0).to(tl.float32)
        b_dq += tl.dot(b_dPq.to(b_k.dtype), b_k)
        b_dk += tl.dot((b_dP + tl.trans(b_dP)).to(b_k.dtype), b_k) + tl.dot(tl.trans(b_dPq).to(b_q.dtype), b_q)
        tl.store(dq + p_qk, b_dq.to(dq.dtype.element_ty), mask=m_tk)
        tl.store(dk + p_qk, b_dk.to(dk.dtype.element_ty), mask=m_tk)

    # the state term of the beta gradient, -sum_e k2_e * exp(g_e) * ks_e
    for i_e in range(E):
        b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        for i_k in range(NK):
            b_dbeta -= b_k2 * exp(b_g) * tl.load(ks + ((i_k * TG + bos + o_t) * H + i_h) * E + i_e, mask=m_t, other=0)
    tl.store(dbeta + (bos + o_t) * H + i_h, b_dbeta.to(dbeta.dtype.element_ty), mask=m_t)
    tl.store(dR + p_A, b_dR, mask=m_t[:, None])
    tl.store(dRq + p_A, b_dRq, mask=m_t[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_bwd_kernel_gate(
    q2,
    k2,
    g,
    beta,
    dR,
    dRq,
    qs,
    ks,
    kd,
    ss,
    dq2,
    dk2,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    TG,
    NTG,
    H: tl.constexpr,
    E: tl.constexpr,
    NK: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Gradients of one slice's second key, second query and gate within a chunk.

    Combines the gradients of `R` and `Rq` with the state reductions of the inter kernel, and writes `dg` after
    the reverse cumulative sum over the chunk that undoes the chunk-local cumsum of the forward pass.
    """
    i_e, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_A = (o_i[:, None] >= o_i[None, :]) & m_t[:, None] & m_t[None, :]
    i_last = tl.minimum(i_t * BT + BT, T) - 1 - i_t * BT
    p_e = (bos + o_t) * H*E + i_h * E + i_e
    p_A = (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]

    b_k2 = tl.load(k2 + p_e, mask=m_t, other=0).to(tl.float32)
    b_q2 = tl.load(q2 + p_e, mask=m_t, other=0).to(tl.float32)
    b_g = tl.load(g + p_e, mask=m_t, other=0).to(tl.float32)
    b_g_last = tl.load(g + (bos + i_t * BT + i_last) * H*E + i_h * E + i_e).to(tl.float32)
    b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)

    # terms of R = sum_e k2_e k2_e^T * D_e and Rq = sum_e q2_e k2_e^T * D_e, with D_e[i, j] = exp(g_e[i] - g_e[j])
    b_d = tl.where(m_A, exp(tl.where(m_A, b_g[:, None] - b_g[None, :], 0)), 0)
    b_tR = tl.load(dR + p_A, mask=m_t[:, None], other=0) * b_d
    b_tRq = tl.load(dRq + p_A, mask=m_t[:, None], other=0) * b_d
    b_dk2 = tl.sum(b_tR * b_k2[None, :], 1) + tl.sum(b_tR * b_k2[:, None], 0) + tl.sum(b_tRq * b_q2[:, None], 0)
    b_dq2 = tl.sum(b_tRq * b_k2[None, :], 1)
    b_tR *= b_k2[:, None] * b_k2[None, :]
    b_tRq *= b_q2[:, None] * b_k2[None, :]
    b_dg = tl.sum(b_tR + b_tRq, 1) - tl.sum(b_tR + b_tRq, 0)

    # terms through the chunk-start state and the chunk-end state gradient
    b_qs = tl.zeros([BT], dtype=tl.float32)
    b_ks = tl.zeros([BT], dtype=tl.float32)
    b_kd = tl.zeros([BT], dtype=tl.float32)
    b_ss = tl.zeros([1], dtype=tl.float32)
    for i_k in range(NK):
        p_r = ((i_k * TG + bos + o_t) * H + i_h) * E + i_e
        b_qs += tl.load(qs + p_r, mask=m_t, other=0)
        b_ks += tl.load(ks + p_r, mask=m_t, other=0)
        b_kd += tl.load(kd + p_r, mask=m_t, other=0)
        b_ss += tl.load(ss + ((i_k * NTG + i_tg) * H + i_h) * E + i_e + tl.zeros([1], dtype=tl.int64))
    b_w = b_k2 * exp(b_g_last - b_g)
    b_dq2 += scale * exp(b_g) * b_qs
    b_dk2 += exp(b_g_last - b_g) * b_kd - b_beta * exp(b_g) * b_ks
    b_dg += scale * b_q2 * exp(b_g) * b_qs - b_beta * b_k2 * exp(b_g) * b_ks - b_w * b_kd
    b_dg += tl.where(o_i == i_last, tl.sum(exp(b_g_last) * b_ss, 0) + tl.sum(b_w * b_kd, 0), 0)
    # undo the chunk-local cumsum
    b_dg = tl.sum(b_dg, 0) - tl.cumsum(b_dg, 0) + b_dg

    tl.store(dk2 + p_e, b_dk2.to(dk2.dtype.element_ty), mask=m_t)
    tl.store(dq2 + p_e, b_dq2.to(dq2.dtype.element_ty), mask=m_t)
    tl.store(dg + p_e, b_dg.to(dg.dtype.element_ty), mask=m_t)


def chunk_triadic_gdn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Ai: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    h: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K, V, E = *k.shape, v.shape[-1], k2.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    if h is None:
        _, h, u, _ = chunk_triadic_gdn_fwd_h(
            k=k,
            v=v,
            k2=k2,
            g=g,
            beta=beta,
            Ai=Ai,
            initial_state=initial_state,
            save_states=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )

    BV, num_warps = get_state_block_size(K, E)
    dh = torch.empty_like(h)
    dx = torch.empty_like(v)
    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None
    chunk_triadic_gdn_bwd_kernel_dhu[(triton.cdiv(V, BV), N * H)](
        q=q,
        k=k,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        Ai=Ai,
        Aqk=Aqk,
        do=do,
        dx=dx,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=BV,
        num_warps=num_warps,
        num_stages=2,
    )

    BK = min(64, triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    qs = q.new_empty(NK, B, T, H, E, dtype=torch.float32)
    ks = torch.empty_like(qs)
    kd = torch.empty_like(qs)
    ss = q.new_empty(NK, B * NT if cu_seqlens is None else NT, H, E, dtype=torch.float32)
    chunk_triadic_gdn_bwd_kernel_inter[(NK, NT, B * H)](
        q=q,
        k=k,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        u=u,
        do=do,
        dx=dx,
        h=h,
        dh=dh,
        dq=dq,
        dk=dk,
        qs=qs,
        ks=ks,
        kd=kd,
        ss=ss,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        TG=B * T,
        NTG=ss.shape[1],
        H=H,
        K=K,
        V=V,
        E=E,
        BT=BT,
        BK=BK,
        BV=triton.next_power_of_2(V),
        num_warps=8,
    )

    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta, dtype=torch.float32)
    dR = q.new_empty(B, T, H, BT, dtype=torch.float32)
    dRq = torch.empty_like(dR)
    chunk_triadic_gdn_bwd_kernel_intra[(NT, B * H)](
        q=q,
        k=k,
        v=v,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        u=u,
        do=do,
        dx=dx,
        dq=dq,
        dk=dk,
        dv=dv,
        dbeta=dbeta,
        dR=dR,
        dRq=dRq,
        ks=ks,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        TG=B * T,
        H=H,
        K=K,
        V=V,
        E=E,
        NK=NK,
        BT=BT,
        BC=16,
        BK=BK,
        BV=min(64, triton.next_power_of_2(V)),
        BE=max(16, triton.next_power_of_2(E)),
        num_warps=8,
        num_stages=1,
    )

    dk2 = torch.empty_like(k2, dtype=torch.float32)
    dq2 = torch.empty_like(q2, dtype=torch.float32)
    dg = torch.empty_like(g, dtype=torch.float32)
    chunk_triadic_gdn_bwd_kernel_gate[(E, NT, B * H)](
        q2=q2,
        k2=k2,
        g=g,
        beta=beta,
        dR=dR,
        dRq=dRq,
        qs=qs,
        ks=ks,
        kd=kd,
        ss=ss,
        dq2=dq2,
        dk2=dk2,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        TG=B * T,
        NTG=ss.shape[1],
        H=H,
        E=E,
        NK=NK,
        BT=BT,
        num_warps=4,
    )
    return dq, dk, dv, dk2, dq2, dg, dbeta, dh0
