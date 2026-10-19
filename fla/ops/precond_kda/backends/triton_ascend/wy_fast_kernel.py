# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Migrated from fla/ops/precond_kda/wy_fast.py for triton-ascend: kernel bodies are
# verbatim copies; the @triton.autotune spaces keep only the tile axes because
# num_warps/num_stages are unsupported on this backend (each pruned candidate was
# one full aarch64 compile, ~9x the necessary cold-compile cost).

"""Precond-KDA WY-recompute kernel adapted for triton-ascend on Ascend NPU."""

import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    q,
    k,
    k_precond,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr = 'tf32x3',
):
    """
    Asymmetric WY representation:
    - w = A @ (k * beta * exp(gk)) - uses original k for read/correction
    - kg = k_precond * exp(gn - gk) - uses k_precond for write/h update
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_b = tl.load(beta + bos*H + i_h + o_t*H, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    p_A = A + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)

    # u computation (unchanged)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_u = u + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(u.dtype.element_ty), mask=m_tv)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_w = w + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        # Use original k for w (read/correction)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kb = b_k * b_b[:, None]

        p_gk = gk + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_gk = tl.load(p_gk, mask=m_tk, other=0.0)
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = q + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
            p_qg = qg + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_tk, other=0.0)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(qg.dtype.element_ty), mask=m_tk)

        if STORE_KG:
            # Use k_precond for kg (write/h update) - ASYMMETRIC
            p_kp = k_precond + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
            b_kp = tl.load(p_kp, mask=m_tk, other=0.0)

            last_idx = min(i_t * BT + BT, T) - 1
            b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.)
            # kg uses k_precond, not original k
            b_kg = b_kp * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = kg + (bos * H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
            tl.store(p_kg, b_kg.to(kg.dtype.element_ty), mask=m_tk)

        # w uses original k
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(w.dtype.element_ty), mask=m_tk)
