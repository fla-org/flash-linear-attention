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
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
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


def recompute_w_u_fwd(
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    q: torch.Tensor | None = None,
    output_qg: bool = False,
    output_kg: bool = True,
):
    """
    Compute WY representation for preconditioned KDA.

    Args:
        k: [B, T, H, K] - original k (for w computation)
        k_precond: [B, T, H, K] - preconditioned k (for kg computation)
        v: [B, T, H, V] - values
        beta: [B, T, H] - beta scaling
        A: [B, T, H, BT] - inverse of Akk
        gk: [B, T, H, K] - cumsum of gates

    Returns:
        w: [B, T, H, K] - A @ (k * beta * exp(gk))
        u: [B, T, H, V] - A @ (v * beta)
        qg: [B, T, H, K] or None - q * exp(gk)
        kg: [B, T, H, K] or None - k_precond * exp(gn - gk) for h update
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if output_qg and q is not None else None
    kg = torch.empty_like(k_precond) if output_kg else None

    grid = (NT, B * H)
    recompute_w_u_fwd_kernel[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg
