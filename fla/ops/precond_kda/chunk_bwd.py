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
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


# ==============================================================================
# dAv kernel: compute dA and dv from inter-chunk backward
# Matches KDA's chunk_kda_bwd_kernel_dAv
# ==============================================================================

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
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_bwd_kernel_dAv(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_r = tl.arange(0, BT)
    p_A = A + (bos * H + i_h) * BT + o_r[:, None] + o_t[None, :] * (H*BT)
    b_A = tl.load(p_A, mask=(o_r[:, None] < BT) & m_t[None, :], other=0.0)

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        # [BV, BT] transposed
        p_v = v + o_v[:, None] + o_t[None, :] * (H*V)
        p_do = do + o_t[:, None] * (H*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v[:, None] & m_t[None, :], other=0.0)
        # [BT, BV]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        # [BT, BT]
        b_dA += tl.dot(b_do, b_v)
        # [BT, BV]
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_tv)

    o_A = tl.arange(0, BT)
    p_dA = dA + o_t[:, None] * (H*BT) + o_A[None, :]
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))


def chunk_precond_kda_bwd_dAv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem():
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_precond_kda_bwd_kernel_dAv[grid](
        q=q,
        k=k,
        v=v,
        A=A,
        do=do,
        dv=dv,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dA, dv


# ==============================================================================
# WY + inter-chunk backward kernel: compute dq, dk, dkg, dv, db, dg, dA
# Matches KDA's chunk_kda_bwd_kernel_wy_dqkg_fused with k/k_precond asymmetry
# ==============================================================================

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
    ],
    key=['BT', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_bwd_kernel_wy_dqkg(
    q,
    k,           # original k (for WY backward)
    k_precond,   # preconditioned k (for inter backward)
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,          # dk from WY backward (original k)
    dkg,         # dkg from inter backward (k_precond)
    dv,
    dv2,
    dg,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    k_precond += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dkg += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += bos * H + i_h
    dA += (bos * H + i_h) * BT

    b_beta = tl.load(beta + o_t*H, mask=m_t, other=0.0)

    o_r = tl.arange(0, BT)
    p_A = A + o_r[:, None] + o_t[None, :] * (H * BT)
    b_A = tl.load(p_A, mask=(o_r[:, None] < BT) & m_t[None, :], other=0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        m_tk = m_t[:, None] & m_k[None, :]
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_kp = k_precond + o_t[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kp = tl.load(p_kp, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkg_raw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_tv = m_t[:, None] & m_v[None, :]
            m_vk = m_v[:, None] & m_k[None, :]
            p_v_new = v_new + o_t[:, None] * (H*V) + o_v[None, :]
            p_do = do + o_t[:, None] * (H*V) + o_v[None, :]
            if TRANSPOSE_STATE:
                p_h = h + o_v[:, None] * K + o_k[None, :]
                p_dh = dh + o_v[:, None] * K + o_k[None, :]
            else:
                p_h = h + o_v[:, None] + o_k[None, :] * V
                p_dh = dh + o_v[:, None] + o_k[None, :] * V
            p_dv = dv + o_t[:, None] * (H*V) + o_v[None, :]
            # [BT, BV]
            b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            # [BV, BK]
            b_h = tl.load(p_h, mask=m_vk, other=0.0)
            b_dh = tl.load(p_dh, mask=m_vk, other=0.0)
            # [BT, BV]
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dkg_raw += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))
            tl.debug_barrier()  # DO NOT REMOVE THIS LINE!
            if i_k == 0:
                p_v = v + o_t[:, None] * (H*V) + o_v[None, :]
                p_dv2 = dv2 + o_t[:, None] * (H*V) + o_v[None, :]

                b_v = tl.load(p_v, mask=m_tv, other=0.0)

                b_dA += tl.dot(b_dv, tl.trans(b_v))

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)

                tl.store(p_dv2, b_dv2.to(dv2.dtype.element_ty), mask=m_tv)

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale

        # WY backward: uses original k
        b_kg_orig = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg_orig.to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg_orig, 1)

        # dk from WY backward (original k)
        b_dk = b_dkgb * b_gb

        # Inter backward: uses k_precond
        b_gn_g = tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)
        b_dkg = b_dkg_raw * b_gn_g

        b_kg_precond = b_kp * b_gn_g
        b_kp_dkg = b_kg_precond * b_dkg_raw
        b_dgk += tl.sum(b_kp_dkg, axis=0)

        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)

        b_dg = (b_q * b_dq                           # inter query
                - b_kp_dkg                              # inter write key
                + m_last[:, None] * b_dgk               # accumulated last position
                + b_kg_orig * b_dkgb * b_beta[:, None])  # WY backward

        p_dq = dq + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (H*K) + o_k[None, :]
        p_dkg = dkg + o_t[:, None] * (H*K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (H*K) + o_k[None, :]
        tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dkg, b_dkg.to(dkg.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(dg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    o_A = tl.arange(0, BT)
    p_dA = dA + o_t[:, None] * (H * BT) + o_A[None, :]
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))
    tl.store(db + o_t*H, b_db.to(db.dtype.element_ty), mask=m_t)


def chunk_precond_kda_bwd_wy_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    """
    Inter-chunk + WY backward for preconditioned KDA.

    Matches KDA's chunk_kda_bwd_wy_dqkg_fused but with k/k_precond asymmetry:
    - WY backward uses original k (for w = A @ (k * beta * exp(gk)))
    - Inter backward uses k_precond (for kg = k_precond * exp(gn - gk))

    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - original k (for WY backward)
        k_precond: [B, T, H, K] - preconditioned k (for inter backward)
        v: [B, T, H, V] - original v (for WY backward)
        v_new: [B, T, H, V] - corrected v (for inter backward)
        g: [B, T, H, K] - gate cumsum (log2 space)
        beta: [B, T, H] - beta scaling
        A: [B, T, H, BT] - WY inverse matrix
        h: [NT, H, K, V] - per-chunk hidden states
        do: [B, T, H, V] - output gradient
        dh: [NT, H, K, V] - hidden state gradient
        dv: [B, T, H, V] - dv from h backward (= du for WY)

    Returns:
        dq, dk, dkg, dv, db, dg, dA
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dkg = torch.empty_like(k_precond, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * H)
    chunk_precond_kda_bwd_kernel_wy_dqkg[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dkg=dkg,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    dv = dv2
    return dq, dk, dkg, dv, db, dg, dA
