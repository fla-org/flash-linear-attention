# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, solve_tril
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_inter(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H*K + o_k, mask=m_k, other=0)
        # [BC, BK]
        p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1))
        # [BC, BC]
        b_ktg = b_kt * exp(b_gn[:, None] - b_gk)
        b_Akk += tl.dot(b_k, b_ktg)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)

    b_Akk *= b_b[:, None]

    p_Akk = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BK", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_intra(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    m_A = (i_t * BT + i_i * BC + o_i) < T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32) * 1.44269504
    b_beta = tl.load(beta + (i_t * BT + i_i * BC + o_i) * H, mask=m_A, other=0)

    # Pre-compute masks for all steps
    o_i = tl.arange(0, BC)
    
    # Accumulators
    acc_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    acc_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    # Add diagonal
    b_Aqk_diag = tl.sum(b_q * b_k, 1)
    acc_Aqk = tl.where(o_i[:, None] == o_i[None, :], b_Aqk_diag[:, None], acc_Aqk)

    # Iterate from large spans down to small spans
    # For BC=64, we need to handle span=32 (log2=5). 
    # Starting from 6 is safe for BC up to 128.
    for log_span in range(3, -1, -1):
        span = 1 << log_span
        # Identify Q and K rows for this span
        # For a block size of 2*span:
        # Top half (0..span-1) are Keys
        # Bottom half (span..2*span-1) are Queries
        # Pivot is at index 'span' relative to block start.
        
        # Global index within chunk is o_i
        # Relative index in 2*span block: o_i % (2*span)
        # Is Query if relative >= span
        is_q = (o_i % (2*span)) >= span
        is_k = (o_i % (2*span)) < span
 
        # Pivot index for each row
        # The pivot is the start of the Q-half of the block (i.e., index `span` relative to block start)
        # pivot = (o_i // (2*span)) * (2*span) + span
        pivot_idx = (o_i // (2*span)) * (2*span) + span - 1

        # Gather g_pivot from b_g using matrix multiplication (permutation)
        # S[i, j] = 1 if j == pivot_idx[i]
        S = ((o_i[None, :] == pivot_idx[:, None])).to(tl.float32)
        b_g_pivot = tl.dot(S, b_g)
        
        mask_i = m_A[:, None]        
        mask_q = is_q[:, None] & mask_i
        mask_k = is_k[:, None] & mask_i

        d_q = tl.where(mask_q, tl.exp2(b_g - b_g_pivot), 0.0)
        d_k = tl.where(mask_k, tl.exp2(b_g_pivot - b_g), 0.0)
        
        # Mask inputs
        b_q_masked = b_q * d_q
        b_k_masked = b_k * d_k
        b_k_q_masked = b_k * d_q

        b_Aqk_sub = tl.dot(b_q_masked, tl.trans(b_k_masked))
        b_Akk_sub = tl.dot(b_k_q_masked, tl.trans(b_k_masked))
        
        # Filter cross-block terms
        block_id = o_i // (2*span)
        same_block = block_id[:, None] == block_id[None, :]
        acc_Aqk += tl.where(same_block, b_Aqk_sub, 0.0)
        acc_Akk += tl.where(same_block, b_Akk_sub, 0.0)
    
    acc_Aqk = tl.where(o_i[:, None] >= o_i[None, :], acc_Aqk * scale, 0.0)
    acc_Akk = tl.where(o_i[:, None] > o_i[None, :], acc_Akk * b_beta[:, None].to(tl.float32), 0.0)
    # Store final results
    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_i * BC), (BC, BC), (1, 0))
    tl.store(p_Aqk, acc_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, acc_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    db += (i_k * all + bos) * H + i_h

    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + (i_t * BT + i_i * BC) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_dq2 *= exp(b_g - b_gn[None, :])
        b_dk2 *= exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + o_i) < T
    o_dA = (i_t * BT + i_i * BC + o_i) * H*BT + i_i * BC
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC]
        b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
        b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
        # [BK]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * exp(b_g - b_gkj[None, :]), 0.)
        b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * exp(b_g - b_gkj[None, :]), 0.)

        p_kj += H*K
        p_gkj += H*K
    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))

    b_dg = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC]
            b_b = tl.load(p_b, boundary_check=(0,))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_kb = tl.load(p_k, boundary_check=(0, 1)) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_qg = b_q * tl.where(m_j[:, None], exp(b_gk - b_gn[None, :]), 0)
            b_kbg = b_kb * tl.where(m_j[:, None], exp(b_gk - b_gn[None, :]), 0)
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp(b_gn[None, :] - b_g)
    o_dA = (i_t * BT + i_i * BC) * H*BT + i_i * BC + o_i
    p_qj = q + (i_t * BT + i_i * BC) * H*K + o_k
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k
    p_bj = beta + (i_t * BT + i_i * BC) * H

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dAqk = tl.load(dAqk + o_dA + j * H*BT)
        b_dAkk = tl.load(dAkk + o_dA + j * H*BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * exp(b_gkj[None, :] - b_g), 0.)
        b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * exp(b_gkj[None, :] - b_g), 0.)

        p_qj += H*K
        p_kj += H*K
        p_gkj += H*K
        p_bj += H
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_dg += (b_dk2 - b_dkt) * b_k
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BK", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_intra_recurrent(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (i_t * BT + i_i * BC + o_i) * H*BT + i_i * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    b_k = b_k * tl.load(beta + (i_t * BT + i_i * BC + o_i) * H, mask=m_A, other=0)[:, None]

    p_kt = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gk = g + (i_t * BT + i_i * BC) * H*K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_ktg = b_kt[None, :] * exp(b_g - b_gk[None, :])
        b_Aqk = tl.sum(b_q * b_ktg, 1) * scale
        b_Akk = tl.sum(b_k * b_ktg, 1)
        tl.store(Aqk + o_A + j, b_Aqk, mask=m_A)
        tl.store(Akk + o_A + j, b_Akk, mask=m_A)
        p_kt += H*K
        p_gk += H*K

    tl.debug_barrier()
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(Aqk + o_A[:, None] + o_i[None, :], b_A, mask=m_A[:, None] & (o_i[:, None] < o_i[None, :]))
    tl.store(Akk + o_A[:, None] + o_i[None, :], b_A, mask=m_A[:, None] & (o_i[:, None] <= o_i[None, :]))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]  # Let autotune choose freely
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'B'])
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Each block processes one token (i) for BH heads
    i_tg = tl.program_id(0)  # global token index
    i_hg = tl.program_id(1)  # head_group index

    i_h_start = i_hg * BH

    if IS_VARLEN:
        # Binary search to find which sequence this token belongs to
        # i_tg is the global token index
        # Range [0, B) where B is num_sequences passed from python

        left = 0
        right = B
        i_n = 0

        # Unrolled binary search (max B=2^32)
        # We can limit iterations based on expected max batch size if needed
        # 20 iterations covers B=1M, usually enough
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                end_val = tl.load(cu_seqlens + mid + 1).to(tl.int32)
                if i_tg < end_val:
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        i_t = i_tg - bos
        T = eos - bos # Current sequence length

        # Safety check
        if i_t >= T or i_tg >= eos:
            return

    else:
        i_b = i_tg // T
        i_t = i_tg % T
        bos = i_b * T

        if i_t >= T:
            return

    # Find which sub-chunk (BC=16) this token belongs to
    BC: tl.constexpr = 16
    i_chunk = i_t // BT  # which BT=64 chunk
    i_subchunk = (i_t % BT) // BC  # which BC=16 sub-chunk within the BT chunk

    subchunk_start = i_chunk * BT + i_subchunk * BC
    subchunk_end = tl.minimum(subchunk_start + BC, T)

    o_h = tl.arange(0, BH)
    m_h = (i_h_start + o_h) < H

    # Marginalize over entire K dimension at once
    BK: tl.constexpr = triton.next_power_of_2(K)
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    # Load q[i_t, h:h+BH, :] - shape [BH, K]
    # For varlen, we use global offset: bos + i_t = i_tg
    p_q = tl.make_block_ptr(q + (bos + i_t) * H * K, (H, K), (K, 1),
                            (i_h_start, 0), (BH, BK), (0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

    # Load g[i_t, h:h+BH, :]
    p_g = tl.make_block_ptr(g + (bos + i_t) * H * K, (H, K), (K, 1),
                            (i_h_start, 0), (BH, BK), (0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

    # Load k[i_t, h:h+BH, :] and beta[i_t, h:h+BH]
    p_k = tl.make_block_ptr(k + (bos + i_t) * H * K, (H, K), (K, 1),
                            (i_h_start, 0), (BH, BK), (0, 1))
    b_k_self = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

    p_beta = beta + (bos + i_t) * H + i_h_start + o_h
    b_beta = tl.load(p_beta, mask=m_h, other=0).to(tl.float32)  # [BH]
    b_k_self = b_k_self * b_beta[:, None]  # [BH, K]

    for j in range(subchunk_start, tl.minimum(i_t + 1, subchunk_end)):

        # Load k[j, h:h+BH, :] with pointer arithmetic
        p_k_j = tl.make_block_ptr(k + (bos + j) * H * K, (H, K), (K, 1),
                                  (i_h_start, 0), (BH, BK), (0, 1))
        b_k_j = tl.load(p_k_j, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

        # Load g[j, h:h+BH, :]
        p_g_j = tl.make_block_ptr(g + (bos + j) * H * K, (H, K), (K, 1),
                                  (i_h_start, 0), (BH, BK), (0, 1))
        b_g_j = tl.load(p_g_j, boundary_check=(0, 1)).to(tl.float32)  # [BH, BK]

        # Compute gated key for all BH heads: [BH, BK]
        if USE_EXP2:
            b_k_j_gated = b_k_j * tl.exp2(b_g - b_g_j)
        else:
            b_k_j_gated = b_k_j * exp(b_g - b_g_j)

        # Apply mask for valid K dimension
        b_k_j_gated = tl.where(m_k[None, :], b_k_j_gated, 0.0)

        # Compute Aqk and Akk for all BH heads: [BH]
        b_Aqk = tl.sum(b_q * b_k_j_gated, axis=1) * scale  # [BH]
        # Akk: only accumulate if j < i_t
        b_Akk = tl.sum(b_k_self * b_k_j_gated, axis=1) * tl.where(j < i_t, 1.0, 0.0)  # [BH]

        # Store with [B, T, H, BT] layout (no transpose needed later)
        j_pos = j % BT
        offs_h = i_h_start + o_h
        offs_out = (bos + i_t) * H * BT + offs_h * BT + j_pos
        tl.store(Aqk + offs_out, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + offs_out, b_Akk.to(Akk.dtype.element_ty), mask=m_h)


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    use_exp2: bool = False,
) -> None:
    B, T, H, K = q.shape
    BT = chunk_size

    # Grid: (total_tokens, H/BH) - each token gets its own block
    if cu_seqlens is not None:
        total_tokens = q.shape[1]
        # Use num_sequences as B for binary search
        B_kernel = len(cu_seqlens) - 1
    else:
        total_tokens = B * T
        B_kernel = B

    def grid(meta):
        BH = meta['BH']
        return (total_tokens, triton.cdiv(H, BH))

    chunk_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        B=B_kernel,
        T=T,
        H=H,
        K=K,
        BT=BT,
        USE_EXP2=use_exp2,
    )


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float32,
    impl_type: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            The query tensor of shape `[B, T, H, K]`.
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`. Default: `None`.
        scale (Optional[float]):
            The scale factor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`
        impl_type (str):
            The implementation type for sub_intra kernel. 
            Options: "auto", "token", "recursive", "recurrent".
            Default: "auto".

    Returns:
        Aqk (torch.Tensor):
            The intra Aqk tensor of shape `[B, T, H, BT]` where `BT` is the chunk size.
        Akk (torch.Tensor):
            The intra Akk tensor of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    assert K <= 256
    
    if impl_type == "auto":
        impl_type = "token" if K >= 128 else "recursive"
            
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = 16
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    Akk = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = (NT, NC * NC, B * H)

    chunk_kda_fwd_kernel_intra_sub_inter[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )

    if impl_type == "token":
        # Token-parallel implementation for sub_intra (each token gets its own block)
        chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
        )
    elif impl_type == "recurrent":
        grid = (NT, NC, B * H)
        chunk_kda_fwd_kernel_intra_sub_intra_recurrent[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=16,
            BK=BK,
        )
    else:
        # Original sub-chunk based implementation
        grid = (NT, NC, B * H)
        chunk_kda_fwd_kernel_intra_sub_intra[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=16,
            BK=BK,
        )

    Akk = solve_tril(
        A=Akk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    return Aqk, Akk


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # NC = 4
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_kda_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = chunk_local_cumsum(
        dg2.add_(dg),
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    return dq, dk2, db, dg
