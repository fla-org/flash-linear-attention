
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BK", "NC", "BT", "K"],
)
@triton.jit
def chunk_dplr_bwd_kernel_intra(
    q,
    k,
    a,
    b,
    gi,
    ge,
    dAqk,
    dAqb,
    dAak,
    dAab,
    dq,
    dk,
    da,
    db,
    dqg,
    dkg,
    dag,
    dbg,
    dgk,
    dgk_offset,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    # offset calculation
    ge += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    gi += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    q += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    a += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    b += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    k += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dq += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dk += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    da += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    db += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dqg += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dag += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dkg += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dbg += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dgk += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dgk_offset += i_bh * T * K if HEAD_FIRST else (bos*H + i_h) * K
    dAqk += i_bh * T * BT if HEAD_FIRST else (bos*H + i_h) * BT
    dAqb += i_bh * T * BT if HEAD_FIRST else (bos*H + i_h) * BT
    dAak += i_bh * T * BT if HEAD_FIRST else (bos*H + i_h) * BT
    dAab += i_bh * T * BT if HEAD_FIRST else (bos*H + i_h) * BT

    stride_qk = K if HEAD_FIRST else H*K
    stride_A = BT if HEAD_FIRST else H*BT

    p_ge = tl.make_block_ptr(ge, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_da = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    b_db = tl.zeros([BC, BK], dtype=tl.float32)
    # intra chunk gradient calculation
    p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (stride_A, 1), (i_t*BT + i_i*BC, i_i*BC), (BC, BC), (1, 0))
    p_dAab = tl.make_block_ptr(dAab, (T, BT), (stride_A, 1), (i_t*BT + i_i*BC, i_i*BC), (BC, BC), (1, 0))
    p_dAqb = tl.make_block_ptr(dAqb, (T, BT), (stride_A, 1), (i_t*BT + i_i*BC, i_i*BC), (BC, BC), (1, 0))
    p_dAak = tl.make_block_ptr(dAak, (T, BT), (stride_A, 1), (i_t*BT + i_i*BC, i_i*BC), (BC, BC), (1, 0))
    o_i = tl.arange(0, BC)
    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t*BT + i_i*BC, i_k*BK), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (stride_qk, 1), (i_t*BT + i_i*BC, i_k*BK), (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a, (T, K), (stride_qk, 1), (i_t*BT + i_i*BC, i_k*BK), (BC, BK), (1, 0))
    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t*BT + i_i*BC, i_k*BK), (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_a = tl.load(p_a, boundary_check=(0, 1)).to(tl.float32)
    b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
    b_dAab = tl.load(p_dAab, boundary_check=(0, 1)).to(tl.float32)
    b_dAqb = tl.load(p_dAqb, boundary_check=(0, 1)).to(tl.float32)
    b_dAak = tl.load(p_dAak, boundary_check=(0, 1)).to(tl.float32)

    # inter chunk gradient calculation
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    if i_i > 0:
        p_gn = gi + (i_t * BT + i_i * BC - 1) * stride_qk + o_k
        p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BK,]
        for i_j in range(0, i_i):
            p_kj = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_bj = tl.make_block_ptr(b, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gkj = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqikj = tl.make_block_ptr(dAqk, (T, BT), (stride_A, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            p_dAaibj = tl.make_block_ptr(dAab, (T, BT), (stride_A, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            p_dAqibj = tl.make_block_ptr(dAqb, (T, BT), (stride_A, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            p_dAaikj = tl.make_block_ptr(dAak, (T, BT), (stride_A, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_kj = tl.load(p_kj, boundary_check=(0, 1))
            b_bj = tl.load(p_bj, boundary_check=(0, 1))
            b_gkj = tl.load(p_gkj, boundary_check=(0, 1))
            tmp = tl.exp(b_gn[None, :] - b_gkj)
            b_kjg = b_kj * tmp
            b_bjg = b_bj * tmp
            # [BC, BC]
            b_dAqikj = tl.load(p_dAqikj, boundary_check=(0, 1))
            b_dAaibj = tl.load(p_dAaibj, boundary_check=(0, 1))
            b_dAqibj = tl.load(p_dAqibj, boundary_check=(0, 1))
            b_dAaikj = tl.load(p_dAaikj, boundary_check=(0, 1))
            # [BC, BK]
            b_dq += tl.dot(b_dAqikj, b_kjg)
            b_dq += tl.dot(b_dAqibj, b_bjg)
            # [BC, BC]
            b_da += tl.dot(b_dAaibj, b_bjg)
            b_da += tl.dot(b_dAaikj, b_kjg)
        b_dq *= tl.exp(b_gi - b_gn[None, :])
        b_da *= tl.exp(b_ge - b_gn[None, :])

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = gi + (min(i_t * BT + i_i * BC + BC, T) - 1)*stride_qk + o_k
        p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            m_j = (i_t * BT + i_j * BC + tl.arange(0, BC)) < T
            p_qj = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_aj = tl.make_block_ptr(a, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gij = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gej = tl.make_block_ptr(ge, (T, K), (stride_qk, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqjki = tl.make_block_ptr(dAqk, (BT, T), (1, stride_A), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            p_dAajbi = tl.make_block_ptr(dAab, (BT, T), (1, stride_A), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            p_dAqjbi = tl.make_block_ptr(dAqb, (BT, T), (1, stride_A), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            p_dAajki = tl.make_block_ptr(dAak, (BT, T), (1, stride_A), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            b_qj = tl.load(p_qj, boundary_check=(0, 1))
            b_aj = tl.load(p_aj, boundary_check=(0, 1))
            b_gij = tl.load(p_gij, boundary_check=(0, 1))
            b_gej = tl.load(p_gej, boundary_check=(0, 1))
            b_gij = tl.where(m_j[:, None] & m_k, b_gij, float('-inf'))
            b_gej = tl.where(m_j[:, None] & m_k, b_gej, float('-inf'))
            b_qjg = b_qj * tl.exp(b_gij - b_gn[None, :])
            b_ajg = b_aj * tl.exp(b_gej - b_gn[None, :])
            # [BC, BC]
            b_dAqjki = tl.load(p_dAqjki, boundary_check=(0, 1))
            b_dAajbi = tl.load(p_dAajbi, boundary_check=(0, 1))
            b_dAqjbi = tl.load(p_dAqjbi, boundary_check=(0, 1))
            b_dAajki = tl.load(p_dAajki, boundary_check=(0, 1))
            b_dk += tl.dot(b_dAqjki, b_qjg)
            b_dk += tl.dot(b_dAajki, b_ajg)
            b_db += tl.dot(b_dAqjbi, b_qjg)
            b_db += tl.dot(b_dAajbi, b_ajg)
        tmp = tl.exp(b_gn[None, :] - b_gi)
        b_dk *= tmp
        b_db *= tmp

    # intra chunk gradient calculation
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # trick to index the block
        mask_idx = tl.arange(0, BC) == j
        b_gij = tl.sum(tl.where(mask_idx[:, None], b_gi, 0), 0)
        b_gej = tl.sum(tl.where(mask_idx[:, None], b_ge, 0), 0)
        b_kj = tl.sum(tl.where(mask_idx[:, None], b_k, 0), 0)
        b_bj = tl.sum(tl.where(mask_idx[:, None], b_b, 0), 0)
        b_dAqk_j = tl.sum(tl.where(mask_idx[None, :], b_dAqk, 0), 1)
        b_dAab_j = tl.sum(tl.where(mask_idx[None, :], b_dAab, 0), 1)
        b_dAqb_j = tl.sum(tl.where(mask_idx[None, :], b_dAqb, 0), 1)
        b_dAak_j = tl.sum(tl.where(mask_idx[None, :], b_dAak, 0), 1)
        m_e = o_i[:, None] > j
        m_i = o_i[:, None] >= j
        tmp1 = tl.exp(b_gi - b_gij[None, :])
        tmp2 = tl.exp(b_ge - b_gij[None, :])
        b_dq += tl.where(m_i, b_dAqk_j[:, None] * b_kj[None, :] * tmp1, 0.)
        b_dq += tl.where(m_i, b_dAqb_j[:, None] * b_bj[None, :] * tmp1, 0.)
        b_da += tl.where(m_e, b_dAab_j[:, None] * b_bj[None, :] * tmp2, 0.)
        b_da += tl.where(m_e, b_dAak_j[:, None] * b_kj[None, :] * tmp2, 0.)
        b_dA_qk_j = tl.sum(tl.where(mask_idx[:, None], b_dAqk, 0), 0)
        b_dA_ab_j = tl.sum(tl.where(mask_idx[:, None], b_dAab, 0), 0)
        b_dA_qb_j = tl.sum(tl.where(mask_idx[:, None], b_dAqb, 0), 0)
        b_dA_ak_j = tl.sum(tl.where(mask_idx[:, None], b_dAak, 0), 0)
        b_qj = tl.sum(tl.where(mask_idx[:, None], b_q, 0), 0)
        b_aj = tl.sum(tl.where(mask_idx[:, None], b_a, 0), 0)
        m_i = o_i[:, None] <= j
        m_e = o_i[:, None] < j
        tmp1 = tl.exp(b_gij[None, :] - b_gi)
        tmp2 = tl.exp(b_gej[None, :] - b_gi)
        b_dk += tl.where(m_i, b_dA_qk_j[:, None] * b_qj[None, :] * tmp1, 0.)
        b_dk += tl.where(m_e, b_dA_ak_j[:, None] * b_aj[None, :] * tmp2, 0.)
        b_db += tl.where(m_i, b_dA_qb_j[:, None] * b_qj[None, :] * tmp1, 0.)
        b_db += tl.where(m_e, b_dA_ab_j[:, None] * b_aj[None, :] * tmp2, 0.)
    # post processing
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_da = tl.make_block_ptr(da, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dqg = tl.make_block_ptr(dqg, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dkg = tl.make_block_ptr(dkg, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dag = tl.make_block_ptr(dag, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dbg = tl.make_block_ptr(dbg, (T, K), (stride_qk, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = gi + (min(i_t * BT + BT, T) - 1)*stride_qk + o_k
    p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_da += tl.load(p_dag, boundary_check=(0, 1)) * tl.exp(b_ge)
    b_dq += tl.load(p_dqg, boundary_check=(0, 1)) * tl.exp(b_gi) * scale
    tmp = tl.exp(b_gn[None, :] - b_gi)
    b_dk += tl.load(p_dkg, boundary_check=(0, 1)) * tmp
    b_db += tl.load(p_dbg, boundary_check=(0, 1)) * tmp
    tl.store(p_dq, (b_dq).to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_da, b_da.to(p_da.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0, 1))
    b_dgk = b_dq * b_q + b_da * b_a - b_dk * b_k - b_db * b_b
    b_dgk_offset = b_da * b_a
    tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dgk_offset, b_dgk_offset.to(p_dgk_offset.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
        for BK in [32, 64]
    ],
    key=["BK", "BT", "K"],
)
@triton.jit
def chunk_dplr_bwd_dgk_kernel(
    dgk,
    dgk_offset,
    dgk_last,
    dgk_output,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    stride_qk = K if HEAD_FIRST else H * K
    dgk += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    dgk_offset += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    dgk_last += ((i_bh * NT + i_t) * K) if HEAD_FIRST else (i_tg * H + i_h) * K
    dgk_output += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    p_dgk_last = dgk_last + tl.arange(0, BK) + i_k * BK
    m_k = tl.arange(0, BK) + i_k * BK < K
    b_dgk_last = tl.load(p_dgk_last, mask=m_k, other=0)
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
    b_dgk_offset = tl.load(p_dgk_offset, boundary_check=(0, 1))
    # m_inv_cumsum = (tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]).to(tl.float32)
    # b_dgk_cumsum = tl.dot(m_inv_cumsum, b_dgk, allow_tf32=False)
    b_dgk_cumsum = tl.cumsum(b_dgk, 0, reverse=True)
    b_dgk_cumsum += b_dgk_last[None, :]
    b_dgk_cumsum -= b_dgk_offset
    p_dgk_output = tl.make_block_ptr(dgk_output, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dgk_output, b_dgk_cumsum.to(p_dgk_output.dtype.element_ty), boundary_check=(0, 1))


def chunk_dplr_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dAqk: torch.Tensor,
    dAqb: torch.Tensor,
    dAak: torch.Tensor,
    dAab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    scale: float = 1.0,
    chunk_size: int = 64,
):
    if head_first:
        B, H, T, K = q.shape
    else:
        B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float)
    dgk_offset = torch.empty_like(gi, dtype=torch.float)

    grid = (NK, NT * NC, B * H)
    chunk_dplr_bwd_kernel_intra[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        dAqk=dAqk,
        dAqb=dAqb,
        dAak=dAak,
        dAab=dAab,
        dq=dq,
        dk=dk,
        dgk=dgk,
        dgk_offset=dgk_offset,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        da=da,
        db=db,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        HEAD_FIRST=head_first
    )

    def grid2(meta): return (NT, triton.cdiv(K, meta['BK']), B * H)
    dgk_output = torch.empty_like(dgk)

    chunk_dplr_bwd_dgk_kernel[grid2](
        dgk=dgk,
        dgk_offset=dgk_offset,
        dgk_last=dgk_last,
        dgk_output=dgk_output,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return dq, dk, da, db, dgk_output
