# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp, gather
from fla.utils import is_amd, is_gather_supported, use_cuda_graph

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'BT'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    a,
    b,
    gi,
    ge,
    qg,
    kg,
    ag,
    bg,
    Aqk,
    Aqb,
    Aab,
    Aak,
    cu_seqlens,
    chunk_indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr
):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + tl.arange(0, BC)) < T
    last_idx = min((i_t+1) * BT, T) - 1
    o_A = (bos + i_t * BT + tl.arange(0, BC)) * H*BT + i_h * BT
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_ge = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_g_last = gi + (bos * H + i_h) * K + last_idx * H * K + tl.arange(0, BK)
    b_g_last = tl.load(p_g_last, mask=m_k, other=0)
    p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))
    p_bg = tl.make_block_ptr(bg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)

    # deal with decay term.
    g_exp = exp(b_gi)
    g_exp_inv = exp(-b_gi + b_g_last[None, :])
    b_qg = b_q * g_exp
    b_kg = b_k * g_exp_inv
    b_bg = b_b * g_exp_inv
    b_ag = b_a * exp(b_ge)
    tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_bg, b_bg.to(p_bg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_ag, b_ag.to(p_ag.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    # tl.debug_barrier()

    b_q = b_q.to(b_k.dtype)
    # inner attn
    for j in range(0, min(BC, T - i_t * BT)):
        # a trick to index the j-th row of b_k, b_g, b_b
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BK], j, dtype=tl.int16)
            # [1, BK]
            b_k_j = gather(b_k, row_idx, axis=0)
            b_gk_j = gather(b_gi, row_idx, axis=0)
            b_b_j = gather(b_b, row_idx, axis=0)
        else:
            mask = tl.arange(0, BC) == j
            b_k_j = tl.sum(tl.where(mask[:, None], b_k, 0), 0)[None, :]
            b_gk_j = tl.sum(tl.where(mask[:, None], b_gi, 0), 0)[None, :]
            b_b_j = tl.sum(tl.where(mask[:, None], b_b, 0), 0)[None, :]
        tmp = exp(b_gi - b_gk_j)
        b_A_qk = tl.sum(b_q * b_k_j * tmp, 1)
        m_i = (o_i >= j).to(tl.float32)
        b_A_qk = b_A_qk * m_i
        b_A_qb = tl.sum(b_q * b_b_j * tmp, 1)
        b_A_qb = b_A_qb * m_i
        tmp2 = exp(b_ge - b_gk_j)
        b_A_ak = tl.sum(b_a * b_k_j * tmp2, 1)
        m_i2 = (o_i > j).to(tl.float32)
        b_A_ak = b_A_ak * m_i2
        b_A_ab = tl.sum(b_a * b_b_j * tmp2, 1)
        b_A_ab = b_A_ab * m_i2

        tl.store(Aqk + o_A + j, b_A_qk.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        tl.store(Aqb + o_A + j, b_A_qb.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        tl.store(Aab + o_A + j, b_A_ab.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)
        tl.store(Aak + o_A + j, b_A_ak.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_A)


def chunk_dplr_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aqb = q.new_empty(B, T, H, BT, dtype=q.dtype)
    # involving matrix inverse and it'd be better to use float here.
    Aab = q.new_empty(B, T, H, BT, dtype=torch.float)
    Aak = q.new_empty(B, T, H, BT, dtype=torch.float)

    grid = (NT, B, H)
    BK = triton.next_power_of_2(K)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    chunk_dplr_fwd_A_kernel_intra_sub_intra[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        Aqk=Aqk,
        Aqb=Aqb,
        Aab=Aab,
        Aak=Aak,
        qg=qg,
        kg=kg,
        ag=ag,
        bg=bg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=is_gather_supported
    )
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg
