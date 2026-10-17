# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.diag_kdn.wy_fast import recompute_w_u_fwd
from fla.ops.precond_kda.chunk_intra_token_parallel import chunk_precond_kda_fwd_intra_token_parallel
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_HOPPER, IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    DOT_PRECISION = tl.constexpr('tf32x3')
else:
    DOT_PRECISION = tl.constexpr('ieee')

NUM_WARPS = [1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=['H', 'K', 'BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_diag_kdn_fwd_kernel_inter_solve_fused(
    q,
    k,
    kappa,
    g,
    Aqk,
    Akkd,
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
    """
    Off-diagonal scores of the two sub-chunks and the inverse of the unit lower-triangular key-key block.
    Assumes BT == 2 * BC.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc1 + o_i
    m_tc0 = o_c0 < T
    m_tc1 = o_c1 < T

    # the second sub-chunk against the first, at tf32x3 on fp32 operands, referenced to the start of the second
    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_M10 = tl.zeros([BC, BC], dtype=tl.float32)
    if i_tc1 < T:
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_k0 = o_c0[:, None] * (H*K) + o_k[None, :]
            p_k1 = o_c1[:, None] * (H*K) + o_k[None, :]
            m_k0 = m_tc0[:, None] & m_k[None, :]
            m_k1 = m_tc1[:, None] & m_k[None, :]
            b_kappa0 = tl.load(kappa + p_k0, mask=m_k0, other=0.).to(tl.float32)
            b_g0 = tl.load(g + p_k0, mask=m_k0, other=0.).to(tl.float32)
            b_q1 = tl.load(q + p_k1, mask=m_k1, other=0.).to(tl.float32)
            b_k1 = tl.load(k + p_k1, mask=m_k1, other=0.).to(tl.float32)
            b_g1 = tl.load(g + p_k1, mask=m_k1, other=0.).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * H*K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_kappa0 * exp2(b_gn1[None, :] - b_g0))
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt, input_precision=DOT_PRECISION)
            b_M10 += tl.dot(b_k1 * b_gqn, b_kgt, input_precision=DOT_PRECISION)
    b_Aqk10 = tl.where(m_tc1[:, None], b_Aqk10, 0.)
    b_M10 = tl.where(m_tc1[:, None], b_M10, 0.)
    if i_tc1 < T:
        tl.store(Aqk + o_c1[:, None] * (H*BT) + o_i[None, :], (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_tc1[:, None])

    # forward substitution on the two diagonal blocks
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Ai00 = tl.load(Akkd + o_c0[:, None] * (H*BC) + o_i[None, :], mask=m_tc0[:, None], other=0.).to(tl.float32)
    b_Ai11 = tl.load(Akkd + o_c1[:, None] * (H*BC) + o_i[None, :], mask=m_tc1[:, None], other=0.).to(tl.float32)
    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)
    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2 * BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    b_Ai00 += m_I
    b_Ai11 += m_I

    # the off-diagonal block of the inverse, at tf32x3
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_M10, input_precision=DOT_PRECISION),
        b_Ai00,
        input_precision=DOT_PRECISION,
    )
    b_Ai10 = tl.where(m_tc1[:, None], b_Ai10, 0.)
    tl.store(Akk + o_c0[:, None] * (H*BT) + o_i[None, :], b_Ai00.to(Akk.dtype.element_ty), mask=m_tc0[:, None])
    tl.store(Akk + o_c1[:, None] * (H*BT) + o_i[None, :], b_Ai10.to(Akk.dtype.element_ty), mask=m_tc1[:, None])
    tl.store(Akk + o_c1[:, None] * (H*BT) + BC + o_i[None, :], b_Ai11.to(Akk.dtype.element_ty), mask=m_tc1[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'BC', 'BK', 'NC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_diag_kdn_bwd_kernel_intra(
    q,
    k,
    kappa,
    g,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dkappa,
    dkappa2,
    dg,
    dg2,
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
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dkappa += (bos * H + i_h) * K
    dkappa2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    o_ti = i_ti + o_i
    m_ti = o_ti < T
    m_tik = m_ti[:, None] & m_k[None, :]
    p_ik = o_ti[:, None] * (H*K) + o_k[None, :]

    b_g = tl.load(g + p_ik, mask=m_tik, other=0.).to(tl.float32)

    # the row side: gradients of the queries and the read keys
    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        b_gn = tl.load(g + i_ti * H*K + o_k, mask=m_k, other=0.).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_tj = i_t * BT + i_j * BC + o_i
            p_jk = o_tj[:, None] * (H*K) + o_k[None, :]
            m_tjk = (o_tj < T)[:, None] & m_k[None, :]
            b_kappa = tl.load(kappa + p_jk, mask=m_tjk, other=0.).to(tl.float32)
            b_gk = tl.load(g + p_jk, mask=m_tjk, other=0.).to(tl.float32)
            b_kappag = b_kappa * exp2(b_gn - b_gk)
            b_dAqk = tl.load(dAqk + o_ti[:, None] * (H*BT) + (i_j * BC + o_i)[None, :], mask=m_ti[:, None], other=0.)
            b_dAkk = tl.load(dAkk + o_ti[:, None] * (H*BT) + (i_j * BC + o_i)[None, :], mask=m_ti[:, None], other=0.)
            # the query gradient runs at tf32x3 on fp32 operands, the key gradient on operands in the input dtype
            b_dq2 += tl.dot(b_dAqk, b_kappag, input_precision=DOT_PRECISION)
            b_dk2 += tl.dot(b_dAkk.to(k.dtype.element_ty), b_kappag.to(k.dtype.element_ty))
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_dA = o_ti * (H*BT) + i_i * BC
    p_kappaj = kappa + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k
    b_q = tl.load(q + p_ik, mask=m_tik, other=0.).to(tl.float32)
    b_k = tl.load(k + p_ik, mask=m_tik, other=0.).to(tl.float32)
    for j in range(0, min(BC, T - i_ti)):
        b_dAqk = tl.load(dAqk + o_dA + j, mask=m_ti, other=0.)
        b_dAkk = tl.load(dAkk + o_dA + j, mask=m_ti, other=0.)
        b_kappaj = tl.load(p_kappaj, mask=m_k, other=0.).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0.).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_gqk = exp2(b_g - b_gkj[None, :])
        b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kappaj[None, :] * b_gqk, 0.)
        b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kappaj[None, :] * b_gqk, 0.)
        p_kappaj += H*K
        p_gkj += H*K
    b_dg2 = b_q * b_dq2
    b_dq2 += tl.load(dq + p_ik, mask=m_tik, other=0.)
    tl.store(dq2 + p_ik, b_dq2.to(dq2.dtype.element_ty), mask=m_tik)
    tl.debug_barrier()

    # the column side: gradients of the write keys, at tf32x3 on fp32 operands
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    NC_active = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC_active - 1:
        b_gn = tl.load(g + (min(i_ti + BC, T) - 1) * H*K + o_k, mask=m_k, other=0.).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC_active):
            o_tj = i_t * BT + i_j * BC + o_i
            m_tj = o_tj < T
            p_jk = o_tj[:, None] * (H*K) + o_k[None, :]
            m_tjk = m_tj[:, None] & m_k[None, :]
            b_qj = tl.load(q + p_jk, mask=m_tjk, other=0.).to(tl.float32)
            b_kj = tl.load(k + p_jk, mask=m_tjk, other=0.).to(tl.float32)
            b_gk = tl.load(g + p_jk, mask=m_tjk, other=0.).to(tl.float32)
            # [BC, BC], the transposed score gradients
            b_dAqk = tl.load(dAqk + o_tj[None, :] * (H*BT) + (i_i * BC + o_i)[:, None], mask=m_tj[None, :], other=0.)
            b_dAkk = tl.load(dAkk + o_tj[None, :] * (H*BT) + (i_i * BC + o_i)[:, None], mask=m_tj[None, :], other=0.)
            b_gkn = tl.where(m_tj[:, None], exp2(b_gk - b_gn), 0)
            b_dkt += tl.dot(b_dAqk, b_qj * b_gkn, input_precision=DOT_PRECISION)
            b_dkt += tl.dot(b_dAkk, b_kj * b_gkn, input_precision=DOT_PRECISION)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * (H*BT) + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k
    for j in range(0, min(BC, T - i_ti)):
        b_dAqk = tl.load(dAqk + o_dA + j * H*BT)
        b_dAkk = tl.load(dAkk + o_dA + j * H*BT)
        b_qj = tl.load(p_qj, mask=m_k, other=0.).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0.).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0.).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_gkq = exp2(b_gkj[None, :] - b_g)
        b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
        b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gkq, 0.)
        p_qj += H*K
        p_kj += H*K
        p_gkj += H*K

    b_kappa = tl.load(kappa + p_ik, mask=m_tik, other=0.).to(tl.float32)
    b_dg2 += b_dk2 * b_k - b_dkt * b_kappa + tl.load(dg + p_ik, mask=m_tik, other=0.)
    b_dk2 += tl.load(dk + p_ik, mask=m_tik, other=0.)
    b_dkt += tl.load(dkappa + p_ik, mask=m_tik, other=0.)
    tl.store(dk2 + p_ik, b_dk2.to(dk2.dtype.element_ty), mask=m_tik)
    tl.store(dkappa2 + p_ik, b_dkt.to(dkappa2.dtype.element_ty), mask=m_tik)
    tl.store(dg2 + p_ik, b_dg2.to(dg2.dtype.element_ty), mask=m_tik)


def chunk_diag_kdn_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 32,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    assert BT == 2 * BC, "The fused intra-chunk solve assumes two sub-chunks per chunk."
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # the causal blocks above the diagonal are never written, so both start from zeros
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float)
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float)
    # the memory reads along k and writes along kappa with unit beta
    chunk_precond_kda_fwd_intra_token_parallel(
        q=q,
        k=k,
        k_precond=kappa,
        gk=gk,
        beta=k.new_ones(B, T, H, dtype=torch.float),
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        sub_chunk_size=BC,
    )
    chunk_diag_kdn_fwd_kernel_inter_solve_fused[(NT, B * H)](
        q=q,
        k=k,
        kappa=kappa,
        g=gk,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    w, u, kg, _ = recompute_w_u_fwd(
        k=k,
        kappa=kappa,
        v=v,
        A=Akk,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, kg, Aqk, Akk


def chunk_diag_kdn_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    g: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dkappa: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    BK = min(32, max(triton.next_power_of_2(K), 16))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    dkappa2 = torch.empty_like(kappa, dtype=torch.float)
    dg2 = torch.empty_like(g, dtype=torch.float)
    chunk_diag_kdn_bwd_kernel_intra[(NK * NC, NT, B * H)](
        q=q,
        k=k,
        kappa=kappa,
        g=g,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dkappa=dkappa,
        dkappa2=dkappa2,
        dg=dg,
        dg2=dg2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return dq2, dk2, dkappa2, dg2
