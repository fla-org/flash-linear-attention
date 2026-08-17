# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kda_kernel(
    q,
    k,
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
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = beta + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        p_u = u + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_w = w + o_t[:, None] * (HV*K) + o_k[None, :]
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kb = b_k * b_b[:, None]

        p_gk = gk + o_t[:, None] * (HV*K) + o_k[None, :]
        b_gk = tl.load(p_gk, mask=m_tk, other=0.0).to(tl.float32)
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
            p_qg = qg + o_t[:, None] * (HV*K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_tk, other=0.0)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), mask=m_tk)
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            b_gn = tl.load(gk + last_idx * HV*K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = kg + o_t[:, None] * (HV*K) + o_k[None, :]
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_tk)

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_tk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kda_kernel(
    k,
    v,
    beta,
    gk,
    A,
    dA,
    dw,
    du,
    dk,
    dk2,
    dv,
    db,
    dg,
    dg2,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    gk += (bos * HV + i_hv) * K
    A += (bos * HV + i_hv) * BT
    dA += (bos * HV + i_hv) * BT
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dw += (bos * HV + i_hv) * K
    du += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    db += bos * HV + i_hv
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = beta + o_t * HV
    p_db = db + o_t * HV
    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_A = A + o_A[:, None] + o_t[None, :] * (HV*BT)

    b_b = tl.load(p_b, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dk2 = dk2 + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dw = dw + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dg2 = dg2 + o_t[:, None] * (HV*K) + o_k[None, :]

        # [BT, BK]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        p_gk = gk + o_t[:, None] * (HV*K) + o_k[None, :]
        b_gk_exp = exp2(tl.load(p_gk, mask=m_k, other=0.0))
        b_kbg = b_k * b_b[:, None] * b_gk_exp
        b_dw = tl.load(p_dw, mask=m_k, other=0.0)

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        b_dk = b_dkbg * b_gk_exp * b_b[:, None] + tl.load(p_dk, mask=m_k, other=0.0)
        b_db += tl.sum(b_dkbg * b_k * b_gk_exp, 1)
        b_dg = b_kbg * b_dkbg + tl.load(p_dg, mask=m_k, other=0.0)

        tl.store(p_dk2, b_dk.to(p_dk2.dtype.element_ty), mask=m_k)
        tl.store(p_dg2, b_dg.to(p_dg2.dtype.element_ty), mask=m_k)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        p_du = du + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, mask=m_v, other=0.0)
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA = tl.where(m_A, -b_dA, 0)

    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV*BT) + o_A[None, :]
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


@dispatch('kda')
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    qg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype) if q is not None else None
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    recompute_w_u_fwd_kda_kernel[(NT, B*HV)](
        q=q,
        k=k,
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
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    dk: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk2 = torch.empty_like(dk, dtype=torch.float)
    dv = torch.empty_like(v)
    dg2 = torch.empty_like(gk, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    prepare_wy_repr_bwd_kda_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        gk=gk,
        A=A,
        dA=dA,
        dw=dw,
        du=du,
        dk=dk,
        dk2=dk2,
        dv=dv,
        db=db,
        dg=dg,
        dg2=dg2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    dk = dk2
    dg = dg2
    return dk, dv, db, dg, dA
