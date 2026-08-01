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
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem


@triton.heuristics({
    'STORE_VG': lambda args: args['vg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    vg,
    beta,
    w,
    u,
    A,
    gv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_VG: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
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
    p_b = beta + bos*H + i_h + o_t * H
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_AT = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_w = w + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_vb = b_v * b_b[:, None]

        p_gv = gv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_gv = tl.load(p_gv, mask=m_tv, other=0.0)
        b_vb *= exp(b_gv)
        if STORE_VG:
            last_idx = min(i_t * BT + BT, T) - 1

            m_v = o_v < V
            b_gn = tl.load(gv + ((bos + last_idx) * H + i_h) * V + o_v, mask=m_v, other=0.)
            b_vg = b_v * exp(b_gn - b_gv)

            p_vg = vg + (bos * H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
            tl.store(p_vg, b_vg.to(p_vg.dtype.element_ty), mask=m_tv)

        b_w = tl.dot(b_A, b_vb.to(b_v.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_tv)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_u = u + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kb = (b_k * b_b[:, None]).to(b_k.dtype)
        b_u = tl.dot(b_A, b_kb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_tk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    gv,
    A,
    dA,
    dw,
    du,
    dk,
    dv,
    db,
    dgv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
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
    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_b = beta + (bos*H + i_h) + o_t * H
    p_db = db + (bos*H + i_h) + o_t * H
    p_A = A + (bos*H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H*BT)

    b_b = tl.load(p_b, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dv = dv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dw = dw + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        p_gv = gv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_gv_exp = exp(tl.load(p_gv, mask=m_tv, other=0.0))
        b_vbg = b_v * b_b[:, None] * b_gv_exp
        b_dw = tl.load(p_dw, mask=m_tv, other=0.0)

        b_dA += tl.dot(b_dw, tl.trans(b_vbg).to(b_dw.dtype))
        b_dvbg = tl.dot(b_A, b_dw)
        b_dv = b_dvbg * b_gv_exp * b_b[:, None]
        b_db += tl.sum(b_dvbg * b_v * b_gv_exp, 1)
        b_dgv = b_dvbg * b_vbg

        p_dgv = dgv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_tv)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_du = du + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        # [BT, BK]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_kb = (b_k * b_b[:, None]).to(b_k.dtype)  # BT BK
        b_du = tl.load(p_du, mask=m_tk, other=0.0)  # BT BK
        b_dA += tl.dot(b_du, tl.trans(b_kb))  # BT BT
        b_dkb = tl.dot(b_A, b_du)  # BT BK
        b_dk = b_dkb * b_b[:, None]
        b_db += tl.sum(b_dkb * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA = tl.where(m_A, -b_dA, 0)

    # if USE_GV:
    m_AT2 = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_AT2)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gv: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(v)
    u = torch.empty_like(k)
    vg = torch.empty_like(v) if gv is not None else None
    recompute_w_u_fwd_kernel[(NT, B*H)](
        k=k,
        v=v,
        vg=vg,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gv=gv,
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
    return w, u, vg


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    gv: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v, dtype=torch.float)

    dgv = torch.empty_like(gv, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)

    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        gv=gv,
        A=A,
        dA=dA,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dgv=dgv,
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

    return dk, dv, db, dgv, dA
