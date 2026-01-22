# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = b_v * b_b[:, None]

        p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vb *= exp(b_gv)
        if STORE_VG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            b_gn = tl.load(gv + ((bos + last_idx) * H + i_h) * V + o_v, mask=m_v, other=0.)
            b_vg = b_v * exp(b_gn - b_gv)

            p_vg = tl.make_block_ptr(vg + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_vg, b_vg.to(p_vg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_vb.to(b_v.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_b[:, None]).to(b_k.dtype)
        b_u = tl.dot(b_A, b_kb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gv_exp = exp(tl.load(p_gv, boundary_check=(0, 1)))
        b_vbg = b_v * b_b[:, None] * b_gv_exp
        b_dw = tl.load(p_dw, boundary_check=(0, 1))

        b_dA += tl.dot(b_dw, tl.trans(b_vbg).to(b_dw.dtype))
        b_dvbg = tl.dot(b_A, b_dw)
        b_dv = b_dvbg * b_gv_exp * b_b[:, None]
        b_db += tl.sum(b_dvbg * b_v * b_gv_exp, 1)
        b_dgv = b_dvbg * b_vbg
        
        p_dgv = tl.make_block_ptr(dgv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_du = tl.make_block_ptr(du + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_b[:, None]).to(b_k.dtype) # BT BK
        b_du = tl.load(p_du, boundary_check=(0, 1)) # BT BK
        b_dA += tl.dot(b_du, tl.trans(b_kb)) # BT BT
        b_dkb = tl.dot(b_A, b_du) # BT BK
        b_dk = b_dkb * b_b[:, None]
        b_db += tl.sum(b_dkb * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA = tl.where(m_A, -b_dA, 0)
    
    # if USE_GV:
    p_dA = tl.make_block_ptr(dA + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gv: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
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
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
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
