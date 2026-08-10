# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs, check_shared_mem

# Blackwell can select unstable Triton configs for prepare_wy_repr_bwd_kernel
# during autotuning (see #913). Restrict it to the config that has been
# validated on B200 until the wider config space is re-validated.
PREPARE_WY_REPR_BWD_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]
PREPARE_WY_REPR_BWD_NUM_STAGES = [4] if IS_NVIDIA_BLACKWELL else [2, 3, 4]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
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
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
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
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_b = beta + bos*HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    p_A = A + (bos*HV + i_h) * BT + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_u = u + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    if USE_G:
        p_g = g + (bos*HV + i_h) + o_t * HV
        b_g = exp2(tl.load(p_g, mask=m_t, other=0.0))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_w = w + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in PREPARE_WY_REPR_BWD_NUM_WARPS
        for num_stages in PREPARE_WY_REPR_BWD_NUM_STAGES
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
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
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_b = beta + (bos*HV + i_h) + o_t * HV
    p_db = db + (bos*HV + i_h) + o_t * HV
    p_A = A + (bos*HV + i_h) * BT + o_A[:, None] + o_t[None, :] * (HV*BT)

    b_b = tl.load(p_b, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        p_g = g + (bos*HV + i_h) + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_exp = exp2(b_g)
        b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dw = dw + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        # [BT, BK]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        if USE_G:
            b_kbg = b_k * (b_b * b_g_exp)[:, None]
        else:
            b_kbg = b_k * b_b[:, None]
        b_dw = tl.load(p_dw, mask=m_k, other=0.0)

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        if USE_G:
            b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
            b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
            b_dg += tl.sum(b_dkbg * b_kbg, 1)
        else:
            b_dk = b_dkbg * b_b[:, None]
            b_db += tl.sum(b_dkbg * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_du = du + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
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

    if USE_G:
        b_dA *= exp2(b_g[:, None] - b_g[None, :])

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)

    # keep this barrier: triton-ascend can misorder b_dA before the dk load-modify-store without it
    tl.debug_barrier()

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kt = tl.trans(b_k)
        b_kb = b_k * b_b[:, None]

        b_A += tl.dot(b_k, b_kt)
        b_dkb = tl.dot(b_dA, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA))
        b_dk += tl.load(p_dk, mask=m_k, other=0.0)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)

    b_A *= b_b[:, None]
    if USE_G:
        b_AdA = b_dA * b_A
        p_dg = dg + (bos*HV + i_h) + o_t * HV
        b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)


@dispatch('gated_delta_rule')
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
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
    return w, u


@dispatch('gated_delta_rule')
def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if k.dtype is torch.float32 and os.environ.get('TRITON_F32_DEFAULT') == 'ieee':
        return _prepare_wy_repr_bwd_split(
            k=k,
            v=v,
            beta=beta,
            A=A,
            dw=dw,
            du=du,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            NT=NT,
        )

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    return_dg = g is not None
    if g is None:
        # exp2(0) is the no-g path; reuse the g-specialized compiler artifact.
        g = torch.zeros_like(beta)

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
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
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    if not return_dg:
        dg = None
    return dk, dv, db, dg


def _prepare_wy_repr_bwd_split(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    NT: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from fla.ops.gated_delta_rule.backends.triton_ascend import wy_fast as wy_split

    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = min(max(triton.next_power_of_2(K), 16), 32)
    BV = min(max(triton.next_power_of_2(V), 16), 32)
    use_g = g is not None
    is_varlen = cu_seqlens is not None

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if use_g else None
    db = torch.empty_like(beta)
    dA_scr = torch.empty_like(A, dtype=torch.float32)
    dA_mid = torch.empty_like(A, dtype=torch.float32)
    dA_out = torch.empty_like(A, dtype=torch.float32)
    a2_scr = torch.empty_like(A, dtype=torch.float32)
    col_acc_scr = torch.empty(
        B,
        triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices),
        HV,
        BT,
        dtype=torch.float32,
        device=k.device,
    )
    g_arg = g if use_g else beta
    dg_arg = dg if use_g else beta
    grid = (NT, B * HV)
    base = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BT,
        IS_VARLEN=is_varlen,
        NT_OFFSET=0,
        BH_OFFSET=0,
        num_warps=2,
    )

    wy_split.prepare_wy_repr_bwd_k_npu[grid](
        k=k, beta=beta, g=g_arg, A=A, dw=dw,
        dk=dk, dA_scr=dA_scr, db=db, dg=dg_arg,
        H=H, HV=HV, K=K, BK=BK, USE_G=use_g,
        **base,
    )
    wy_split.prepare_wy_repr_bwd_v_npu[grid](
        v=v, beta=beta, A=A, du=du, dv=dv, dA_scr=dA_scr, db=db,
        HV=HV, V=V, BV=BV,
        **base,
    )
    wy_split.prepare_wy_repr_bwd_da_mask_npu[grid](
        dA_scr=dA_scr,
        HV=HV,
        **base,
    )
    wy_split.prepare_wy_repr_bwd_da_dot1_npu[grid](
        A=A, dA_scr=dA_scr, dA_mid=dA_mid,
        HV=HV,
        **base,
    )
    wy_split.prepare_wy_repr_bwd_da_dot2_npu[grid](
        A=A, dA_mid=dA_mid, dA_out=dA_out,
        HV=HV,
        **base,
    )
    if use_g:
        wy_split.prepare_wy_repr_bwd_da_gate_npu[grid](
            g=g_arg, dA_out=dA_out,
            HV=HV, BC=16,
            **base,
        )
    wy_split.prepare_wy_repr_bwd_finalize_k_npu[grid](
        k=k, beta=beta, dA_out=dA_out, dk=dk, db=db,
        H=H, HV=HV, K=K, BK=BK,
        **base,
    )
    if use_g:
        wy_split.prepare_wy_repr_bwd_finalize_a2_npu[grid](
            k=k, beta=beta, a2_scr=a2_scr,
            H=H, HV=HV, K=K, BK=BK,
            **base,
        )
        wy_split.prepare_wy_repr_bwd_finalize_dg_npu[grid](
            dA_out=dA_out, a2_scr=a2_scr, dg=dg_arg, col_acc_scr=col_acc_scr,
            HV=HV, BC=16,
            **base,
        )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


fwd_recompute_w_u = recompute_w_u_fwd
bwd_prepare_wy_repr = prepare_wy_repr_bwd
