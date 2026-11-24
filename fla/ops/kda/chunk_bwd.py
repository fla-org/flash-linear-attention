# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.backends import dispatch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu_pre_process, expand_h0
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.kda.wy_fast import prepare_dual_wy_fwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.graph import get_static_buffer
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA_HOPPER,
    IS_NVIDIA_SM100,
    autotune_cache_kwargs,
    check_shared_mem,
)

BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]
DUAL_BV_LIST = [32, 64, 128] if check_shared_mem('ampere') else [16, 32]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv(
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
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_A = A + (bos * HV + i_hv) * BT + o_A[:, None] + o_t[None, :] * (HV*BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vT = m_v[:, None] & m_t[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        p_v = v + o_v[:, None] + o_t[None, :] * (HV*V)
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        # [BV, BT]
        b_v = tl.load(p_v, mask=m_vT, other=0.0)
        # [BT, BV]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        # [BT, BT]
        b_dA = tl.dot(b_do, b_v, b_dA)
        # [BT, BV]
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)

    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
        if not (IS_NVIDIA_SM100 and BK == 32 and num_warps != 2)
    ],
    key=['BT', 'HV', 'STATE_V_FIRST', 'K', 'V'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
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
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
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
    v += (bos * HV + i_hv) * V
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K*V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K*V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT

    p_beta = beta + o_t * HV
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_A = A + o_A[:, None] + o_t[None, :] * (HV * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]

        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_tv = m_t[:, None] & (o_v[None, :] < V)
            m_h = (o_v[:, None] < V) & m_k[None, :]
            p_v_new = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
            p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
            if STATE_V_FIRST:
                p_h = h + o_v[:, None] * K + o_k[None, :]
                p_dh = dh + o_v[:, None] * K + o_k[None, :]
            else:
                p_h = h + o_v[:, None] + o_k[None, :] * V
                p_dh = dh + o_v[:, None] + o_k[None, :] * V
            p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
            # [BT, BV]
            b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            # [BV, BK]
            b_h = tl.load(p_h, mask=m_h, other=0.0)
            b_dh = tl.load(p_dh, mask=m_h, other=0.0)
            # [BT, BV]
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
            b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk)
            b_dw = tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype), b_dw)
            tl.debug_barrier()  # DO NOT REMOVE THIS LINE!
            if i_k == 0:
                p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
                p_dv2 = dv2 + o_t[:, None] * (HV*V) + o_v[None, :]

                b_v = tl.load(p_v, mask=m_tv, other=0.0)

                b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA)

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)

                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), mask=m_tv)

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA = tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)), b_dA)

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = dq + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (HV*K) + o_k[None, :]
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
    p_db = db + o_t * HV
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


@dispatch
def chunk_kda_bwd_dAv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, HV, V = *k.shape, do.shape[2], do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if use_graph:
        dA = get_static_buffer("dAv_dA", (B, T, HV, BT), torch.float, v.device)
        dv = get_static_buffer("dAv_dv", tuple(do.shape), do.dtype, do.device)
    else:
        dA = v.new_empty(B, T, HV, BT, dtype=torch.float)
        dv = torch.empty_like(do)
    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_dAv[grid](
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
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_GRAPH=use_graph,
    )
    return dA, dv


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in BV_LIST
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['HV', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv_dual(
    v,
    A_eff,
    Aqk,
    Akk,
    beta,
    do,
    dv,
    dAqk,
    wm_a,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    """Compute intra dY and absorb dA_eff into dAqk and WM_A = Aqk^T @ dA_eff in registers."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    A_eff += (bos * HV + i_hv) * BT
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    beta += bos * HV + i_hv
    dAqk += (bos * HV + i_hv) * BT
    wm_a += (bos * HV + i_hv) * BT

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    m_tril = (o_i[:, None] >= o_i[None, :]) & m_t[:, None]

    b_Aeff = tl.where(m_tril, tl.load(A_eff + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0), 0.0)
    b_Aeff_T = tl.trans(b_Aeff).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        b_v = tl.load(v + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_do = tl.load(do + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_dA = tl.dot(b_do, tl.trans(b_v), b_dA)
        b_dv = tl.dot(b_Aeff_T, b_do)
        tl.store(dv + o_t[:, None] * (HV * V) + o_v[None, :], b_dv.to(dv.dtype.element_ty), mask=m_tv)

    b_dA = tl.where(m_tril, b_dA, 0.0)
    b_beta = tl.load(beta + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_M = tl.where(m_tril, tl.load(Akk + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0), 0.0)
    b_Aqk = tl.where(m_tril, tl.load(Aqk + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0), 0.0)

    b_dAqk = tl.dot((b_dA * b_beta[None, :]).to(b_M.dtype), tl.trans(b_M)) * scale
    b_dAqk = tl.where(m_tril, b_dAqk, 0.0)
    b_wm_a = tl.dot(tl.trans(b_Aqk), b_dA.to(b_Aqk.dtype))
    b_wm_a = tl.where(m_tril, b_wm_a, 0.0)

    tl.store(dAqk + o_t[:, None] * (HV * BT) + o_i[None, :], b_dAqk.to(dAqk.dtype.element_ty), mask=m_A)
    tl.store(wm_a + o_t[:, None] * (HV * BT) + o_i[None, :], b_wm_a.to(wm_a.dtype.element_ty), mask=m_A)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in DUAL_BV_LIST
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'HV', 'STATE_V_FIRST', 'K', 'V', 'BK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T', 'NT_GRID'])
def chunk_kda_bwd_kernel_dual_wy_dqkg(
    q,
    k,
    v_new,
    g,
    beta,
    Akk,
    h,
    do,
    dh,
    dv,
    dq,
    dk,
    dg,
    wm,
    cu_seqlens,
    chunk_indices,
    scale,
    B,
    T,
    NT_GRID,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_tk, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_t, i_k = i_tk // NK, i_tk % NK
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    total_tokens = B.to(tl.int64) * T.to(tl.int64)
    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_tg = i_b * NT_GRID + i_t
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    last_idx = (min(T, i_t * BT + BT) - 1).to(tl.int64)
    m_last = o_t == last_idx
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    m_tril = (o_i[:, None] >= o_i[None, :]) & m_t[:, None]

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    v_new += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv) * K * V
    dh += (i_tg * HV + i_hv) * K * V

    b_dQd = tl.zeros([BT, BK], dtype=tl.float32)
    b_dT = tl.zeros([BT, BK], dtype=tl.float32)
    b_dKd = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        b_v_new = tl.load(v_new + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_do = tl.load(do + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_dv = tl.load(dv + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0)
        if STATE_V_FIRST:
            m_h = m_v[:, None] & m_k[None, :]
            b_h = tl.load(h + o_v[:, None] * K + o_k[None, :], mask=m_h, other=0.0)
            b_dh = tl.load(dh + o_v[:, None] * K + o_k[None, :], mask=m_h, other=0.0)
            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dQd = tl.dot(b_do, b_h.to(b_do.dtype), b_dQd)
            b_dT = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dT)
            b_dKd = tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype), b_dKd)
        else:
            m_h = m_k[:, None] & m_v[None, :]
            b_h = tl.load(h + o_k[:, None] * V + o_v[None, :], mask=m_h, other=0.0)
            b_dh = tl.load(dh + o_k[:, None] * V + o_v[None, :], mask=m_h, other=0.0)
            b_dgk += tl.sum(b_h * b_dh, axis=1)
            b_h_t = tl.trans(b_h)
            b_dh_t = tl.trans(b_dh)
            b_dQd = tl.dot(b_do, b_h_t.to(b_do.dtype), b_dQd)
            b_dT = tl.dot(b_v_new, b_dh_t.to(b_v_new.dtype), b_dT)
            b_dKd = tl.dot(b_dv.to(b_v_new.dtype), b_h_t.to(b_v_new.dtype), b_dKd)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    Akk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K

    b_g = tl.load(g + o_t[:, None] * (HV * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    b_gk_exp = exp2(b_g)
    b_dq = b_dQd * b_gk_exp * scale
    tl.store(dq + o_t[:, None] * (HV * K) + o_k[None, :], b_dq.to(dq.dtype.element_ty), mask=m_tk)

    b_dk_d = -b_dKd * b_gk_exp
    b_beta = tl.load(beta + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_M = tl.where(m_tril, tl.load(Akk + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0), 0.0)

    b_gn = tl.load(g + last_idx * (HV * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_gkn_exp = tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0.0)

    b_dKr = tl.dot(b_M, (b_dT * b_beta[:, None]).to(b_M.dtype))
    b_dk_r = b_dKr * b_gkn_exp
    b_dk = b_dk_r + b_dk_d

    b_k = tl.load(k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    b_q = tl.load(q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)

    b_dgk = b_dgk * exp2(b_gn) + tl.sum(b_k * b_dk_r, axis=0)
    b_dg = b_q * b_dq + b_k * b_dk_d - b_k * b_dk_r + m_last[:, None] * b_dgk

    tl.store(dk + o_t[:, None] * (HV * K) + o_k[None, :], b_dk.to(dk.dtype.element_ty), mask=m_tk)
    tl.store(dg + o_t[:, None] * (HV * K) + o_k[None, :], b_dg.to(dg.dtype.element_ty), mask=m_tk)

    b_kr = (b_k * b_gkn_exp).to(b_M.dtype)
    b_WM = tl.dot(b_kr, tl.trans(b_dT.to(b_M.dtype)))
    b_WM = tl.where(m_tril, b_WM, 0.0)

    wm += ((i_k * total_tokens + bos) * HV + i_hv) * BT
    tl.store(wm + o_t[:, None] * (HV * BT) + o_i[None, :], b_WM.to(wm.dtype.element_ty), mask=m_A)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'HV', 'NK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_dual_wy_dA(
    beta,
    Akk,
    wm_a,
    wm,
    db,
    dAkk,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    NK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV

    total_tokens = B.to(tl.int64) * T.to(tl.int64)
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    m_tril = (o_i[:, None] >= o_i[None, :]) & m_t[:, None]

    beta += bos * HV + i_hv
    Akk += (bos * HV + i_hv) * BT
    wm_a += (bos * HV + i_hv) * BT
    wm += (bos * HV + i_hv) * BT
    db += bos * HV + i_hv
    dAkk += (bos * HV + i_hv) * BT

    b_beta = tl.load(beta + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
    b_M = tl.where(m_tril, tl.load(Akk + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0), 0.0)
    b_WM = tl.load(wm_a + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0).to(tl.float32)

    stride_wm = total_tokens * HV * BT
    for i_k in range(NK):
        b_WM += tl.load(wm + i_k * stride_wm + o_t[:, None] * (HV * BT) + o_i[None, :], mask=m_A, other=0.0).to(tl.float32)

    b_db = tl.sum(b_M.to(tl.float32) * b_WM, axis=0)

    m_strict_tril = (o_i[:, None] > o_i[None, :]) & m_t[:, None]
    b_dM = tl.where(m_strict_tril, b_WM * b_beta[None, :], 0.0)
    b_MT = tl.trans(b_M)
    b_dL = tl.dot(b_dM.to(b_M.dtype), b_MT)
    b_dL = tl.dot(b_MT, b_dL.to(b_M.dtype))
    b_dL = tl.where(m_strict_tril, -b_dL, 0.0)

    tl.store(dAkk + o_t[:, None] * (HV * BT) + o_i[None, :], b_dL.to(dAkk.dtype.element_ty), mask=m_A)
    tl.store(db + o_t * HV, b_db.to(db.dtype.element_ty), mask=m_t)


def chunk_kda_bwd_dAv_dual(
    v: torch.Tensor,
    do: torch.Tensor,
    A_eff: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, HV, V = do.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if use_graph:
        dAqk = get_static_buffer("dAv_dA", (B, T, HV, BT), torch.float, v.device)
        dv = get_static_buffer("dAv_dv", tuple(do.shape), do.dtype, do.device)
        wm_a = get_static_buffer("dAv_wma", (B, T, HV, BT), torch.float, v.device)
    else:
        dAqk = v.new_empty(B, T, HV, BT, dtype=torch.float)
        dv = torch.empty_like(do)
        wm_a = v.new_empty(B, T, HV, BT, dtype=torch.float)

    chunk_kda_bwd_kernel_dAv_dual[(NT, B * HV)](
        v=v,
        A_eff=A_eff,
        Aqk=Aqk,
        Akk=Akk,
        beta=beta,
        do=do,
        dv=dv,
        dAqk=dAqk,
        wm_a=wm_a,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        HV=HV,
        V=V,
        BT=BT,
        USE_GRAPH=use_graph,
    )
    return dAqk, dv, wm_a


def chunk_kda_bwd_dual_wy_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    wm_a: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
):
    B, T, H, K, HV, V = *k.shape, v_new.shape[2], v_new.shape[-1]
    BT = chunk_size
    BK = min(max(triton.next_power_of_2(K), 16), 64 if check_shared_mem() else 32)
    NK = triton.cdiv(K, BK)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if use_graph:
        dq = get_static_buffer("wy_dq", (B, T, HV, K), torch.float, q.device)
        dk = get_static_buffer("wy_dk", (B, T, HV, K), torch.float, q.device)
        dg = get_static_buffer("wy_dg", tuple(g.shape), torch.float, g.device)
        db = get_static_buffer("wy_db", tuple(beta.shape), torch.float, beta.device)
        dAkk = get_static_buffer("wy_dA", tuple(Akk.shape), torch.float, Akk.device)
        wm = get_static_buffer("wy_wm", (NK, B, T, HV, BT), torch.float, q.device)
    else:
        dq = g.new_empty(B, T, HV, K, dtype=torch.float)
        dk = g.new_empty(B, T, HV, K, dtype=torch.float)
        dg = torch.empty_like(g, dtype=torch.float)
        db = torch.empty_like(beta, dtype=torch.float)
        dAkk = torch.empty_like(Akk, dtype=torch.float)
        wm = g.new_empty(NK, B, T, HV, BT, dtype=torch.float)

    chunk_kda_bwd_kernel_dual_wy_dqkg[(NK * NT, B * HV)](
        q=q,
        k=k,
        v_new=v_new,
        g=g,
        beta=beta,
        Akk=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        dq=dq,
        dk=dk,
        dg=dg,
        wm=wm,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        NT_GRID=NT,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        NK=NK,
        STATE_V_FIRST=state_v_first,
        USE_GRAPH=use_graph,
    )
    chunk_kda_bwd_kernel_dual_wy_dA[(NT, B * HV)](
        beta=beta,
        Akk=Akk,
        wm_a=wm_a,
        wm=wm,
        db=db,
        dAkk=dAkk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        HV=HV,
        BT=BT,
        NK=NK,
        USE_GRAPH=use_graph,
    )
    return dq, dk, db, dg, dAkk


@dispatch
def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
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
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
):
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # dq, dk are allocated at HV dimension; caller reduces to H if GVA
    if use_graph:
        dq = get_static_buffer("wy_dq", (B, T, HV, K), torch.float, q.device)
        dk = get_static_buffer("wy_dk", (B, T, HV, K), torch.float, q.device)
        dv2 = get_static_buffer("wy_dv2", tuple(v.shape), v.dtype, v.device)
        dg = get_static_buffer("wy_dg", tuple(g.shape), torch.float, g.device)
        db = get_static_buffer("wy_db", tuple(beta.shape), torch.float, beta.device)
        dA = get_static_buffer("wy_dA", tuple(A.shape), torch.float, A.device)
    else:
        dq = g.new_empty(B, T, HV, K, dtype=torch.float)
        dk = g.new_empty(B, T, HV, K, dtype=torch.float)
        dv2 = torch.empty_like(v)
        dg = torch.empty_like(g, dtype=torch.float)
        db = torch.empty_like(beta, dtype=torch.float)
        dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
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
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        USE_GRAPH=use_graph,
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    cp_context: FLACPContext | None = None,
    use_graph: bool = False,
    chunk_offsets: torch.LongTensor | None = None,
    use_dual_wy: bool = False,
    **kwargs,
):
    H, HV = q.shape[2], v.shape[2]
    G = HV // H

    if disable_recompute is False:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
                use_graph=use_graph,
            )
        if use_dual_wy:
            w, kg, qg, A_eff = prepare_dual_wy_fwd(
                q=q,
                k=k,
                beta=beta,
                Akk=Akk,
                gk=g,
                Aqk=Aqk,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_graph=use_graph,
            )
            u_rec = v
        else:
            w, u, qg, kg = recompute_w_u_fwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                A=Akk,
                gk=g,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_graph=use_graph,
            )
            u_rec = u
        if cp_context is not None:
            # Restore the full initial_state tensor from the compressed version.
            # Only the first sequence's state is non-zero as it's the only one that could be cross-rank.
            initial_state = expand_h0(initial_state, context=cp_context)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u_rec,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )
    else:
        w, u, qg, kg, v_new, h = kwargs["w"], kwargs["u"], kwargs["qg"], kwargs["kg"], kwargs["v_new"], kwargs["h"]
        if use_dual_wy:
            A_eff = u
        if cp_context is not None:
            # Restore the full initial_state tensor from the compressed version.
            # Only the first sequence's state is non-zero as it's the only one that could be cross-rank.
            initial_state = expand_h0(initial_state, context=cp_context)

    if use_dual_wy:
        dAqk, dv, wm_a = chunk_kda_bwd_dAv_dual(
            v=v_new,
            do=do,
            A_eff=A_eff,
            Aqk=Aqk,
            Akk=Akk,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            use_graph=use_graph,
        )
    else:
        # dAqk = do @ v.T
        # dv = A @ do
        dAqk, dv = chunk_kda_bwd_dAv(
            q=q,
            k=k,
            v=v_new,
            do=do,
            A=Aqk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            use_graph=use_graph,
        )

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=qg,
            k=kg,
            w=w,
            do=do,
            dv=dv,
            gk=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
            use_graph=use_graph,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        state_v_first=state_v_first,
    )

    if use_dual_wy:
        # In Dual-WY, dv from bwd_dhu is already the final gradient w.r.t. v.
        dq, dk, db, dg, dAkk = chunk_kda_bwd_dual_wy_dqkg(
            q=q,
            k=k,
            v_new=v_new,
            g=g,
            beta=beta,
            Akk=Akk,
            wm_a=wm_a,
            h=h,
            do=do,
            dh=dh,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
            use_graph=use_graph,
        )
    else:
        dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
            q=q,
            k=k,
            v=v,
            v_new=v_new,
            g=g,
            beta=beta,
            A=Akk,
            h=h,
            do=do,
            dh=dh,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            state_v_first=state_v_first,
            use_graph=use_graph,
        )

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        use_graph=use_graph,
    )

    # For GVA, reduce dq and dk from [B, T, HV, K] back to [B, T, H, K]
    if HV > H:
        dq = dq.view(*dq.shape[:2], H, G, dq.shape[-1]).sum(dim=3)
        dk = dk.view(*dk.shape[:2], H, G, dk.shape[-1]).sum(dim=3)

    dA, dbias = None, None
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_graph=use_graph,
    )
    if use_gate_in_kernel:
        dg, dA, dbias = kda_gate_bwd(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=lower_bound,
        )

    return dq, dk, dv, db, dg, dh0, dA, dbias
