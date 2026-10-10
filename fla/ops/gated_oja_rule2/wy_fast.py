# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# WY-representation recompute for Oja2.
#
# Given the solved chunk inverse ``A = (I + tril(T, -1))^{-1}``, this kernel produces the two within-chunk auxiliaries
# consumed by the inter-chunk state recurrence:
#
#   w = A @ ((b * v) * exp(gv))     # erase-side, M-axis
#   u = A @ (c * k)                 # write-side, K-axis
#
# Difference vs ``gated_oja_rule``: the scalar ``beta`` is split into a per-slot erase gate ``b`` on the M axis and a
# per-channel write gate ``c`` on the K axis. The two live on different axes, so the kernel loads them per BV / BK block.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'STORE_VG': lambda args: args['vg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_oja2_kernel(
    k,
    v,
    vg,
    b_gate,
    c_gate,
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

    o_A = tl.arange(0, BT)
    m_AT = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_w = w + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_bg = b_gate + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_bg = tl.load(p_bg, mask=m_tv, other=0.0)
        b_vb = b_v * b_bg

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

        b_w = tl.dot(b_A, b_vb.to(b_A.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_tv)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_u = u + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_cg = c_gate + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_cg = tl.load(p_cg, mask=m_tk, other=0.0)
        b_kb = (b_k * b_cg).to(b_k.dtype)
        b_u = tl.dot(b_A, b_kb.to(b_A.dtype), allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_tk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_oja2_kernel(
    k,
    v,
    b_gate,
    c_gate,
    gv,
    A,
    dA,
    dw,
    du,
    dk,
    dv,
    db_gate,
    dc_gate,
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
    p_A = A + (bos*H + i_h) * BT + o_A[:, None] + o_t[None, :] * (H*BT)

    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    # w path: w = A @ ((b * v) * exp(gv))
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_bg = b_gate + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dv = dv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dbg = db_gate + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_dw = dw + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_gv = gv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_bg = tl.load(p_bg, mask=m_tv, other=0.0)
        b_gv_exp = exp(tl.load(p_gv, mask=m_tv, other=0.0))
        b_vbg = b_bg * b_v * b_gv_exp
        b_dw = tl.load(p_dw, mask=m_tv, other=0.0)

        b_dA = tl.dot(b_dw, tl.trans(b_vbg).to(b_dw.dtype), b_dA)
        b_dvbg = tl.dot(b_A, b_dw.to(b_A.dtype))
        b_dv = b_dvbg * b_bg * b_gv_exp
        b_dbg = b_dvbg * b_v * b_gv_exp
        b_dgv = b_dvbg * b_vbg

        p_dgv = dgv + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_tv)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)
        tl.store(p_dbg, b_dbg.to(p_dbg.dtype.element_ty), mask=m_tv)

    # u path: u = A @ (c * k)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_cg = c_gate + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dcg = dc_gate + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_du = du + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        # [BT, BK]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_cg = tl.load(p_cg, mask=m_tk, other=0.0)
        b_kb = (b_cg * b_k).to(b_k.dtype)
        b_du = tl.load(p_du, mask=m_tk, other=0.0)
        b_dA = tl.dot(b_du, tl.trans(b_kb), b_dA)
        b_dkb = tl.dot(b_A, b_du.to(b_A.dtype))
        b_dk = b_dkb * b_cg
        b_dcg = b_dkb * b_k
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dcg, b_dcg.to(p_dcg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA = tl.where(m_A, -b_dA, 0)

    m_AT2 = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_AT2)


def recompute_w_u_fwd_oja2(
    k: torch.Tensor,
    v: torch.Tensor,
    b_gate: torch.Tensor,
    c_gate: torch.Tensor,
    A: torch.Tensor,
    gv: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    r"""
    Produce the WY auxiliaries of Oja2.

    Args:
        k (torch.Tensor):
            The keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            The slot code of shape `[B, T, H, M]`.
        b_gate (torch.Tensor):
            The per-slot erase gate of shape `[B, T, H, M]`.
        c_gate (torch.Tensor):
            The per-channel write gate of shape `[B, T, H, K]`.
        A (torch.Tensor):
            The solved chunk inverse of shape `[B, T, H, BT]`.
        gv (torch.Tensor, Optional):
            The chunk-local cumulative log-decay of shape `[B, T, H, M]`. Default: `None`.

    Returns:
        `w` of shape `[B, T, H, M]`, `u` of shape `[B, T, H, K]`, and the decay-normalized slot code `vg` of shape
        `[B, T, H, M]` (`None` if `gv` is `None`).
    """
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
    recompute_w_u_fwd_oja2_kernel[(NT, B*H)](
        k=k,
        v=v,
        vg=vg,
        b_gate=b_gate,
        c_gate=c_gate,
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


def prepare_wy_repr_bwd_oja2(
    k: torch.Tensor,
    v: torch.Tensor,
    b_gate: torch.Tensor,
    c_gate: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    gv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Backward of `recompute_w_u_fwd_oja2`.

    Returns:
        The gradients `dk`, `dv`, `db_gate`, `dc_gate`, `dgv`, and the gradient `dA` w.r.t. the input of `solve_tril`.
    """
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
    db_gate = torch.empty_like(b_gate, dtype=torch.float)
    dc_gate = torch.empty_like(c_gate, dtype=torch.float)
    dgv = torch.empty_like(gv, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    prepare_wy_repr_bwd_oja2_kernel[(NT, B * H)](
        k=k,
        v=v,
        b_gate=b_gate,
        c_gate=c_gate,
        gv=gv,
        A=A,
        dA=dA,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db_gate=db_gate,
        dc_gate=dc_gate,
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

    return dk, dv, db_gate, dc_gate, dgv, dA
