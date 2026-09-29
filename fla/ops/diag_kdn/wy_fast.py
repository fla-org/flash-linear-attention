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
from fla.ops.utils.op import exp2
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    DOT_PRECISION = tl.constexpr('tf32x3')
    DOT_PRECISION_FAST = tl.constexpr('tf32')
else:
    DOT_PRECISION = tl.constexpr('ieee')
    DOT_PRECISION_FAST = tl.constexpr('ieee')


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    q,
    k,
    kappa,
    qg,
    kg,
    v,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    gk += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    kg += (bos * H + i_h) * K
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    u += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    last_idx = min(i_t * BT + BT, T) - 1

    # the WY products run at tf32x3 on fp32 operands
    b_A = tl.load(A + o_t[:, None] * (H*BT) + o_i[None, :], mask=m_t[:, None], other=0.).to(tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v < V)[None, :]
        b_v = tl.load(v + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_u = tl.dot(b_A, b_v, input_precision=DOT_PRECISION)
        tl.store(u + o_t[:, None] * (H*V) + o_v[None, :], b_u.to(u.dtype.element_ty), mask=m_tv)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_k = o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(k + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_gk = tl.load(gk + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_kb = b_k * exp2(b_gk)
        if STORE_QG:
            b_q = tl.load(q + p_k, mask=m_tk, other=0.).to(tl.float32)
            tl.store(qg + p_k, (b_q * exp2(b_gk)).to(qg.dtype.element_ty), mask=m_tk)
        b_gn = tl.load(gk + last_idx * H*K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_kappa = tl.load(kappa + p_k, mask=m_tk, other=0.).to(tl.float32)
        # the write keys decayed to the chunk end, which feed the state recurrence
        b_kg = b_kappa * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_gk), 0)
        tl.store(kg + p_k, b_kg.to(kg.dtype.element_ty), mask=m_tk)
        b_w = tl.dot(b_A, b_kb, input_precision=DOT_PRECISION)
        tl.store(w + p_k, b_w.to(w.dtype.element_ty), mask=m_tk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # two warps spill registers and run about 7x slower
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_dqkg(
    q,
    k,
    kappa,
    v_new,
    g,
    A,
    h,
    do,
    dh,
    du,
    dq,
    dk,
    dkappa,
    dv,
    dg,
    dw,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dkappa += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K
    v_new += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    du += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K*V
    dh += (i_tg * H + i_h) * K*V

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    last_idx = min(i_t * BT + BT, T) - 1
    m_last = o_t == last_idx

    # [BT, BT], the transposed WY inverse
    b_A = tl.load(A + o_t[None, :] * (H*BT) + o_i[:, None], mask=m_t[None, :], other=0.).to(tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkappa = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_tv = m_t[:, None] & m_v[None, :]
            p_v = o_t[:, None] * (H*V) + o_v[None, :]
            b_v_new = tl.load(v_new + p_v, mask=m_tv, other=0.).to(tl.float32)
            b_do = tl.load(do + p_v, mask=m_tv, other=0.).to(tl.float32)
            b_du = tl.load(du + p_v, mask=m_tv, other=0.).to(tl.float32)
            # [BV, BK], the states loaded in memory order and transposed, which is faster than a strided load
            m_kv = m_k[:, None] & m_v[None, :]
            b_h = tl.trans(tl.load(h + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.).to(tl.float32))
            b_dh = tl.trans(tl.load(dh + o_k[:, None] * V + o_v[None, :], mask=m_kv, other=0.).to(tl.float32))
            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h, input_precision=DOT_PRECISION_FAST)
            b_dkappa += tl.dot(b_v_new, b_dh, input_precision=DOT_PRECISION_FAST)
            b_dw += tl.dot(b_du.to(do.dtype.element_ty), b_h.to(do.dtype.element_ty))
            tl.debug_barrier()
            if i_k == 0:
                b_dv = tl.dot(b_A.to(do.dtype.element_ty), b_du.to(do.dtype.element_ty))
                tl.store(dv + p_v, b_dv.to(dv.dtype.element_ty), mask=m_tv)

        p_k = o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(k + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_g = tl.load(g + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_gn = tl.load(g + last_idx * H*K + o_k, mask=m_k, other=0.).to(tl.float32)
        b_gk_exp = exp2(b_g)
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dkappa *= tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0.)
        b_kg = b_k * b_gk_exp
        b_dw = -b_dw
        b_dkgb = tl.dot(b_A.to(do.dtype.element_ty), b_dw.to(do.dtype.element_ty))
        tl.store(dw + p_k, b_dw.to(dw.dtype.element_ty), mask=m_tk)

        b_q = tl.load(q + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_kappa = tl.load(kappa + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_kdk = b_kappa * b_dkappa
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb
        b_dk = b_dkgb * b_gk_exp
        tl.store(dq + p_k, b_dq.to(dq.dtype.element_ty), mask=m_tk)
        tl.store(dk + p_k, b_dk.to(dk.dtype.element_ty), mask=m_tk)
        tl.store(dkappa + p_k, b_dkappa.to(dkappa.dtype.element_ty), mask=m_tk)
        tl.store(dg + p_k, b_dg.to(dg.dtype.element_ty), mask=m_tk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_dA(
    k,
    v,
    g,
    A,
    du,
    dw,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    du += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    dA += (bos * H + i_h) * BT

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T

    # the gradient of the strictly lower key-key block, all at tf32 on fp32 operands
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v < V)[None, :]
        b_v = tl.load(v + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_du = tl.load(du + o_t[:, None] * (H*V) + o_v[None, :], mask=m_tv, other=0.).to(tl.float32)
        b_dA += tl.dot(b_du, tl.trans(b_v), input_precision=DOT_PRECISION_FAST)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        p_k = o_t[:, None] * (H*K) + o_k[None, :]
        b_dw = tl.load(dw + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_k = tl.load(k + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_g = tl.load(g + p_k, mask=m_tk, other=0.).to(tl.float32)
        b_dA += tl.dot(b_dw, tl.trans(b_k * exp2(b_g)), input_precision=DOT_PRECISION_FAST)

    m_A = (o_i[:, None] > o_i[None, :]) & (m_t[:, None] & m_t[None, :])
    b_dA = tl.where(m_A, b_dA, 0.)
    # [BT, BT], the transposed WY inverse
    b_A = tl.load(A + o_t[None, :] * (H*BT) + o_i[:, None], mask=m_t[None, :], other=0.).to(tl.float32)
    b_dA = tl.dot(b_dA, b_A, input_precision=DOT_PRECISION_FAST)
    b_dA = tl.dot(b_A, b_dA, input_precision=DOT_PRECISION_FAST)
    b_dA = tl.where(m_A, -b_dA, 0.)
    tl.store(dA + o_t[:, None] * (H*BT) + o_i[None, :], b_dA.to(dA.dtype.element_ty), mask=m_t[:, None])


def recompute_w_u_fwd(
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)

    w = torch.empty_like(k, dtype=torch.float)
    u = torch.empty_like(v, dtype=torch.float)
    kg = torch.empty_like(k, dtype=torch.float)
    qg = torch.empty_like(q, dtype=torch.float) if q is not None else None
    recompute_w_u_fwd_kernel[(NT, B * H)](
        q=q,
        k=k,
        kappa=kappa,
        qg=qg,
        kg=kg,
        v=v,
        w=w,
        u=u,
        A=A,
        gk=gk,
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
    return w, u, kg, qg


def prepare_wy_repr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    du: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dkappa = torch.empty_like(kappa, dtype=torch.float)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    dw = torch.empty_like(k, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    prepare_wy_repr_bwd_kernel_dqkg[(NT, B * H)](
        q=q,
        k=k,
        kappa=kappa,
        v_new=v_new,
        g=g,
        A=A,
        h=h,
        do=do,
        dh=dh,
        du=du,
        dq=dq,
        dk=dk,
        dkappa=dkappa,
        dv=dv,
        dg=dg,
        dw=dw,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    prepare_wy_repr_bwd_kernel_dA[(NT, B * H)](
        k=k,
        v=v,
        g=g,
        A=A,
        du=du,
        dw=dw,
        dA=dA,
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
    return dq, dk, dkappa, dv, dg, dA
