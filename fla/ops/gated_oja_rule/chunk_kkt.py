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


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
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

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = g + bos*H + i_h + o_t * H
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    o_A = tl.arange(0, BT)
    m_AT = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + (bos*H + i_h) * BT + o_t[:, None] * (BT*H) + o_A[None, :]
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_AT)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    k,
    g,
    beta,
    A,
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
    i_t, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_r = o_r < T
    p_b = beta + bos * H + i_h + o_r * H
    b_b = tl.load(p_b, mask=m_r, other=0.0)

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_kj = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_rk = m_r[:, None] & m_k[None, :]
        m_kj = m_k[:, None] & (o_kj[None, :] < T)
        p_k = k + o_r[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_r[:, None] * (H*K) + o_k[None, :]
        b_kt = k + o_k[:, None] + o_kj[None, :] * (H*K)
        p_gk = g + o_k[:, None] + o_kj[None, :] * (H*K)

        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H*K + o_k, mask=m_k, other=0)
        # [BC, BK]
        b_g = tl.load(p_g, mask=m_rk, other=0.0)
        b_k = tl.load(p_k, mask=m_rk, other=0.0) * exp(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, mask=m_kj, other=0.0)
        b_kt = tl.load(b_kt, mask=m_kj, other=0.0) * exp(b_gn[:, None] - b_gk)
        # [BC, BC]
        b_A += tl.dot(b_k, b_kt)
    b_A *= b_b[:, None]

    o_Aj = i_j * BC + tl.arange(0, BC)
    p_A = A + o_r[:, None] * (H*BT) + o_Aj[None, :]
    tl.store(p_A, b_A.to(A.dtype.element_ty), mask=m_r[:, None] & (o_Aj[None, :] < BT))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    beta,
    A,
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
    i_t, i_i, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H*BT + i_h * BT + i_i * BC

    o_r = i_t * BT + i_i * BC + o_i
    m_rk = m_A[:, None] & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + o_r[:, None] * (H*K) + o_k[None, :]
    p_g = g + (bos * H + i_h) * K + o_r[:, None] * (H*K) + o_k[None, :]
    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h

    b_k = tl.load(p_k, mask=m_rk, other=0.0) * tl.load(p_b, mask=m_A, other=0)[:, None]
    b_g = tl.load(p_g, mask=m_rk, other=0.0)

    p_kt = k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_k * b_kt[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A, 0.)

        tl.store(A + o_A + j, b_A, mask=m_A)
        p_kt += H*K
        p_gk += H*K


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BK', 'NC', 'BT'],
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_scaled_dot_kkt_bwd_kernel_gk(
    k,
    g,
    beta,
    dA,
    dk,
    dg,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC

    all = B.to(tl.int64) * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h

    dA += (bos * H + i_h) * BT
    dk += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    db += (i_k * all + bos) * H + i_h

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_r = o_r < T
    m_rk = m_r[:, None] & m_k[None, :]
    p_g = g + o_r[:, None] * (H*K) + o_k[None, :]
    p_b = beta + o_r * H
    # [BC, BK]
    b_g = tl.load(p_g, mask=m_rk, other=0.0)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC]
    b_b = tl.load(p_b, mask=m_r, other=0.0)
    if i_i > 0:
        p_gn = g + (i_t * BT + i_i * BC) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
            m_jk = (o_j[:, None] < T) & m_k[None, :]
            o_dAj = i_j * BC + tl.arange(0, BC)
            p_k = k + o_j[:, None] * (H*K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (H*K) + o_k[None, :]
            p_dA = dA + o_r[:, None] * (H*BT) + o_dAj[None, :]
            # [BC, BK]
            b_k = tl.load(p_k, mask=m_jk, other=0.0)
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kg = b_k * exp(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dA = tl.load(p_dA, mask=m_r[:, None] & (o_dAj[None, :] < BT), other=0.0)
            # [BC, BK]
            b_dkb = tl.dot(b_dA, b_kg) * exp(b_g - b_gn[None, :])
            b_dk += b_dkb

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + o_i) < T
    o_dA = (i_t * BT + i_i * BC + o_i) * H*BT + i_i * BC
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k

    p_k = k + o_r[:, None] * (H*K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_rk, other=0.0)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_dkb = tl.where(m_i, b_dA[:, None] * b_kj[None, :] * exp(b_g - b_gkj[None, :]), 0.)
        b_dk += b_dkb

        p_kj += H*K
        p_gkj += H*K
    b_db = tl.sum(b_dk * b_k, 1)
    b_dk *= b_b[:, None]
    p_db = db + o_r * H
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_r)

    tl.debug_barrier()
    # [BC, BK]
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            o_dAi = i_i * BC + tl.arange(0, BC)
            p_k = k + o_j[:, None] * (H*K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (H*K) + o_k[None, :]
            p_dA = dA + o_dAi[:, None] + o_j[None, :] * (H*BT)
            p_b = beta + o_j * H

            # [BC]
            b_b = tl.load(p_b, mask=m_j, other=0.0)
            # [BC, BK]
            b_kb = tl.load(p_k, mask=m_jk, other=0.0).to(tl.float32) * b_b[:, None]
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kbg = b_kb * tl.where(m_j[:, None], exp(b_gk - b_gn[None, :]), 0)
            # [BC, BC]
            b_dA = tl.load(p_dA, mask=(o_dAi[:, None] < BT) & m_j[None, :], other=0.0)
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt += tl.dot(b_dA, b_kbg)
        b_dkt *= exp(b_gn[None, :] - b_g)
    o_dA = (i_t * BT + i_i * BC) * H*BT + i_i * BC + o_i
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k
    p_bj = beta + (i_t * BT + i_i * BC) * H

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * H*BT)
        # [BK,]
        b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        b_kbgj = b_kbj[None, :] * exp(b_gkj[None, :] - b_g)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dkt += tl.where(m_i, b_dA[:, None] * b_kbgj, 0.)

        p_kj += H*K
        p_gkj += H*K
        p_bj += H
    b_dg = (b_dk - b_dkt) * b_k
    b_dk += b_dkt

    p_dk = dk + o_r[:, None] * (H*K) + o_k[None, :]
    p_dg = dg + o_r[:, None] * (H*K) + o_k[None, :]
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_rk)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_rk)


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_indices (torch.LongTensor):
            Pre-computed chunk indices. Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if gk is None:
        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
        chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
            k=k,
            g=g,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
        )
        return A

    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = (NT, NC * NC, B * H)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )

    grid = (NT, NC, B * H)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
    )
    return A


def chunk_scaled_dot_kkt_bwd_gk(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dk = torch.empty_like(k, dtype=torch.float)
    dg = torch.empty_like(g, dtype=torch.float)
    db = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    grid = (NK, NT * NC, B * H)
    chunk_scaled_dot_kkt_bwd_kernel_gk[grid](
        k=k,
        g=g,
        beta=beta,
        dA=dA,
        dk=dk,
        dg=dg,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    db = db.sum(0)

    return dk, dg, db
