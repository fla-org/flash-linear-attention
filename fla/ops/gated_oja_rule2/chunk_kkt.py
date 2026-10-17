# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Intra-chunk score matrix A = tril((b * v) exp(gv) v^T, -1) for Oja2.
#
# Oja2 runs the delta rule on the slot code, so the "key" of this kernel is the slot code of shape [B, T, H, M].
# The per-slot erase gate ``b`` [B, T, H, M] is folded into the row side only, which makes A asymmetric.

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja2_kkt_fwd_kernel_intra_sub_inter(
    k,
    g,
    b,
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
    i_t, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
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
    b += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_r = o_r < T

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_kj = i_t * BT + i_j * BC + tl.arange(0, BC)
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_rk = m_r[:, None] & m_k[None, :]
        m_kj = m_k[:, None] & (o_kj[None, :] < T)
        p_k = k + o_r[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_r[:, None] * (H*K) + o_k[None, :]
        p_b = b + o_r[:, None] * (H*K) + o_k[None, :]
        p_kt = k + o_k[:, None] + o_kj[None, :] * (H*K)
        p_gk = g + o_k[:, None] + o_kj[None, :] * (H*K)

        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H*K + o_k, mask=m_k, other=0)
        # [BC, BK]
        b_g = tl.load(p_g, mask=m_rk, other=0.0)
        b_k = tl.load(p_k, mask=m_rk, other=0.0)
        b_b = tl.load(p_b, mask=m_rk, other=0.0)
        b_k = (b_b * b_k).to(b_k.dtype) * exp(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, mask=m_kj, other=0.0)
        b_kt = tl.load(p_kt, mask=m_kj, other=0.0) * exp(b_gn[:, None] - b_gk)
        # [BC, BC]
        b_A = tl.dot(b_k, b_kt, b_A)

    o_Aj = i_j * BC + tl.arange(0, BC)
    p_A = A + o_r[:, None] * (H*BT) + o_Aj[None, :]
    tl.store(p_A, b_A.to(A.dtype.element_ty), mask=m_r[:, None] & (o_Aj[None, :] < BT))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja2_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    b,
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
    i_t, i_i, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
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
    p_b = b + (bos * H + i_h) * K + o_r[:, None] * (H*K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_rk, other=0.0)
    b_b = tl.load(p_b, mask=m_rk, other=0.0)
    b_k = (b_b * b_k).to(b_k.dtype)
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
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_oja2_kkt_bwd_kernel_gk(
    k,
    g,
    b,
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
    NK = tl.cdiv(K, BK)
    i_k, i_c, i_bh = tl.program_id(0).to(tl.int64) % NK, tl.program_id(0).to(tl.int64) // NK, tl.program_id(1).to(tl.int64)
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
    b += (bos * H + i_h) * K

    dA += (bos * H + i_h) * BT
    dk += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    # db is laid out as [NK, B, T, H, BK]: one BK-wide slice per K block
    db += ((i_k * all + bos) * H + i_h) * BK

    o_r = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_r = o_r < T
    m_rk = m_r[:, None] & m_k[None, :]
    p_g = g + o_r[:, None] * (H*K) + o_k[None, :]
    p_b = b + o_r[:, None] * (H*K) + o_k[None, :]
    # [BC, BK]
    b_g = tl.load(p_g, mask=m_rk, other=0.0)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    b_b = tl.load(p_b, mask=m_rk, other=0.0)
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
            b_dkb = tl.dot(b_dA, b_kg.to(b_dA.dtype)) * exp(b_g - b_gn[None, :])
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
    # the erase gate only touches the row side, so its gradient is the pre-gate dk times k
    b_db = b_dk * b_k
    b_dk *= b_b
    p_db = db + o_r[:, None] * (H*BK) + tl.arange(0, BK)[None, :]
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_r[:, None])

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
            p_bj = b + o_j[:, None] * (H*K) + o_k[None, :]
            p_dA = dA + o_dAi[:, None] + o_j[None, :] * (H*BT)

            # [BC, BK]
            b_bj = tl.load(p_bj, mask=m_jk, other=0.0)
            b_kb = tl.load(p_k, mask=m_jk, other=0.0).to(tl.float32) * b_bj
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kbg = b_kb * tl.where(m_j[:, None], exp(b_gk - b_gn[None, :]), 0)
            # [BC, BC]
            b_dA = tl.load(p_dA, mask=(o_dAi[:, None] < BT) & m_j[None, :], other=0.0)
            # [BC, BK]
            # fp32 operands are required here to keep precision
            b_dkt = tl.dot(b_dA, b_kbg, b_dkt)
        b_dkt *= exp(b_gn[None, :] - b_g)
    o_dA = (i_t * BT + i_i * BC) * H*BT + i_i * BC + o_i
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k
    p_bj = b + (i_t * BT + i_i * BC) * H*K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * H*BT)
        # [BK,]
        b_bj = tl.load(p_bj, mask=m_k, other=0).to(tl.float32)
        b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * b_bj
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        b_kbgj = b_kbj[None, :] * exp(b_gkj[None, :] - b_g)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dkt += tl.where(m_i, b_dA[:, None] * b_kbgj, 0.)

        p_kj += H*K
        p_gkj += H*K
        p_bj += H*K
    b_dg = (b_dk - b_dkt) * b_k
    b_dk += b_dkt

    p_dk = dk + o_r[:, None] * (H*K) + o_k[None, :]
    p_dg = dg + o_r[:, None] * (H*K) + o_k[None, :]
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_rk)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_rk)


def chunk_oja2_kkt_fwd(
    k: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute the strictly lower triangular score matrix of the slot code, with the erase gate on the row side.

    Args:
        k (torch.Tensor):
            The slot code of shape `[B, T, H, M]`.
        gk (torch.Tensor):
            The chunk-local cumulative log-decay of shape `[B, T, H, M]`.
        b (torch.Tensor):
            The per-slot erase gate of shape `[B, T, H, M]`.
        cu_seqlens (torch.LongTensor, Optional):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        chunk_indices (torch.LongTensor, Optional):
            Pre-computed chunk indices. Default: `None`.
        chunk_size (int, Optional):
            The chunk size. Default: 64.
        output_dtype (torch.dtype, Optional):
            The dtype of the output tensor. Default: `torch.float32`.

    Returns:
        The score matrix of shape `[B, T, H, BT]`, where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = (NT, NC * NC, B * H)
    chunk_oja2_kkt_fwd_kernel_intra_sub_inter[grid](
        k=k,
        g=gk,
        b=b,
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
    chunk_oja2_kkt_fwd_kernel_intra_sub_intra[grid](
        k=k,
        g=gk,
        b=b,
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


def chunk_oja2_kkt_bwd_gk(
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Backward of `chunk_oja2_kkt_fwd`.

    Returns:
        The gradients of the slot code, the cumulative log-decay and the erase gate, each of shape `[B, T, H, M]`.
    """
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
    db = k.new_empty(NK, B, T, H, BK, dtype=torch.float)
    grid = (NK * NT * NC, B * H)
    chunk_oja2_kkt_bwd_kernel_gk[grid](
        k=k,
        g=g,
        b=b,
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
    # [NK, B, T, H, BK] -> [B, T, H, NK * BK], then drop the padding of the last K block
    db = db.permute(1, 2, 3, 0, 4).reshape(B, T, H, NK * BK)[..., :K].contiguous()

    return dk, dg, db
