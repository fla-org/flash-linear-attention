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
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk32(
    a,
    b,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,  # dummy placeholder
    IS_VARLEN: tl.constexpr,
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
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_a = m_t[:, None] & (o_k[None, :] < K)
        m_b = (o_k[:, None] < K) & m_t[None, :]
        p_a = a + (bos * H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_b = b + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (K*H)
        b_a = tl.load(p_a, mask=m_a, other=0.0)
        b_b = tl.load(p_b, mask=m_b, other=0.0)
        b_A += tl.dot(b_a, b_b)

    b_A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]

    o_A = tl.arange(0, BT)
    p_A = A + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
    a,
    b,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BC], dtype=tl.float32)

    o_c = tl.arange(0, BC)
    o_t1 = i_t * BT + o_c
    o_t2 = i_t * BT + BC + o_c
    m_t1 = o_t1 < T
    m_t2 = o_t2 < T
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_a1 = m_t1[:, None] & (o_k[None, :] < K)
        m_a2 = m_t2[:, None] & (o_k[None, :] < K)
        m_b1 = (o_k[:, None] < K) & m_t1[None, :]
        m_b2 = (o_k[:, None] < K) & m_t2[None, :]
        p_a1 = a + (bos * H + i_h) * K + o_t1[:, None] * (H*K) + o_k[None, :]
        p_a2 = a + (bos * H + i_h) * K + o_t2[:, None] * (H*K) + o_k[None, :]
        p_b1 = b + (bos * H + i_h) * K + o_k[:, None] + o_t1[None, :] * (K*H)
        p_b2 = b + (bos * H + i_h) * K + o_k[:, None] + o_t2[None, :] * (K*H)
        b_a1 = tl.load(p_a1, mask=m_a1, other=0.0)
        b_a2 = tl.load(p_a2, mask=m_a2, other=0.0)
        b_b1 = tl.load(p_b1, mask=m_b1, other=0.0)
        b_b2 = tl.load(p_b2, mask=m_b2, other=0.0)
        b_A += tl.dot(b_a1, b_b1, allow_tf32=False)
        b_A2 += tl.dot(b_a2, b_b2, allow_tf32=False)
        b_A3 += tl.dot(b_a2, b_b1, allow_tf32=False)

    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3, allow_tf32=False), b_A, allow_tf32=False)

    p_A1 = A + (bos*H + i_h) * BT + o_t1[:, None] * (H*BT) + o_c[None, :]
    p_A2 = A + (bos*H + i_h) * BT + o_t2[:, None] * (H*BT) + (BC + o_c)[None, :]
    p_A3 = A + (bos*H + i_h) * BT + o_t2[:, None] * (H*BT) + o_c[None, :]
    p_A4 = A + (bos*H + i_h) * BT + o_t1[:, None] * (H*BT) + (BC + o_c)[None, :]
    tl.store(p_A1, b_A.to(p_A1.dtype.element_ty), mask=m_t1[:, None] & (o_c[None, :] < BT))
    tl.store(p_A2, b_A2.to(p_A2.dtype.element_ty), mask=m_t2[:, None] & ((BC + o_c)[None, :] < BT))
    tl.store(p_A3, b_A3.to(p_A3.dtype.element_ty), mask=m_t2[:, None] & (o_c[None, :] < BT))
    # causal mask
    tl.store(p_A4, tl.zeros([BC, BC], dtype=tl.float32).to(
        p_A4.dtype.element_ty), mask=m_t1[:, None] & ((BC + o_c)[None, :] < BT))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(
    w,
    u,
    a,
    k,
    v,
    A,
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
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
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
    p_A = A + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]

    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_Aak = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_a = a + (bos * H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_w = w + (bos * H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_a = tl.load(p_a, mask=m_k, other=0.0)
        b_w = tl.dot(b_A, b_a)
        b_Aak += tl.dot(b_a, tl.trans(b_k))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)

    b_Aak = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_Aak, 0)
    b_Aak = b_Aak.to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_u = u + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_v = tl.dot(b_Aak, b_v).to(v.dtype.element_ty)
        b_u = tl.dot(b_A, b_v)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)


def prepare_wy_repr_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = a.shape
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    BK = min(max(triton.next_power_of_2(K), 16), 64)

    A = torch.empty(B, T, H, BT, device=a.device, dtype=a.dtype)
    fwd_fn = prepare_wy_repr_fwd_kernel_chunk64 if BT == 64 else prepare_wy_repr_fwd_kernel_chunk32

    fwd_fn[(NT, B * H)](
        a=a,
        b=b,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        BC=BC,
    )
    w, u = wu_fwd(
        a=a,
        v=v,
        k=k,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return w, u, A


def wu_fwd(
    a: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *a.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    u = torch.empty_like(v)
    w = torch.empty_like(a)
    wu_fwd_kernel[(NT, B*H)](
        a=a,
        v=v,
        w=w,
        u=u,
        A=A,
        k=k,
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
    return w, u


fwd_prepare_wy_repr = prepare_wy_repr_fwd

fwd_wu = wu_fwd
