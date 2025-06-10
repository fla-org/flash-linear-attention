
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import check_shared_mem, is_nvidia_hopper

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK']
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (K, T), (1, K*H), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_A += tl.dot(b_a, b_b)

    b_A = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK']
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_a1 = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_a2 = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        p_b1 = tl.make_block_ptr(b + (bos * H + i_h) * K, (K, T), (1, K*H), (i_k * BK, i_t * BT), (BK, BC), (0, 1))
        p_b2 = tl.make_block_ptr(b + (bos * H + i_h) * K, (K, T), (1, K*H), (i_k * BK, i_t * BT + BC), (BK, BC), (0, 1))
        b_a1 = tl.load(p_a1, boundary_check=(0, 1))
        b_a2 = tl.load(p_a2, boundary_check=(0, 1))
        b_b1 = tl.load(p_b1, boundary_check=(0, 1))
        b_b2 = tl.load(p_b2, boundary_check=(0, 1))
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

    p_A1 = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A2 = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
    p_A3 = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
    p_A4 = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
    tl.store(p_A1, b_A.to(p_A1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A2, b_A2.to(p_A2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A3, b_A3.to(p_A3.dtype.element_ty), boundary_check=(0, 1))
    # causal mask
    tl.store(p_A4, tl.zeros([BC, BC], dtype=tl.float32).to(p_A4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BK', 'BV']
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_Aak = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_w = tl.dot(b_A, b_a)
        b_Aak += tl.dot(b_a, tl.trans(b_k))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    b_Aak = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_Aak, 0)
    b_Aak = b_Aak.to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.dot(b_Aak, b_v).to(v.dtype.element_ty)
        b_u = tl.dot(b_A, b_v)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


def prepare_wy_repr_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = a.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    BK = min(triton.next_power_of_2(K), 64)

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
        chunk_size=chunk_size
    )
    return w, u, A


def wu_fwd(
    a: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *a.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(triton.next_power_of_2(K), CONST_TILING)
    BV = min(triton.next_power_of_2(V), CONST_TILING)

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
