
# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["BT"]
)
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk32(
    A_ab,
    A_ab_inv, 
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr, #placeholder, do not delete
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if HEAD_FIRST:
        p_Aab = tl.make_block_ptr(A_ab + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_Aab_inv = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_Aab = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_Aab_inv = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_ab = tl.load(p_Aab, boundary_check=(0, 1))
    b_A_ab = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A_ab, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A_ab, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A_ab, 0) * (tl.arange(0, BT) < i)
        b_A_ab = tl.where(mask[:, None], b_a, b_A_ab)
    b_A_ab += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    tl.store(p_Aab_inv, b_A_ab.to(p_Aab_inv.dtype.element_ty), boundary_check=(0, 1))
    

@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"]
)
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk64(
    A_ab,
    A_ab_inv,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:

        p_A1 = tl.make_block_ptr(A_ab + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_A2 = tl.make_block_ptr(A_ab + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_A3 = tl.make_block_ptr(A_ab + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_A_inv1 = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_A_inv2 = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_A_inv3 = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_A_inv4 = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
    else:
        p_A1 = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_A2 = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_A3 = tl.make_block_ptr(A_ab + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_A_inv1 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_A_inv2 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_A_inv3 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_A_inv4 = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))

    b_A = tl.load(p_A1, boundary_check=(0, 1))
    b_A2 = tl.load(p_A2, boundary_check=(0, 1))
    b_A3 = tl.load(p_A3, boundary_check=(0, 1))
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
    tl.debug_barrier()
    tl.store(p_A_inv1, b_A.to(p_A_inv1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A_inv2, b_A2.to(p_A_inv2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A_inv3, b_A3.to(p_A_inv3.dtype.element_ty), boundary_check=(0, 1))
    # causal mask
    tl.store(p_A_inv4, tl.zeros([BC, BC], dtype=tl.float32).to(p_A_inv4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4]
    ],
    key=["BT", "BK", "BV"]
)
@triton.jit
def fwd_wu_kernel(
    u,
    w,
    ag,
    v,
    A_ab_inv,
    A_ak,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_A_ab_inv = tl.make_block_ptr(A_ab_inv + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_A_ak = tl.make_block_ptr(A_ak + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_A_ab_inv = tl.make_block_ptr(A_ab_inv + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_A_ak = tl.make_block_ptr(A_ak + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aak = tl.load(p_A_ak, boundary_check=(0, 1))
    o_s = tl.arange(0, BT)
    b_Aab_inv = tl.where(o_s[:, None] >= o_s[None, :], b_Aab_inv, 0)
    b_Aak = tl.where(o_s[:, None] > o_s[None, :], b_Aak, 0)
    # let's use fp32 here
    b_Aak = tl.dot(b_Aab_inv, b_Aak, allow_tf32=True)
    # (SY 01/04) should be bf16 or tf32? To verify.
    b_Aak = b_Aak.to(u.dtype.element_ty)
    b_Aab_inv = b_Aab_inv.to(u.dtype.element_ty)

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_ag = tl.make_block_ptr(ag + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_ag = tl.make_block_ptr(ag + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_w = tl.dot(b_Aab_inv.to(b_ag.dtype), b_ag, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
 

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + i_bh * T * V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Aak.to(b_v.dtype), b_v, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


def fwd_prepare_wy_repr(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K = ag.shape
    else:
        B, T, H, K = ag.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)
    BC = min(BT, 32)
    fwd_fn = fwd_prepare_wy_repr_kernel_chunk64 if BT == 64 else fwd_prepare_wy_repr_kernel_chunk32
    A_ab_inv = torch.empty_like(A_ab)
    fwd_fn[(NT, B * H)](
        A_ab=A_ab,
        A_ab_inv=A_ab_inv,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
        HEAD_FIRST=head_first
    )
    w, u = fwd_wu(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return w, u, A_ab_inv


def fwd_wu(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *ag.shape, v.shape[-1]
    else:
        B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    u = torch.empty_like(v)
    w = torch.empty_like(ag)
    fwd_wu_kernel[(NT, B*H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return w, u


