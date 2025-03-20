# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.exp import safe_exp


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['BK'],
)
@triton.jit(do_not_specialize=['T'])
def fwd_prepare_wy_repr_kernel_chunk32(
    k,
    g,
    beta,
    Aw,
    Au,
    offsets,
    indices,
    T,
    K: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
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

    b_Aw = tl.zeros([BC, BC], dtype=tl.float32)
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_beta = tl.load(p_beta, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_Aw += tl.dot(b_kb, tl.trans(b_k))

    b_Aw = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_Aw, 0)

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,))
    b_Au = b_Aw * safe_exp(b_g[:, None] - b_g[None, :])

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_aw = tl.sum(tl.where(mask[:, None], b_Aw, 0), 0)
        b_au = tl.sum(tl.where(mask[:, None], b_Au, 0), 0)
        b_aw = b_aw + tl.sum(b_aw[:, None] * b_Aw, 0) * (tl.arange(0, BC) < i)
        b_au = b_au + tl.sum(b_au[:, None] * b_Au, 0) * (tl.arange(0, BC) < i)
        b_Aw = tl.where(mask[:, None], b_aw, b_Aw)
        b_Au = tl.where(mask[:, None], b_au, b_Au)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_Aw += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_Au += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    if HEAD_FIRST:
        p_Aw = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Au = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    else:
        p_Aw = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Au = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    tl.store(p_Aw, b_Aw.to(p_Aw.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Au, b_Au.to(p_Au.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['BK'],
)
@triton.jit(do_not_specialize=['T'])
def fwd_prepare_wy_repr_kernel_chunk64(
    k,
    g,
    beta,
    Aw,
    Au,
    offsets,
    indices,
    T,
    K: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
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

    b_Aw = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aw2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aw3 = tl.zeros([BC, BC], dtype=tl.float32)
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh*T, (T,), (1,), (i_t * BT,), (BC,), (0,))
        p_beta2 = tl.make_block_ptr(beta + i_bh*T, (T,), (1,), (i_t * BT + BC,), (BC,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BC,), (0,))
        p_beta2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + BC,), (BC,), (0,))

    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_beta2 = tl.load(p_beta2, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_kb2 = (b_k2 * b_beta2[:, None]).to(b_k2.dtype)
        b_Aw += tl.dot(b_kb, tl.trans(b_k))
        b_Aw2 += tl.dot(b_kb2, tl.trans(b_k2))
        b_Aw3 += tl.dot(b_kb2, tl.trans(b_k))

    b_Aw = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_Aw, 0)
    b_Aw2 = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_Aw2, 0)

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh*T, (T,), (1,), (i_t * BT,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + i_bh*T, (T,), (1,), (i_t * BT + BC,), (BC,), (0,))
    else:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT + BC,), (BC,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g2 = tl.load(p_g2, boundary_check=(0,))

    mask_c = tl.arange(0, BC)[:, None] >= tl.arange(0, BC)[None, :]
    mask_g = i_t * BT + tl.arange(0, BC) < T
    mask_g2 = i_t * BT + BC + tl.arange(0, BC) < T

    b_Au = tl.where(mask_g[None, :] & mask_c, b_Aw * safe_exp(b_g[:, None] - b_g[None, :]), 0)
    b_Au2 = tl.where(mask_g2[None, :] & mask_c, b_Aw2 * safe_exp(b_g2[:, None] - b_g2[None, :]), 0)
    b_Au3 = tl.where(mask_g[None, :], b_Aw3 * safe_exp(b_g2[:, None] - b_g[None, :]), 0)

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_aw = tl.sum(tl.where(mask[:, None], b_Aw, 0), 0)
        b_aw2 = tl.sum(tl.where(mask[:, None], b_Aw2, 0), 0)
        b_au = tl.sum(tl.where(mask[:, None], b_Au, 0), 0)
        b_au2 = tl.sum(tl.where(mask[:, None], b_Au2, 0), 0)
        b_aw = b_aw + tl.sum(b_aw[:, None] * b_Aw, 0) * (tl.arange(0, BC) < i)
        b_aw2 = b_aw2 + tl.sum(b_aw2[:, None] * b_Aw2, 0) * (tl.arange(0, BC) < i)
        b_au = b_au + tl.sum(b_au[:, None] * b_Au, 0) * (tl.arange(0, BC) < i)
        b_au2 = b_au2 + tl.sum(b_au2[:, None] * b_Au2, 0) * (tl.arange(0, BC) < i)
        b_Aw = tl.where(mask[:, None], b_aw, b_Aw)
        b_Aw2 = tl.where(mask[:, None], b_aw2, b_Aw2)
        b_Au = tl.where(mask[:, None], b_au, b_Au)
        b_Au2 = tl.where(mask[:, None], b_au2, b_Au2)
    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_Aw += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_Aw2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    # improve precision by disallowing tf32.
    b_Aw3 = -tl.dot(tl.dot(b_Aw2, b_Aw3, allow_tf32=False), b_Aw, allow_tf32=False)
    b_Au += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_Au2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_Au3 = -tl.dot(tl.dot(b_Au2, b_Au3, allow_tf32=False), b_Au, allow_tf32=False)

    if HEAD_FIRST:
        p_Aw1 = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Aw2 = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_Aw3 = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_Aw4 = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
        p_Au1 = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Au2 = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_Au3 = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_Au4 = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
    else:
        p_Aw1 = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Aw2 = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_Aw3 = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_Aw4 = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
        p_Au1 = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Au2 = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_Au3 = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_Au4 = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))

    tl.store(p_Aw1, b_Aw.to(p_Aw1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aw2, b_Aw2.to(p_Aw2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aw3, b_Aw3.to(p_Aw3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aw4, tl.zeros([BC, BC], dtype=tl.float32).to(p_Aw4.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Au1, b_Au.to(p_Au1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Au2, b_Au2.to(p_Au2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Au3, b_Au3.to(p_Au3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Au4, tl.zeros([BC, BC], dtype=tl.float32).to(p_Au4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def fwd_recompute_w_u_kernel(
    k,
    v,
    beta,
    w,
    u,
    Aw,
    Au,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
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
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_Au = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_Au = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_Au = tl.load(p_Au, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Au, b_vb, allow_tf32=False)
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    b_Au = None
    if HEAD_FIRST:
        p_Aw = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_Aw = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_Aw = tl.load(p_Aw, boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_Aw, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def fwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(BT, 32)
    BK = min(triton.next_power_of_2(K), 64)
    # bf16 should be good enough.
    Aw = torch.empty(B, *((H, T) if head_first else (T, H)), BT, device=k.device, dtype=k.dtype)
    Au = torch.empty(B, *((H, T) if head_first else (T, H)), BT, device=k.device, dtype=k.dtype)

    fwd_fn = fwd_prepare_wy_repr_kernel_chunk64 if BT == 64 else fwd_prepare_wy_repr_kernel_chunk32
    fwd_fn[(NT, B*H)](
        k=k,
        g=g,
        beta=beta,
        Aw=Aw,
        Au=Au,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        BC=BC,
        HEAD_FIRST=head_first
    )
    w, u = fwd_recompute_w_u(
        k=k,
        v=v,
        beta=beta,
        Aw=Aw,
        Au=Au,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return w, u, Aw, Au


def fwd_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
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
    w = torch.empty_like(k)
    fwd_recompute_w_u_kernel[(NT, B*H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        Aw=Aw,
        Au=Au,
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


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def bwd_prepare_wy_repr_kernel(
    k,
    v,
    beta,
    g,
    Aw,
    Au,
    dw,
    du,
    dk,
    dv,
    dbeta,
    dg,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(Aw + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    else:
        p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
        b_dk_beta = tl.dot(b_A, b_dw, allow_tf32=False)
        b_dk = b_dk_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA, 0).to(k.dtype.element_ty)

    if HEAD_FIRST:
        p_A = tl.make_block_ptr(Au + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    else:
        p_A = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA2 = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_du = tl.make_block_ptr(du + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA2 += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dA2 = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA2, 0)
    b_dA2 = tl.dot(b_dA2.to(b_A.dtype), b_A)
    b_dA2 = tl.dot(b_A, b_dA2.to(b_A.dtype))
    b_dA2 = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA2, 0).to(k.dtype.element_ty)
    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_dA2 *= safe_exp(b_g[:, None] - b_g[None, :])
    b_dA += b_dA2
    b_dA = b_dA.to(k.dtype.element_ty)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_A += tl.dot(b_k_beta, tl.trans(b_k))
        b_dk_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += tl.dot(tl.trans(b_dA), b_k_beta, allow_tf32=False)
        b_dk += b_dk_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    b_dA2 *= b_A
    b_dg = tl.sum(b_dA2, axis=1) - tl.sum(b_dA2, axis=0)
    if HEAD_FIRST:
        p_dg = tl.make_block_ptr(dg + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_dg = tl.make_block_ptr(dg + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dbeta = tl.make_block_ptr(dbeta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


def bwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
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

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    dg = torch.empty_like(g)
    bwd_prepare_wy_repr_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        Aw=Aw,
        Au=Au,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        dbeta=dbeta,
        dg=dg,
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
    return dk, dv, dbeta, dg
