
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

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
    key=["BK"]
)
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk32(
    k,
    beta,
    A,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
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
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)

    b_A = -tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]

    if HEAD_FIRST:
        p_A = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, (b_A).to(p_A.dtype.element_ty), boundary_check=(0, 1))
    b_A = b_A.to(k.dtype.element_ty)


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["BK"]
)
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk64(
    k,
    beta,
    A,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
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

    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BC,), (0,))
        p_beta2 = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT + BC,), (BC,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BC,), (0,))
        p_beta2 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + BC,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_beta2 = tl.load(p_beta2, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_kb2 = (b_k2 * b_beta2[:, None]).to(b_k2.dtype)
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)
        b_A2 += tl.dot(b_kb2, tl.trans(b_k2), allow_tf32=False)
        b_A3 += tl.dot(b_kb2, tl.trans(b_k), allow_tf32=False)

    b_A = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)
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
    b_A3 = -tl.dot(tl.dot(b_A2, b_A3, allow_tf32=False), b_A, allow_tf32=False)

    if HEAD_FIRST:
        p_A1 = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_A2 = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_A3 = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_A4 = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
    else:
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
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT", "BK", "BV"]
)
@triton.jit
def fwd_recompute_w_u_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
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
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False)
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A.to(b_kb.dtype), b_kb, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["BT", "BK", "BV"]
)
@triton.jit
def bwd_prepare_wy_repr_kernel(
    k,
    v,
    beta,
    A,
    dw,
    du,
    dk,
    dv,
    dbeta,
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T*BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
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
        b_dA += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
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

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

        b_dk_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += tl.dot(tl.trans(b_dA), b_k_beta, allow_tf32=False)
        b_dk += b_dk_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    if HEAD_FIRST:
        p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_dbeta = tl.make_block_ptr(dbeta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


def fwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
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
    BC = min(BT, 32)
    BK = min(triton.next_power_of_2(K), 64)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    A = torch.empty(B, *((H, T) if head_first else (T, H)), BT, device=k.device, dtype=k.dtype)
    fwd_fn = fwd_prepare_wy_repr_kernel_chunk64 if BT == 64 else fwd_prepare_wy_repr_kernel_chunk32
    fwd_fn[(NT, B * H)](
        k=k,
        beta=beta,
        A=A,
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
        A=A,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return w, u, A


def fwd_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
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
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    fwd_recompute_w_u_kernel[(NT, B*H)](
        k,
        v,
        beta,
        w,
        u,
        A,
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


def bwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    bwd_prepare_wy_repr_kernel[(NT, B * H)](
        k,
        v,
        beta,
        A,
        dw,
        du,
        dk,
        dv,
        dbeta,
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
    return dk, dv, dbeta
