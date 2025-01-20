# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.exp import safe_exp
from fla.utils import contiguous, tensor_cache


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
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
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gk = tl.make_block_ptr(g + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(g + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        else:
            p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gk = tl.make_block_ptr(g + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)

    if HEAD_FIRST:
        p_A = tl.make_block_ptr(A + i_bh*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    else:
        p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
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
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    if HEAD_FIRST:
        o_A = i_bh * T*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
    else:
        o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_j * BC
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)

        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K if HEAD_FIRST else H*K
        p_gk += K if HEAD_FIRST else H*K


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"]
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    g,
    A,
    offsets,
    indices,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    if HEAD_FIRST:
        o_A = (i_k * B*H + i_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
    else:
        o_A = (i_k * all + bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BC + i_h * BC
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K if HEAD_FIRST else H*K
        p_gk += K if HEAD_FIRST else H*K


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC"]
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    offsets,
    indices,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_c * BC >= T:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        if HEAD_FIRST:
            p_A = tl.make_block_ptr(A + (i_k*B*H+i_bh)*T*BC, (T, BC), (BC, 1), (i_t*BT + i_c*BC, 0), (BC, BC), (1, 0))
        else:
            p_A = tl.make_block_ptr(A + (i_k*all+bos)*H*BC+i_h*BC, (T, BC), (H*BC, 1), (i_t*BT + i_c*BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    if HEAD_FIRST:
        p_A2 = tl.make_block_ptr(A2 + i_bh*T*BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    else:
        p_A2 = tl.make_block_ptr(A2 + (bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=["BT"],
)
@triton.jit
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    offsets,
    indices,
    scale,
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
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    if HEAD_FIRST:
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "NC", "BT"],
)
@triton.jit
def chunk_gla_bwd_kernel_intra(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        p_g = tl.make_block_ptr(g + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        if HEAD_FIRST:
            p_gn = g + (i_bh * T + i_t * BT + i_i * BC) * K + o_k
        else:
            p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h*K + o_k
        p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            else:
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = tl.make_block_ptr(g+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA+(bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = (b_k * tl.exp(b_gn[None, :] - b_gk))
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    if HEAD_FIRST:
        o_dA = i_bh * T*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
        p_kj = tl.max_contiguous(tl.multiple_of(k + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_gkj = tl.max_contiguous(tl.multiple_of(g + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        o_dA = bos*H*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_i * BC
        p_kj = tl.max_contiguous(tl.multiple_of(k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gkj = tl.max_contiguous(tl.multiple_of(g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g - b_gkj[None, :]), 0.)
        p_kj += K if HEAD_FIRST else H*K
        p_gkj += K if HEAD_FIRST else H*K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    if HEAD_FIRST:
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        if HEAD_FIRST:
            p_gn = g + (i_bh * T + min(i_t * BT + i_i * BC + BC, T) - 1) * K + o_k
        else:
            p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + i_h * K + o_k
        p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            if HEAD_FIRST:
                p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t*BT + i_j*BC, i_k*BK), (BC, BK), (1, 0))
                p_gq = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t*BT + i_j*BC, i_k*BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (BT, T), (1, BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            else:
                p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
                p_gq = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA + (bos*H+i_h)*BT, (BT, T), (1, H*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1))
            b_qg = b_q * safe_exp(b_gq - b_gn[None, :])
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)
    if HEAD_FIRST:
        o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
        p_qj = tl.max_contiguous(tl.multiple_of(q + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_gqj = tl.max_contiguous(tl.multiple_of(g + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_dk = tl.make_block_ptr(dk + i_bh*T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        o_dA = bos*H*BT + (i_t * BT + i_i * BC) * H*BT + i_h * BT + i_i * BC + tl.arange(0, BC)
        p_qj = tl.max_contiguous(tl.multiple_of(q + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gqj = tl.max_contiguous(tl.multiple_of(g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * (1 if HEAD_FIRST else H) * BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)
        p_qj += K if HEAD_FIRST else H*K
        p_gqj += K if HEAD_FIRST else H*K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BV", "BT"],
)
@triton.jit
def chunk_gla_bwd_kernel_dA(
    v,
    do,
    dA,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        else:
            p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v)
    if HEAD_FIRST:
        p_dA = tl.make_block_ptr(dA + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    else:
        p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=["BT"],
)
@triton.jit
def chunk_gla_bwd_kernel_dv(
    k,
    g,
    A,
    do,
    dh,
    dv,
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
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    else:
        p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # (SY 09/17) important to disallow tf32 here to maintain a good precision.
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * T*K + min(i_t * BT + BT, T) * K - K + o_k, BK), BK)
            p_dh = tl.make_block_ptr(dh + (i_bh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_gn = tl.max_contiguous(tl.multiple_of(g + (bos + min(i_t * BT + BT, T) - 1)*H*K + i_h * K + o_k, BK), BK)
            p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = (b_k * b_gn).to(b_k.dtype)
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BT, BV]
        # (SY 09/17) it is ok to have bf16 interchunk gradient contribution here
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=["BT"]
)
@triton.jit
def chunk_gla_bwd_kernel_inter(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    offsets,
    indices,
    scale,
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
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    if HEAD_FIRST:
        p_gk = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * T*K + (min(T, i_t * BT + BT)-1) * K + o_k, BK), BK)
    else:
        p_gk = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + (bos + min(T, i_t * BT + BT)-1) * H*K + i_h * K + o_k, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + i_bh * NT*K*V + i_t * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V + i_t * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BK]
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * tl.exp(b_gn[None, :] - b_gk)

    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    # tl.debug_barrier()
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) + b_dgk[None, :]
    if HEAD_FIRST:
        p_dq = tl.make_block_ptr(dq2 + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk2 + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_dq = tl.make_block_ptr(dq2 + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk2 + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_intra_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)

    A = q.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    grid = (NT, NC * NC, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q,
        k,
        g,
        A,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        HEAD_FIRST=head_first
    )

    grid = (NT, NC, B * H)
    # load the entire [BC, K] blocks into SRAM at once
    if K <= 256:
        BK = triton.next_power_of_2(K)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
            q,
            k,
            g,
            A,
            offsets,
            indices,
            scale,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            HEAD_FIRST=head_first
        )
    # split then merge
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, *((H, T) if head_first else (T, H)), BC, dtype=torch.float)

        grid = (NK, NT * NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](
            q,
            k,
            g,
            A_intra,
            offsets,
            indices,
            scale,
            B=B,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            NC=NC,
            HEAD_FIRST=head_first
        )

        grid = (NT, NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](
            A_intra,
            A,
            offsets,
            indices,
            B=B,
            T=T,
            H=H,
            BT=BT,
            BC=BC,
            NK=NK,
            HEAD_FIRST=head_first
        )
    return A


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_fwd_kernel_o[grid](
        q,
        v,
        g,
        h,
        o,
        A,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return o


def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, V = v.shape
    else:
        B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BV = min(64, triton.next_power_of_2(V))

    dA = v.new_empty(B, *((H, T) if head_first else (T, H)), BT, dtype=torch.float)
    grid = (NT, B * H)
    chunk_gla_bwd_kernel_dA[grid](
        v,
        do,
        dA,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dA


def chunk_gla_bwd_dv(
    k: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K, V = *k.shape, do.shape[-1]
    else:
        B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    dv = torch.empty_like(do)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_bwd_kernel_dv[grid](
        k,
        g,
        A,
        do,
        dh,
        dv,
        offsets,
        indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return dv


def chunk_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K = q.shape
    else:
        B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = (NK, NT * NC, B * H)
    chunk_gla_bwd_kernel_intra[grid](
        q,
        k,
        g,
        dA,
        dq,
        dk,
        offsets,
        indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        HEAD_FIRST=head_first
    )
    return dq, dk


def chunk_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    dg = torch.empty_like(g)
    # work around triton compiler bugs.
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    def grid(meta): return (triton.cdiv(K, meta['BK']), NT, B * H)
    chunk_gla_bwd_kernel_inter[grid](
        q,
        k,
        v,
        h,
        g,
        do,
        dh,
        dq,
        dk,
        dq2,
        dk2,
        dg,
        offsets,
        indices,
        scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return dq2, dk2, dg


def chunk_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: Optional[torch.Tensor],
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, offsets=offsets, head_first=head_first)

    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )

    # the intra A is kept in fp32
    # the computation has very marginal effect on the entire throughput
    A = chunk_gla_fwd_intra_gk(
        q=q,
        k=k,
        g=g_cumsum,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g_cumsum,
        A=A,
        h=h,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return g_cumsum, A, h, ht, o


def chunk_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: Optional[torch.Tensor],
    scale: float,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, offsets=offsets, head_first=head_first)

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            h0=initial_state,
            output_final_state=False,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT,
            states_in_fp32=True
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
        states_in_fp32=True
    )

    dv = chunk_gla_bwd_dv(
        k=k,
        g=g_cumsum,
        A=A,
        do=do,
        dh=dh,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )

    # dq dk in fp32
    dA = chunk_gla_bwd_dA(
        v=v,
        do=do,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dq, dk = chunk_gla_bwd_dqk_intra(
        q=q,
        k=k,
        g=g_cumsum,
        dA=dA,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dq, dk, dg = chunk_gla_bwd_dqkg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g_cumsum,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return dq, dk, dv, dg, dh0


@tensor_cache
def prepare_varlen_inputs(
    offsets: torch.LongTensor,
    chunk_size: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
    return indices


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        offsets,
        head_first
    ):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_varlen_inputs(offsets, chunk_size) if offsets is not None else None
        g_cumsum, A, h, ht, o = chunk_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=None,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
        chunk_size, scale, offsets, indices, head_first = ctx.chunk_size, ctx.scale, ctx.offsets, ctx.indices, ctx.head_first
        dq, dk, dv, dg, dh0 = chunk_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            scale=scale,
            h=None,
            A=A,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None, None


def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]` applied to keys.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gla import chunk_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_gla(q, k, v, g,
                              initial_state=h0,
                              output_final_state=True,
                              head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gla(q, k, v, g,
                                      initial_state=h0,
                                      output_final_state=True,
                                      cu_seqlens=cu_seqlens,
                                      head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens, head_first)
    return o, final_state
