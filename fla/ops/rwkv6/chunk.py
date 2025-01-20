# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.gla.chunk import (chunk_gla_bwd_dA, chunk_gla_bwd_dv,
                               chunk_gla_fwd_o_gk)
from fla.utils import contiguous


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S', 'BT']
)
@triton.jit
def chunk_rwkv6_fwd_cumsum_kernel(
    s,
    oi,
    oe,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1., 0.)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_oi = tl.make_block_ptr(oi + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_oe = tl.make_block_ptr(oe + i_bh * T*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_oi = tl.make_block_ptr(oi + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_oe = tl.make_block_ptr(oe + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_oi = tl.dot(m_i, b_s, allow_tf32=False)
    b_oe = tl.dot(m_e, b_s, allow_tf32=False)
    tl.store(p_oi, b_oi.to(p_oi.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_oe, b_oe.to(p_oe.dtype.element_ty), boundary_check=(0, 1))


def chunk_rwkv6_fwd_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    gi, ge = torch.empty_like(g, dtype=torch.float), torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g,
        gi,
        ge,
        offsets,
        indices,
        T=T,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return gi, ge


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
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    gi,  # cumulative decay inclusive
    ge,  # cumulative decay exclusive
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

    m_i = i_t * BT + i_i * BC + tl.arange(0, BC) < T

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_gq = tl.make_block_ptr(ge + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gk = tl.make_block_ptr(gi + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(gi + (i_bh * T + i_t * BT + i_i * BC - 1) * K + o_k, BK), BK)
        else:
            p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_gq = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gk = tl.make_block_ptr(gi + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
            p_gn = tl.max_contiguous(tl.multiple_of(gi + (bos + i_t * BT + i_i * BC - 1) * H*K + i_h * K + o_k, BK), BK)

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.where(m_i[:, None] & m_k, tl.load(p_gq, boundary_check=(0, 1)), float('-inf'))
        b_qg = b_q * tl.exp(b_gq - b_gn[None, :]) * scale
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
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    gi,
    ge,
    u,
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
        p_g = tl.make_block_ptr(ge + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_qj = tl.max_contiguous(tl.multiple_of(q + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_kj = tl.max_contiguous(tl.multiple_of(k + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(gi + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
    else:
        o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_j * BC
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
        p_qj = tl.max_contiguous(tl.multiple_of(q + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)
        p_kj = tl.max_contiguous(tl.multiple_of(k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(gi + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (0,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_kj[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        b_A = tl.where(o_i != j, b_A, tl.sum(b_qj * b_kj * b_u * scale))
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_qj += K if HEAD_FIRST else H*K
        p_kj += K if HEAD_FIRST else H*K
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
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    gi,
    ge,
    u,
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
        p_g = tl.make_block_ptr(ge + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_qj = tl.max_contiguous(tl.multiple_of(q + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_kj = tl.max_contiguous(tl.multiple_of(k + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(gi + (i_bh * T + i_t * BT + i_j * BC) * K + o_k, BK), BK)
    else:
        o_A = (i_k * all + bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BC + i_h * BC
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_qj = tl.max_contiguous(tl.multiple_of(q + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)
        p_kj = tl.max_contiguous(tl.multiple_of(k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(gi + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_kj[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        b_A = tl.where(o_i != j, b_A, tl.sum(b_qj * b_kj * b_u * scale))
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_qj += K if HEAD_FIRST else H*K
        p_kj += K if HEAD_FIRST else H*K
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
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge(
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
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT']
)
@triton.jit
def chunk_rwkv6_bwd_kernel_dh(
    q,
    gi,
    ge,
    do,
    dh,
    dht,
    dh0,
    offsets,
    chunk_offsets,
    scale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_nh // NG
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        if HEAD_FIRST:
            p_dh = tl.make_block_ptr(dh + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_do = tl.make_block_ptr(do + i_nh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if HEAD_FIRST:
            p_gk = tl.make_block_ptr(ge + i_bg * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gi + (i_bg * T + last_idx) * K + i_k * BK + tl.arange(0, BK)
        else:
            p_gk = tl.make_block_ptr(ge + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gi + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)

        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_q = (b_q * tl.exp(b_gk) * scale).to(b_q.dtype)
        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
        b_dh *= tl.exp(b_gk_last)[:, None]
        b_dh += tl.dot(b_q, b_do)

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BK", "NC", "BT"],
)
@triton.jit
def chunk_rwkv6_bwd_kernel_intra(
    q,
    k,
    gi,
    ge,
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
        p_ge = tl.make_block_ptr(ge + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        p_ge = tl.make_block_ptr(ge + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        if HEAD_FIRST:
            p_gn = tl.max_contiguous(tl.multiple_of(gi + (i_bh * T + i_t * BT + i_i * BC - 1) * K + o_k, BK), BK)
        else:
            p_gn = tl.max_contiguous(tl.multiple_of(gi + (bos + i_t * BT + i_i * BC - 1) * H*K + i_h*K + o_k, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = tl.make_block_ptr(gi + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA + i_bh * T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            else:
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = tl.make_block_ptr(gi+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA+(bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * tl.exp(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_ge - b_gn[None, :])

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    if HEAD_FIRST:
        o_dA = i_bh * T*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
        p_kj = tl.max_contiguous(tl.multiple_of(k + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_gkj = tl.max_contiguous(tl.multiple_of(gi + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        o_dA = bos*H*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_i * BC
        p_kj = tl.max_contiguous(tl.multiple_of(k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gkj = tl.max_contiguous(tl.multiple_of(gi + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] > j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_ge - b_gkj[None, :]), 0.)
        p_kj += K if HEAD_FIRST else H*K
        p_gkj += K if HEAD_FIRST else H*K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    if HEAD_FIRST:
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(gi + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(gi + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        if HEAD_FIRST:
            p_gn = gi + i_bh * T*K + (min(i_t * BT + i_i * BC + BC, T) - 1)*K + o_k
        else:
            p_gn = gi + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + i_h*K + o_k
        p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            m_j = (i_t * BT + i_j * BC + tl.arange(0, BC)) < T
            if HEAD_FIRST:
                p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_gq = tl.make_block_ptr(ge + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA + i_bh * T*BT, (BT, T), (1, BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            else:
                p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
                p_gq = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
                p_dA = tl.make_block_ptr(dA + (bos*H+i_h)*BT, (BT, T), (1, H*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.where(m_j[:, None] & m_k, tl.load(p_gq, boundary_check=(0, 1)), float('-inf'))
            b_qg = b_q * tl.exp(b_gq - b_gn[None, :])
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)
    if HEAD_FIRST:
        o_dA = i_bh * T*BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
        p_qj = tl.max_contiguous(tl.multiple_of(q + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_gqj = tl.max_contiguous(tl.multiple_of(ge + (i_bh * T + i_t * BT + i_i * BC) * K + o_k, BK), BK)
        p_dk = tl.make_block_ptr(dk + i_bh*T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    else:
        o_dA = bos*H*BT + (i_t * BT + i_i * BC) * H*BT + i_h * BT + i_i * BC + tl.arange(0, BC)
        p_qj = tl.max_contiguous(tl.multiple_of(q + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_gqj = tl.max_contiguous(tl.multiple_of(ge + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k, BK), BK)
        p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * (1 if HEAD_FIRST else H) * BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] < j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)
        p_qj += K if HEAD_FIRST else H*K
        p_gqj += K if HEAD_FIRST else H*K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


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
def chunk_rwkv6_bwd_kernel_inter(
    q,
    k,
    v,
    h,
    gi,
    ge,
    u,
    do,
    dh,
    dA,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    du,
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
        p_gk = tl.make_block_ptr(ge + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gi = tl.make_block_ptr(gi + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(gi + i_bh * T*K + (min(T, i_t * BT + BT)-1) * K + o_k, BK), BK)
    else:
        p_gk = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gi = tl.make_block_ptr(gi + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(gi + (bos + min(T, i_t * BT + BT)-1) * H*K + i_h * K + o_k, BK), BK)
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
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * tl.exp(b_gn[None, :] - b_gi)

    o_i = tl.arange(0, BT)
    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dA_dig = dA + (i_bh * T + i_t * BT + o_i) * BT + o_i
    else:
        p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dA_dig = dA + ((bos + i_t * BT + o_i) * H + i_h) * BT + o_i
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)

    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :] - b_q * b_dq
    # [BT,]
    b_dA_dig = tl.load(p_dA_dig, mask=(i_t * BT + o_i) < T, other=0)

    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    # scale is already applied to b_dA_diag
    b_dq += (b_dA_dig[:, None] * b_u[None, :] * b_k)
    b_dk += (b_dA_dig[:, None] * b_u[None, :] * b_q)
    b_du = tl.sum(b_dA_dig[:, None] * b_q * b_k, axis=0)
    p_du = tl.make_block_ptr(du + (i_tg * H + i_h) * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    tl.store(p_du, b_du, boundary_check=(0,))

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


def chunk_rwkv6_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    u: torch.Tensor,
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
    chunk_rwkv6_fwd_A_kernel_intra_sub_inter[grid](
        q,
        k,
        gi,
        ge,
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
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra[grid](
            q,
            k,
            gi,
            ge,
            u,
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
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split[grid](
            q,
            k,
            gi,
            ge,
            u,
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
        chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge[grid](
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


def chunk_rwkv6_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
    states_in_fp32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(offsets) - 1, len(indices)
        chunk_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
    NG = HQ // H

    if head_first:
        dh = k.new_empty(B, HQ, NT, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    else:
        dh = k.new_empty(B, NT, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_rwkv6_bwd_kernel_dh[grid](
        q=q,
        gi=gi,
        ge=ge,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        NG=NG,
        HEAD_FIRST=head_first
    )
    return dh, dh0


def chunk_rwkv6_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
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
    chunk_rwkv6_bwd_kernel_intra[grid](
        q,
        k,
        gi,
        ge,
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


def chunk_rwkv6_bwd_dqkgu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    u: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dA: torch.Tensor,
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

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    dg = torch.empty_like(g)
    du = u.new_empty(B * NT, H, K, dtype=torch.float)
    def grid(meta): return (triton.cdiv(K, meta['BK']), NT, B * H)
    chunk_rwkv6_bwd_kernel_inter[grid](
        q,
        k,
        v,
        h,
        gi,
        ge,
        u,
        do,
        dh,
        dA,
        dq,
        dk,
        dq2,
        dk2,
        dg,
        du,
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
    du = du.sum(0)
    return dq2, dk2, dg, du


def chunk_rwkv6_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gi, ge = chunk_rwkv6_fwd_cumsum(g, chunk_size=chunk_size, offsets=offsets, indices=indices, head_first=head_first)
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=gi,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size,
        states_in_fp32=False
    )
    # the intra A is kept in fp32
    # the computation has very marginal effect on the entire throughput
    A = chunk_rwkv6_fwd_intra(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        u=u,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=ge,
        A=A,
        h=h,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return A, h, ht, o


def chunk_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    gi, ge = chunk_rwkv6_fwd_cumsum(g, chunk_size=chunk_size, offsets=offsets, indices=indices, head_first=head_first)
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=gi,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size,
        states_in_fp32=True
    )
    dh, dh0 = chunk_rwkv6_bwd_dh(
        q=q,
        k=k,
        v=v,
        gi=gi,
        ge=ge,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size,
        states_in_fp32=True
    )

    # dq dk in fp32
    dA = chunk_gla_bwd_dA(
        v=v,
        do=do,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dv = chunk_gla_bwd_dv(
        k=k,
        g=gi,
        A=A,
        do=do,
        dh=dh,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dq, dk = chunk_rwkv6_bwd_dqk_intra(
        q=q,
        k=k,
        gi=gi,
        ge=ge,
        dA=dA,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dq, dk, dg, du = chunk_rwkv6_bwd_dqkgu(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        gi=gi,
        ge=ge,
        u=u,
        do=do,
        dh=dh,
        dA=dA,
        dq=dq,
        dk=dk,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return dq, dk, dv, dg, du, dh0


class ChunkRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        u,
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
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        A, h, ht, o = chunk_rwkv6_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            u=u,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )

        ctx.save_for_backward(q, k, v, g, initial_state, A, u)

        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        return o, ht

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        q, k, v, g, initial_state, A, u = ctx.saved_tensors
        chunk_size, scale, offsets, indices, head_first = ctx.chunk_size, ctx.scale, ctx.offsets, ctx.indices, ctx.head_first
        dq, dk, dv, dg, du, dh0 = chunk_rwkv6_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            u=u,
            scale=scale,
            initial_state=initial_state,
            A=A,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), du.to(u), None, dh0, None, None, None


def chunk_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
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
        u (torch.Tensor):
            bonus representations of shape `[H]`.
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
        final_state (Optional[torch.Tensor]):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.rwkv6 import chunk_rwkv6
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> u = torch.randn(H, K, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_rwkv6(q, k, v, g, u,
                                initial_state=h0,
                                output_final_state=True,
                                head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_rwkv6(q, k, v, g, u,
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
    o, final_state = ChunkRWKV6Function.apply(
        q,
        k,
        v,
        g,
        u,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first
    )
    return o, final_state
