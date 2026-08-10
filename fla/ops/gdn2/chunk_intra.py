# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Intra-chunk Triton kernels for GDN-2.
#
# Three kernels live here:
#   * chunk_gdn2_fwd_kernel_intra_sub_chunk   - sub-chunk diagonal builder used
#     by the safe-gate path (alternative to the token-parallel diagonal kernel).
#   * chunk_gdn2_fwd_kernel_inter_solve_fused - off-diagonal Akk blocks + the
#     blocked WY triangular solve, fused. Reads diagonal blocks from Akkd, emits
#     the full lower-triangular inverse to Akk.
#   * chunk_gdn2_bwd_kernel_intra             - backward over the within-chunk
#     score matrices, producing dq, dk, db, dg contributions.
#
# Compared with KDA, the GDN-2 variant takes a channel-wise erase gate ``b`` on
# the K axis instead of a per-head scalar ``beta``; the b tensor is folded into
# the relevant tiles directly rather than applied as a row-scalar multiply.

import torch
import triton
import triton.language as tl

from fla.ops.gdn2.chunk_intra_token_parallel import chunk_gdn2_fwd_intra_token_parallel
from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2, gather
from fla.utils import IS_GATHER_SUPPORTED, IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gdn2_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    b,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    b = b + (bos * H + i_h) * K
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    o_k = tl.arange(0, BK)
    m_kc = m_c[:, None] & (o_k[None, :] < K)
    p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
    p_g = g + o_c[:, None] * (H * K) + o_k[None, :]
    p_b = b + o_c[:, None] * (H * K) + o_k[None, :]

    b_q = tl.load(p_q, mask=m_kc, other=0.0)
    b_k = tl.load(p_k, mask=m_kc, other=0.0)
    b_g = tl.load(p_g, mask=m_kc, other=0.0)
    b_b = tl.load(p_b, mask=m_kc, other=0.0)

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_gk)

    b_bk = (b_b.to(tl.float32) * b_k.to(tl.float32)).to(b_k.dtype)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_bk * b_gq, b_kgt)

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    m_Aqk = m_c[:, None] & ((i_i * BC + o_i)[None, :] < BT)
    m_Akk = m_c[:, None] & (o_i[None, :] < BC)
    p_Aqk = Aqk + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :]
    p_Akk = Akk + o_c[:, None] * (H * BC) + o_i[None, :]
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=m_Aqk)
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), mask=m_Akk)

    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gdn2_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    b,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = k + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        p_g0 = g + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_g0 = tl.load(p_g0, mask=m_tc0[:, None] & m_k[None, :], other=0.0).to(tl.float32)

        if i_tc1 < T:
            p_q1 = q + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            p_g1 = g + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            p_b1 = b + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            b_q1 = tl.load(p_q1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_g1 = tl.load(p_g1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_b1 = tl.load(p_b1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_bk1 = b_b1 * b_k1
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_bk1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = q + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                p_g2 = g + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                p_b2 = b + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                b_q2 = tl.load(p_q2, mask=m_tc2[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                b_g2 = tl.load(p_g2, mask=m_tc2[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                b_b2 = tl.load(p_b2, mask=m_tc2[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_bkg2 = (b_b2 * b_k2) * b_gqn2
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_bkg2, b_kgt)
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_bkg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = q + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    p_g3 = g + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    p_b3 = b + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    b_q3 = tl.load(p_q3, mask=m_tc3[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                    b_g3 = tl.load(p_g3, mask=m_tc3[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                    b_b3 = tl.load(p_b3, mask=m_tc3[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_bkg3 = (b_b3 * b_k3) * b_gqn3
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_bkg3, b_kgt)
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_bkg3, b_kgt)
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_bkg3, b_kgt)

    if i_tc1 < T:
        p_Aqk10 = Aqk + (i_tc1 + o_i)[:, None] * (H * BT) + o_i[None, :]
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_tc1[:, None] & (o_i[None, :] < BT))
    if i_tc2 < T:
        p_Aqk20 = Aqk + (i_tc2 + o_i)[:, None] * (H * BT) + o_i[None, :]
        p_Aqk21 = Aqk + (i_tc2 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), mask=m_tc2[:, None] & (o_i[None, :] < BT))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), mask=m_tc2[:, None] & ((BC + o_i)[None, :] < BT))
    if i_tc3 < T:
        p_Aqk30 = Aqk + (i_tc3 + o_i)[:, None] * (H * BT) + o_i[None, :]
        p_Aqk31 = Aqk + (i_tc3 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
        p_Aqk32 = Aqk + (i_tc3 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :]
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), mask=m_tc3[:, None] & (o_i[None, :] < BT))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), mask=m_tc3[:, None] & ((BC + o_i)[None, :] < BT))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), mask=m_tc3[:, None] & ((2 * BC + o_i)[None, :] < BT))

    p_Akk00 = Akkd + (i_tc0 + o_i)[:, None] * (H * BC) + o_i[None, :]
    p_Akk11 = Akkd + (i_tc1 + o_i)[:, None] * (H * BC) + o_i[None, :]
    p_Akk22 = Akkd + (i_tc2 + o_i)[:, None] * (H * BC) + o_i[None, :]
    p_Akk33 = Akkd + (i_tc3 + o_i)[:, None] * (H * BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_tc0[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, mask=m_tc1[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, mask=m_tc2[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, mask=m_tc3[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    p_Akk00 = Akk + (i_tc0 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk10 = Akk + (i_tc1 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk11 = Akk + (i_tc1 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
    p_Akk20 = Akk + (i_tc2 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk21 = Akk + (i_tc2 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
    p_Akk22 = Akk + (i_tc2 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :]
    p_Akk30 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk31 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
    p_Akk32 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :]
    p_Akk33 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + (3 * BC + o_i)[None, :]

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), mask=m_tc0[:, None] & (o_i[None, :] < BT))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), mask=m_tc1[:, None] & (o_i[None, :] < BT))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), mask=m_tc1[:, None] & ((BC + o_i)[None, :] < BT))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), mask=m_tc2[:, None] & (o_i[None, :] < BT))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), mask=m_tc2[:, None] & ((BC + o_i)[None, :] < BT))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), mask=m_tc2[:, None] & ((2 * BC + o_i)[None, :] < BT))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & (o_i[None, :] < BT))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & ((BC + o_i)[None, :] < BT))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & ((2 * BC + o_i)[None, :] < BT))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & ((3 * BC + o_i)[None, :] < BT))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_gdn2_bwd_kernel_intra(
    q,
    k,
    g,
    b,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
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
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B.to(tl.int64) * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all + bos) * H * BK + i_h * BK

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T
    m_kc = m_c[:, None] & m_k[None, :]
    p_g = g + o_c[:, None] * (H * K) + o_k[None, :]
    b_g = tl.load(p_g, mask=m_kc, other=0.0).to(tl.float32)

    p_b = b + o_c[:, None] * (H * K) + o_k[None, :]
    b_b = tl.load(p_b, mask=m_kc, other=0.0)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

    if i_i > 0:
        p_gn = g + i_ti * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dj = m_c[:, None] & ((i_j * BC + o_i)[None, :] < BT)
            p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (H * K) + o_k[None, :]
            p_dAqk = dAqk + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :]
            p_dAkk = dAkk + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :]
            b_kj = tl.load(p_k, mask=m_jk, other=0.0)
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kg = b_kj * exp2(b_gn - b_gk)
            b_dAqk = tl.load(p_dAqk, mask=m_dj, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dj, other=0.0)
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H * BT + i_i * BC
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k

    p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_kc, other=0.0)
    b_k = tl.load(p_k, mask=m_kc, other=0.0)

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        m_dd = m_c[:, None] & ((i_i * BC + o_i)[None, :] < BT)
        p_dAqk = dAqk + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :]
        p_dAkk = dAkk + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :]
        b_dAqk_diag_qk = tl.load(p_dAqk, mask=m_dd, other=0.0).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, mask=m_dd, other=0.0).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H * K
            p_gkj += H * K

    # db gets the elementwise b * k contribution from dk2 (before b is folded
    # back into dk2 itself); then dk2 is rescaled by the channel-wise gate.
    b_db_tile = b_dk2 * b_k
    b_dk2 = b_dk2 * b_b

    o_db = tl.arange(0, BK)
    m_db = m_c[:, None] & (o_db[None, :] < BK)
    p_dq = dq + o_c[:, None] * (H * K) + o_k[None, :]
    p_dq2 = dq2 + o_c[:, None] * (H * K) + o_k[None, :]
    p_db = db + o_c[:, None] * (H * BK) + o_db[None, :]

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_kc, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_kc)
    tl.store(p_db, b_db_tile.to(p_db.dtype.element_ty), mask=m_db)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dj = ((i_i * BC + o_i)[:, None] < BT) & m_j[None, :]
            p_q = q + o_j[:, None] * (H * K) + o_k[None, :]
            p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (H * K) + o_k[None, :]
            p_bj = b + o_j[:, None] * (H * K) + o_k[None, :]
            p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT)
            p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT)
            b_bj = tl.load(p_bj, mask=m_jk, other=0.0)
            b_qj = tl.load(p_q, mask=m_jk, other=0.0)
            b_kbj = tl.load(p_k, mask=m_jk, other=0.0) * b_bj
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, mask=m_dj, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dj, other=0.0)
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0)
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * H * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k
    p_bj_ptr = b + i_ti * H * K + o_k

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_kc, other=0.0)
        m_dk = ((i_i * BC + o_i)[:, None] < BT) & m_c[None, :]
        p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (H * BT)
        p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (H * BT)
        b_dAqk_diag_kk = tl.load(p_dAqk, mask=m_dk, other=0.0).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, mask=m_dk, other=0.0).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
            b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_bj = tl.load(p_bj_ptr, mask=m_k, other=0).to(tl.float32)
            b_kbj = b_kj * b_bj
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H * K
            p_kj += H * K
            p_gkj += H * K
            p_bj_ptr += H * K

    p_dk = dk + o_c[:, None] * (H * K) + o_k[None, :]
    p_dk2 = dk2 + o_c[:, None] * (H * K) + o_k[None, :]
    p_dg = dg + o_c[:, None] * (H * K) + o_k[None, :]
    p_dg2 = dg2 + o_c[:, None] * (H * K) + o_k[None, :]

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_kc, other=0.0)
    b_dk2 += tl.load(p_dk, mask=m_kc, other=0.0)
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_kc)
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_kc)


def chunk_gdn2_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    """Intra-chunk forward: build Aqk, Akk_inv, and the WY auxiliaries (w, u)."""
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular.
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    if safe_gate:
        grid = (NT, triton.cdiv(BT, BC), B * H)
        BK = triton.next_power_of_2(K)
        chunk_gdn2_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            b=b,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
        )
    else:
        chunk_gdn2_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            b=b,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Fused inter-subchunk Akk + solve_tril → full Akk_inv.
    grid = (NT, B * H)
    chunk_gdn2_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        b=b,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=safe_gate,
    )

    w, u, qg, kg = recompute_w_u_fwd_gdn2(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


def chunk_gdn2_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    """Intra-chunk backward: q, k, g, b contributions from dAqk, dAkk.

    Adds the intra-chunk piece of db (channel-wise, K-dim) to the running ``db``
    accumulator passed in.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    db2 = q.new_empty(NK, B, T, H, BK, dtype=torch.float32)
    dg2 = torch.empty_like(dg, dtype=torch.float32)

    grid = (NK * NC, NT, B * H)
    chunk_gdn2_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        b=b,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
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
        SAFE_GATE=safe_gate,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    dq = dq2
    dk = dk2
    db2_combined = db2.permute(1, 2, 3, 0, 4).contiguous().reshape(B, T, H, NK * BK)[..., :K]
    db = db.add_(db2_combined)
    dg = dg2
    return dq, dk, db, dg
