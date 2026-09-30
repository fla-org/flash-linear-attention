# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Precond-KDA chunk-intra kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.precond_kda.chunk_intra import DEFAULT_SOLVE_TRIL_PRECISION, chunk_precond_kda_fwd_kernel_intra_sub_chunk
from fla.ops.precond_kda.chunk_intra_token_parallel import chunk_precond_kda_fwd_intra_token_parallel
from fla.ops.precond_kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_GATHER_SUPPORTED

_NUM_WARPS = 4


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'BK': lambda args: min(triton.next_power_of_2(args['K']), 64),
})
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_fwd_kernel_inter_solve_fused_npu(
    q,
    k,           # Original k (for Akk row side)
    k_precond,   # Preconditioned k (for column side)
    g,
    beta,
    Aqk,
    Akk_diag,
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
    SOLVE_TRIL_DOT_PRECISION: tl.constexpr = 'tf32x3',
    USE_SAFE_GATE: tl.constexpr = False,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Asymmetric version: Aqk = q @ k_precond^T, Akk = k @ k_precond^T
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akk_diag.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akk_diag (fp32)
    4. Does forward substitution on diagonals (skipped when USE_SAFE_GATE)
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
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
    k_precond += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akk_diag += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc1 + o_i
    o_c2 = i_tc2 + o_i
    o_c3 = i_tc3 + o_i
    m_tc0 = o_c0 < T
    m_tc1 = o_c1 < T
    m_tc2 = o_c2 < T
    m_tc3 = o_c3 < T

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

    ################################################################################
    # 1. off-diagonal blocks - ASYMMETRIC: column uses k_precond, row uses k
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        # Column side: load k_precond transposed (for Aqk and Akk column)
        m_kc0 = m_k[:, None] & m_tc0[None, :]
        p_kp0 = k_precond + o_k[:, None] + o_c0[None, :] * (H*K)
        p_g0 = g + o_k[:, None] + o_c0[None, :] * (H*K)
        b_kpt0 = tl.load(p_kp0, mask=m_kc0, other=0.0).to(tl.float32)  # k_precond transposed
        b_gt0 = tl.load(p_g0, mask=m_kc0, other=0.0).to(tl.float32)

        b_kpt1, b_gt1 = b_kpt0, b_gt0
        b_kpt2, b_gt2 = b_kpt0, b_gt0

        if i_tc1 < T:
            # Row side: q for Aqk, original k for Akk
            m_c1k = m_tc1[:, None] & m_k[None, :]
            p_q1 = q + o_c1[:, None] * (H*K) + o_k[None, :]
            p_k1 = k + o_c1[:, None] * (H*K) + o_k[None, :]
            p_kp1 = k_precond + o_c1[:, None] * (H*K) + o_k[None, :]
            p_g1 = g + o_c1[:, None] * (H*K) + o_k[None, :]
            # [BC, BK]
            b_q1 = tl.load(p_q1, mask=m_c1k, other=0.0).to(tl.float32)
            b_k1 = tl.load(p_k1, mask=m_c1k, other=0.0).to(tl.float32)  # Original k for row
            b_kp1 = tl.load(p_kp1, mask=m_c1k, other=0.0).to(tl.float32)  # k_precond for column
            b_g1 = tl.load(p_g1, mask=m_c1k, other=0.0).to(tl.float32)
            # [BK, BC]
            b_kpt1 = tl.trans(b_kp1)  # k_precond transposed
            b_gt1 = tl.trans(b_g1)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            f_tc1 = m_tc1[:, None].to(tl.float32)
            # Mask folded into the exponent: inf * 0 would be NaN.
            b_gqn1 = exp2((b_g1 - b_gn1[None, :]) * f_tc1 - 127.0 * (1.0 - f_tc1)) * f_tc1
            b_qg1 = b_q1 * b_gqn1
            b_kg1 = b_k1 * b_gqn1  # Original k for Akk row
            # [BK, BC]
            b_kpgt = b_kpt0 * exp2(b_gn1[:, None] - b_gt0)  # k_precond for column
            # [BC, BC]
            b_Aqk10 = tl.dot(b_qg1, b_kpgt, b_Aqk10)
            b_Akk10 = tl.dot(b_kg1, b_kpgt, b_Akk10)  # Asymmetric: k @ k_precond^T

        if i_tc2 < T:
            m_c2k = m_tc2[:, None] & m_k[None, :]
            p_q2 = q + o_c2[:, None] * (H*K) + o_k[None, :]
            p_k2 = k + o_c2[:, None] * (H*K) + o_k[None, :]
            p_kp2 = k_precond + o_c2[:, None] * (H*K) + o_k[None, :]
            p_g2 = g + o_c2[:, None] * (H*K) + o_k[None, :]

            b_q2 = tl.load(p_q2, mask=m_c2k, other=0.0).to(tl.float32)
            b_k2 = tl.load(p_k2, mask=m_c2k, other=0.0).to(tl.float32)
            b_kp2 = tl.load(p_kp2, mask=m_c2k, other=0.0).to(tl.float32)
            b_g2 = tl.load(p_g2, mask=m_c2k, other=0.0).to(tl.float32)
            b_kpt2 = tl.trans(b_kp2)
            b_gt2 = tl.trans(b_g2)

            b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            f_tc2 = m_tc2[:, None].to(tl.float32)
            # Mask folded into the exponent: inf * 0 would be NaN.
            b_gqn2 = exp2((b_g2 - b_gn2[None, :]) * f_tc2 - 127.0 * (1.0 - f_tc2)) * f_tc2
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            b_kpgt = b_kpt0 * exp2(b_gn2[:, None] - b_gt0)
            b_Aqk20 = tl.dot(b_qg2, b_kpgt, b_Aqk20)
            b_Akk20 = tl.dot(b_kg2, b_kpgt, b_Akk20)

            b_kpgt = b_kpt1 * exp2(b_gn2[:, None] - b_gt1)
            b_Aqk21 = tl.dot(b_qg2, b_kpgt, b_Aqk21)
            b_Akk21 = tl.dot(b_kg2, b_kpgt, b_Akk21)

        if i_tc3 < T:
            m_c3k = m_tc3[:, None] & m_k[None, :]
            p_q3 = q + o_c3[:, None] * (H*K) + o_k[None, :]
            p_k3 = k + o_c3[:, None] * (H*K) + o_k[None, :]
            p_g3 = g + o_c3[:, None] * (H*K) + o_k[None, :]
            b_q3 = tl.load(p_q3, mask=m_c3k, other=0.0).to(tl.float32)
            b_k3 = tl.load(p_k3, mask=m_c3k, other=0.0).to(tl.float32)
            b_g3 = tl.load(p_g3, mask=m_c3k, other=0.0).to(tl.float32)

            b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            f_tc3 = m_tc3[:, None].to(tl.float32)
            # Mask folded into the exponent: inf * 0 would be NaN.
            b_gqn3 = exp2((b_g3 - b_gn3[None, :]) * f_tc3 - 127.0 * (1.0 - f_tc3)) * f_tc3
            b_qg3 = b_q3 * b_gqn3
            b_kg3 = b_k3 * b_gqn3
            b_kpgt = b_kpt0 * exp2(b_gn3[:, None] - b_gt0)
            b_Aqk30 = tl.dot(b_qg3, b_kpgt, b_Aqk30)
            b_Akk30 = tl.dot(b_kg3, b_kpgt, b_Akk30)

            b_kpgt = b_kpt1 * exp2(b_gn3[:, None] - b_gt1)
            b_Aqk31 = tl.dot(b_qg3, b_kpgt, b_Aqk31)
            b_Akk31 = tl.dot(b_kg3, b_kpgt, b_Akk31)

            b_kpgt = b_kpt2 * exp2(b_gn3[:, None] - b_gt2)
            b_Aqk32 = tl.dot(b_qg3, b_kpgt, b_Aqk32)
            b_Akk32 = tl.dot(b_kg3, b_kpgt, b_Akk32)

    ################################################################################
    # 2. save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = Aqk + o_c1[:, None] * (H*BT) + o_i[None, :]
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_tc1[:, None] & (o_i[None, :] < BC))

        b_b1 = tl.load(beta + bos * H + i_h + o_c1*H, mask=m_tc1, other=0.0).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = Aqk + o_c2[:, None] * (H*BT) + o_i[None, :]
        p_Aqk21 = Aqk + o_c2[:, None] * (H*BT) + (o_i + BC)[None, :]
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), mask=m_tc2[:, None] & (o_i[None, :] < BC))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), mask=m_tc2[:, None] & (o_i[None, :] < BC))

        b_b2 = tl.load(beta + bos * H + i_h + o_c2*H, mask=m_tc2, other=0.0).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = Aqk + o_c3[:, None] * (H*BT) + o_i[None, :]
        p_Aqk31 = Aqk + o_c3[:, None] * (H*BT) + (o_i + BC)[None, :]
        p_Aqk32 = Aqk + o_c3[:, None] * (H*BT) + (o_i + 2*BC)[None, :]
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), mask=m_tc3[:, None] & (o_i[None, :] < BC))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), mask=m_tc3[:, None] & (o_i[None, :] < BC))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), mask=m_tc3[:, None] & (o_i[None, :] < BC))

        b_b3 = tl.load(beta + bos * H + i_h + o_c3*H, mask=m_tc3, other=0.0).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    ################################################################################
    # 3. load diagonal Akk blocks
    ################################################################################
    p_Akk00 = Akk_diag + o_c0[:, None] * (H*BC) + o_i[None, :]
    p_Akk11 = Akk_diag + o_c1[:, None] * (H*BC) + o_i[None, :]
    p_Akk22 = Akk_diag + o_c2[:, None] * (H*BC) + o_i[None, :]
    p_Akk33 = Akk_diag + o_c3[:, None] * (H*BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_tc0[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, mask=m_tc1[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, mask=m_tc2[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, mask=m_tc3[:, None] & (o_i[None, :] < BC), other=0.0).to(tl.float32)

    ################################################################################
    # 4. forward substitution on diagonals
    ################################################################################
    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    if not USE_SAFE_GATE:
        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akk_diag + (i_tc0 + i) * H*BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2*BC, T - i_tc0)):
            b_a11 = -tl.load(Akk_diag + (i_tc0 + i) * H*BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
            b_a22 = -tl.load(Akk_diag + (i_tc0 + i) * H*BC + o_i)
            b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
        for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
            b_a33 = -tl.load(Akk_diag + (i_tc0 + i) * H*BC + o_i)
            b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # 5. compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    ################################################################################
    # 6. store full Akk_inv to Akk
    ################################################################################
    m_col = o_i[None, :] < BC
    p_Akk00 = Akk + o_c0[:, None] * (H*BT) + o_i[None, :]
    p_Akk10 = Akk + o_c1[:, None] * (H*BT) + o_i[None, :]
    p_Akk11 = Akk + o_c1[:, None] * (H*BT) + (o_i + BC)[None, :]
    p_Akk20 = Akk + o_c2[:, None] * (H*BT) + o_i[None, :]
    p_Akk21 = Akk + o_c2[:, None] * (H*BT) + (o_i + BC)[None, :]
    p_Akk22 = Akk + o_c2[:, None] * (H*BT) + (o_i + 2*BC)[None, :]
    p_Akk30 = Akk + o_c3[:, None] * (H*BT) + o_i[None, :]
    p_Akk31 = Akk + o_c3[:, None] * (H*BT) + (o_i + BC)[None, :]
    p_Akk32 = Akk + o_c3[:, None] * (H*BT) + (o_i + 2*BC)[None, :]
    p_Akk33 = Akk + o_c3[:, None] * (H*BT) + (o_i + 3*BC)[None, :]

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), mask=m_tc0[:, None] & m_col)
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), mask=m_tc1[:, None] & m_col)
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), mask=m_tc1[:, None] & m_col)
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), mask=m_tc2[:, None] & m_col)
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), mask=m_tc2[:, None] & m_col)
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), mask=m_tc2[:, None] & m_col)
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & m_col)
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & m_col)
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & m_col)
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), mask=m_tc3[:, None] & m_col)


def chunk_precond_kda_fwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    solve_tril_precision: str | None = None,
    safe_gate: bool = False,
):
    """
    Forward pass for preconditioned KDA intra-chunk.

    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - original keys (for Akk row side, WY w computation)
        k_precond: [B, T, H, K] - preconditioned keys (for column side, kg output)
        v: [B, T, H, V] - values
        gk: [B, T, H, K] - cumsum of gates
        beta: [B, T, H] - beta scaling
        scale: attention scale

    Returns:
        w: [B, T, H, K] - WY w vector (uses original k)
        u: [B, T, H, V] - WY u vector
        kg: [B, T, H, K] - gated k_precond for hidden state update
        Aqk: [B, T, H, BT] - q @ k_precond^T attention matrix
        Akk: [B, T, H, BT] - k @ k_precond^T (asymmetric) for WY
    """
    if solve_tril_precision is None:
        solve_tril_precision = DEFAULT_SOLVE_TRIL_PRECISION
    if solve_tril_precision in ('tf32', 'tf32x3'):
        solve_tril_precision = 'hf32'
    # float(scale): a unit int constant miscompiles under the CANN-bundled hivmc (#1298).
    scale = float(scale)

    B, T, H, K = k.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = 16
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akk_diag = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Step 1: Compute diagonal blocks into Akk_diag (fp32)
    if safe_gate:
        grid = (NT, NC, B * H)
        BK = triton.next_power_of_2(K)
        chunk_precond_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            k_precond=k_precond,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk_diag,
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
        Aqk, Akk_diag = chunk_precond_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            k_precond=k_precond,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk_diag,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril
    grid = (NT, B * H)
    chunk_precond_kda_fwd_kernel_inter_solve_fused_npu[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk_diag=Akk_diag,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        SOLVE_TRIL_DOT_PRECISION=solve_tril_precision,
        USE_SAFE_GATE=safe_gate,
        num_warps=_NUM_WARPS,
    )

    # Step 3: WY representation
    # w uses original k (for read/correction), kg uses k_precond (for write/h update)
    w, u, _, kg = recompute_w_u_fwd(
        k=k,           # Original k for w
        k_precond=k_precond,  # k_precond for kg
        v=v,
        beta=beta,
        A=Akk,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, kg, Aqk, Akk


def chunk_precond_kda_bwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    k_precond: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dk_precond: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    """Asymmetric intra backward for preconditioned KDA, in torch.

    The mainline triton kernel miscompiles under the CANN-bundled hivmc,
    so this stage mirrors its math sub-chunk by sub-chunk. The forward
    matrices Aqk/Akk are lower-triangular within each chunk:
        Aqk[t, s] = (q[t] @ k_precond[s]) * exp2(g[t] - g[s]) * beta[s]
        Akk[t, s] = (k[t] @ k_precond[s]) * exp2(g[t] - g[s]) * beta[s]
    """
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    dev = k.device
    CLAMP = 88.0  # exp2(88) is finite; decay differences are <= 0 on valid lanes

    varlen = cu_seqlens is not None
    if varlen:
        # packed layout: batch is 1, sequences are packed along T
        bos = cu_seqlens[:-1].to(torch.long)
        eos = cu_seqlens[1:].to(torch.long)
        nseq = len(bos)
        # squeeze the batch dim: all ops below index tokens globally
        def sq(t): return t.squeeze(0)
    else:
        bos = torch.zeros(B, dtype=torch.long, device=dev)
        eos = torch.full((B,), T, dtype=torch.long, device=dev)
        nseq = B
        def sq(t): return t
    lens = eos - bos
    n_chunks = max(int(triton.cdiv(int(lens.max().item()), BT)), 0)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    dk_precond2 = torch.empty_like(k_precond)
    dg2 = torch.empty_like(dg, dtype=torch.float32)
    # NK slices collapse in the caller via sum(0); keep a single slice.
    db2 = torch.zeros(1, B, T, H, dtype=torch.float32, device=dev)

    qq, kk, kpk, gg, bbe = sq(q), sq(k), sq(k_precond), sq(g), sq(beta)
    dAq, dAk = sq(dAqk), sq(dAkk)
    dqq, dkk, dkpq, dgg = sq(dq), sq(dk), sq(dk_precond), sq(dg)

    o = torch.arange(BC, device=dev)

    def clamp(x):
        return torch.clamp(x, max=CLAMP)

    for c0 in range(n_chunks):
        for i_i in range(NC):
            base = c0 * BT + i_i * BC          # sub-chunk start, sequence-local
            ridx = bos[:, None] + base + o[None, :]          # [B, BC] global tokens
            rvalid = (base + o[None, :]) < lens[:, None]     # [B, BC]
            if not rvalid.any():
                continue
            ridx_s = torch.minimum(ridx, eos[:, None] - 1)

            g_rows = gg[ridx_s].transpose(1, 2) if varlen else gg[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            q_rows = qq[ridx_s].transpose(1, 2) if varlen else qq[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            k_rows = kk[ridx_s].transpose(1, 2) if varlen else kk[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            kp_rows = kpk[ridx_s].transpose(1, 2) if varlen else kpk[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            beta_rows = bbe[ridx_s].transpose(1, 2) if varlen else bbe[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)          # [B,H,BC]
            dAqk_rows = dAq[ridx_s].transpose(1, 2) if varlen else dAq[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            dAkk_rows = dAk[ridx_s].transpose(1, 2) if varlen else dAk[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            dq_in = dqq[ridx_s].transpose(1, 2) if varlen else dqq[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            dk_in = dkk[ridx_s].transpose(1, 2) if varlen else dkk[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            dkp_in = dkpq[ridx_s].transpose(1, 2) if varlen else dkpq[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)
            dg_in = dgg[ridx_s].transpose(1, 2) if varlen else dgg[torch.arange(
                B, device=dev)[:, None].expand(B, BC), ridx_s].transpose(1, 2)

            s_idx = torch.minimum(bos + base, eos - 1)           # [B]
            g_start = (gg[s_idx] if varlen else gg[torch.arange(B, device=dev), s_idx]).unsqueeze(2)  # [B,H,1,K]

            dq2_acc = torch.zeros(nseq, H, BC, K, dtype=torch.float32, device=dev)
            dk2_acc = torch.zeros_like(dq2_acc)

            # Part 1a: previous sub-chunks of the same chunk (i_j < i_i)
            for i_j in range(i_i):
                cb = c0 * BT + i_j * BC
                cidx = bos[:, None] + cb + o[None, :]
                cvalid = (cb + o[None, :]) < lens[:, None]
                cidx_s = torch.minimum(cidx, eos[:, None] - 1)
                kp_c = kpk[cidx_s].transpose(1, 2) if varlen else kpk[torch.arange(
                    B, device=dev)[:, None].expand(B, BC), cidx_s].transpose(1, 2)
                g_c = gg[cidx_s].transpose(1, 2) if varlen else gg[torch.arange(
                    B, device=dev)[:, None].expand(B, BC), cidx_s].transpose(1, 2)
                kp_exp = kp_c * torch.exp2(clamp(g_start - g_c)) * cvalid[:, None, :, None]
                dq2_acc += torch.matmul(dAqk_rows[:, :, :, i_j * BC:(i_j + 1) * BC], kp_exp)
                dk2_acc += torch.matmul(dAkk_rows[:, :, :, i_j * BC:(i_j + 1) * BC], kp_exp)

            dq2_acc = dq2_acc * torch.exp2(clamp(g_rows - g_start))
            dk2_acc = dk2_acc * torch.exp2(clamp(g_rows - g_start))

            if safe_gate:
                # Vectorized upper diagonal (dq2/dk2): column side uses k_precond
                pick = torch.minimum(torch.tensor(BC // 2, dtype=torch.long, device=dev), lens - base - 1)
                pick = torch.clamp(pick, min=0)
                gn_idx = torch.minimum(bos + base + pick, eos - 1)
                gn = (gg[gn_idx] if varlen else gg[torch.arange(B, device=dev), gn_idx]).unsqueeze(2)  # [B,H,1,K]

                m_i = (o[:, None] >= o[None, :]) & rvalid[:, :, None] & rvalid[:, None, :]  # [B,BC,BC]
                g_diag = (g_rows - gn) * rvalid[:, None, :, None]
                exp_b = torch.exp2(clamp(g_diag)) * rvalid[:, None, :, None]
                exp_n = torch.exp2(clamp(-g_diag)) * rvalid[:, None, :, None]
                kp_exp_diag = kp_rows * exp_n
                dq2_acc += torch.matmul(dAqk_rows[:, :, :, i_i * BC:(i_i + 1) * BC] * m_i[:, None], kp_exp_diag) * exp_b
                dk2_acc += torch.matmul(dAkk_rows[:, :, :, i_i * BC:(i_i + 1) * BC] * m_i[:, None], kp_exp_diag) * exp_b
            else:
                # Diagonal j-loop: rows o_i >= j against column j of this sub-chunk
                for j in range(BC):
                    col_ok = (base + j) < lens                    # [B]; kernel loops only over valid j
                    cj = bos + base + j
                    cj_s = torch.minimum(cj, eos - 1)
                    kp_j = kpk[cj_s] if varlen else kpk[torch.arange(B, device=dev), cj_s]  # [B,H,K]
                    g_j = gg[cj_s] if varlen else gg[torch.arange(B, device=dev), cj_s]
                    m_ij = (o[None, :] >= j) & rvalid & col_ok[:, None]
                    kpgj = kp_j[:, :, None, :] * torch.exp2(clamp(g_rows - g_j[:, :, None, :]))
                    dq2_acc += (dAqk_rows[:, :, :, i_i * BC + j][..., None] * kpgj) * m_ij[:, None, :, None]
                    dk2_acc += (dAkk_rows[:, :, :, i_i * BC + j][..., None] * kpgj) * m_ij[:, None, :, None]

            # db: sum over K of dk2 * k, accumulated BEFORE the beta scale
            b_db = torch.sum(dk2_acc * k_rows, dim=-1)                    # [B,H,BC]
            sel = rvalid
            flat = ridx[sel] if varlen else (torch.arange(B, device=dev)[:, None] * T + ridx)[sel]
            db2[0].view(B * T, H).scatter_(0, flat[:, None].expand(-1, H),
                                           b_db.permute(0, 2, 1)[sel].to(torch.float32))
            dk2_acc = dk2_acc * beta_rows[..., None]

            dg2_acc = q_rows * dq2_acc

            # Part 2: dk_precond (lower triangular). Later sub-chunks first.
            dkt = torch.zeros_like(dq2_acc)
            for i_j in range(i_i + 1, NC):
                cb = c0 * BT + i_j * BC
                cidx = bos[:, None] + cb + o[None, :]
                cvalid = (cb + o[None, :]) < lens[:, None]
                cidx_s = torch.minimum(cidx, eos[:, None] - 1)
                q_c = qq[cidx_s].transpose(1, 2) if varlen else qq[torch.arange(
                    B, device=dev)[:, None].expand(B, BC), cidx_s].transpose(1, 2)
                k_c = kk[cidx_s].transpose(1, 2) if varlen else kk[torch.arange(
                    B, device=dev)[:, None].expand(B, BC), cidx_s].transpose(1, 2)
                g_c = gg[cidx_s].transpose(1, 2) if varlen else gg[torch.arange(
                    B, device=dev)[:, None].expand(B, BC), cidx_s].transpose(1, 2)
                beta_c = bbe[cidx_s].transpose(1, 2) if varlen else bbe[torch.arange(B, device=dev)[
                    :, None].expand(B, BC), cidx_s].transpose(1, 2)          # [B,H,BC]
                gn_t_local = torch.minimum(torch.tensor(c0 * BT + (i_i + 1) * BC, dtype=torch.long, device=dev), lens) - 1
                gn_t_idx = torch.minimum(bos + torch.clamp(gn_t_local, min=0), eos - 1)
                gn_t = (gg[gn_t_idx] if varlen else gg[torch.arange(B, device=dev), gn_t_idx]).unsqueeze(2)  # [B,H,1,K]
                gkn_t = torch.exp2(clamp(g_c - gn_t)) * cvalid[:, None, :, None]
                # kernel: b_dAqk_t[o_i, rj] = dAqk[token(o_rj), i_i*BC+o_i] - rows are the
                # later sub-chunk, columns this one, then transposed for the matmul
                dAqk_blk = (dAq[cidx_s].transpose(1, 2) if varlen else dAq[torch.arange(B, device=dev)[:, None].expand(
                    B, BC), cidx_s].transpose(1, 2))[:, :, :, i_i * BC:(i_i + 1) * BC].transpose(2, 3)
                dAkk_blk = (dAk[cidx_s].transpose(1, 2) if varlen else dAk[torch.arange(B, device=dev)[:, None].expand(
                    B, BC), cidx_s].transpose(1, 2))[:, :, :, i_i * BC:(i_i + 1) * BC].transpose(2, 3)
                dkt += torch.matmul(dAqk_blk, q_c * gkn_t)
                dkt += torch.matmul(dAkk_blk, k_c * gkn_t * beta_c[..., None])
            dkt = dkt * torch.exp2(clamp(gn_t - g_rows)) if i_i < NC - 1 else dkt

            if safe_gate:
                # Vectorized lower diagonal (dkt): row side uses q and k*beta
                gn_t2 = gn
                g_diag = (g_rows - gn_t2) * rvalid[:, None, :, None]
                exp_b = torch.exp2(clamp(g_diag)) * rvalid[:, None, :, None]
                exp_n = torch.exp2(clamp(-g_diag)) * rvalid[:, None, :, None]
                m_i = (o[:, None] <= o[None, :]) & rvalid[:, :, None] & rvalid[:, None, :]
                q_exp = q_rows * exp_b
                kb_exp = k_rows * beta_rows[..., None] * exp_b
                # diag_kk blocks index dA[col=o_i, row=o_ti] - transposed vs diag_qk
                dkt += torch.matmul(dAqk_rows[:, :, :, i_i * BC:(i_i + 1) * BC].transpose(2, 3) * m_i[:, None], q_exp) * exp_n
                dkt += torch.matmul(dAkk_rows[:, :, :, i_i * BC:(i_i + 1) * BC].transpose(2, 3) * m_i[:, None], kb_exp) * exp_n
            else:
                # Diagonal j-loop: rows o_i <= j against column j of this sub-chunk
                for j in range(BC):
                    col_ok = (base + j) < lens                    # [B]; kernel loops only over valid j
                    cj = bos + base + j
                    cj_s = torch.minimum(cj, eos - 1)
                    q_j = qq[cj_s] if varlen else qq[torch.arange(B, device=dev), cj_s]   # [B,H,K]
                    k_j = kk[cj_s] if varlen else kk[torch.arange(B, device=dev), cj_s]
                    g_j = gg[cj_s] if varlen else gg[torch.arange(B, device=dev), cj_s]
                    b_j = bbe[cj_s] if varlen else bbe[torch.arange(B, device=dev), cj_s]    # [B,H]
                    m_ij = (o[None, :] <= j) & rvalid & col_ok[:, None]
                    gkq = torch.exp2(clamp(g_j[:, :, None, :] - g_rows))
                    # kernel indexes dA at FIXED row i_ti+j with the column varying
                    # over o_i (i_i*BC + o_i), not the other way around
                    dAqk_c = (dAq[cj_s] if varlen else dAq[torch.arange(B, device=dev), cj_s])[:, :, i_i * BC:(i_i + 1) * BC]
                    dAkk_c = (dAk[cj_s] if varlen else dAk[torch.arange(B, device=dev), cj_s])[:, :, i_i * BC:(i_i + 1) * BC]
                    contrib = (dAkk_c[..., None] * k_j[:, :, None, :] * b_j[:, :, None, None]
                               + dAqk_c[..., None] * q_j[:, :, None, :]) * gkq
                    dkt += contrib * m_ij[:, None, :, None]

            # Outputs
            dq2.view(B * T, H * K).scatter_(0, flat[:, None].expand(-1, H * K),
                                            (dq2_acc + dq_in).permute(0, 2, 1, 3)[sel].reshape(-1, H * K).to(dq2.dtype))
            dk2.view(B * T, H * K).scatter_(0, flat[:, None].expand(-1, H * K),
                                            (dk2_acc + dk_in).permute(0, 2, 1, 3)[sel].reshape(-1, H * K).to(dk2.dtype))
            dk_precond2.view(B * T, H * K).scatter_(0, flat[:, None].expand(-1, H * K),
                                                    (dkt + dkp_in).permute(0, 2, 1, 3)[sel].reshape(-1, H * K).to(dk_precond2.dtype))
            dg2.view(B * T, H * K).scatter_(0, flat[:, None].expand(-1, H * K),
                                            (dg2_acc + dk2_acc * k_rows - dkt * kp_rows + dg_in).permute(0, 2, 1, 3)[sel].reshape(-1, H * K))

    db_out = db2.sum(0).add_(db)
    dg_out = chunk_local_cumsum(
        dg2,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return dq2, dk2, dk_precond2, db_out, dg_out
