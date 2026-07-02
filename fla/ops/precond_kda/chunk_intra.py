# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.precond_kda.chunk_intra_token_parallel import chunk_precond_kda_fwd_intra_token_parallel
from fla.ops.precond_kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.op import exp2, gather
from fla.utils import IS_GATHER_SUPPORTED, IS_TF32_SUPPORTED, autotune_cache_kwargs

DEFAULT_SOLVE_TRIL_PRECISION = 'tf32x3' if IS_TF32_SUPPORTED else 'ieee'


################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'BK': lambda args: min(triton.next_power_of_2(args['K']), 64),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_fwd_kernel_inter_solve_fused(
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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

    m_tc1 = (i_tc1 + tl.arange(0, BC)) < T
    m_tc2 = (i_tc2 + tl.arange(0, BC)) < T
    m_tc3 = (i_tc3 + tl.arange(0, BC)) < T

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
        p_kp0 = tl.make_block_ptr(k_precond, (K, T), (1, H*K), (i_k * BK, i_tc0), (BK, BC), (0, 1))
        p_g0 = tl.make_block_ptr(g, (K, T), (1, H*K), (i_k * BK, i_tc0), (BK, BC), (0, 1))
        b_kpt0 = tl.load(p_kp0, boundary_check=(0, 1)).to(tl.float32)  # k_precond transposed
        b_gt0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        b_kpt1, b_gt1 = b_kpt0, b_gt0
        b_kpt2, b_gt2 = b_kpt0, b_gt0

        if i_tc1 < T:
            # Row side: q for Aqk, original k for Akk
            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_kp1 = tl.make_block_ptr(k_precond, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)  # Original k for row
            b_kp1 = tl.load(p_kp1, boundary_check=(0, 1)).to(tl.float32)  # k_precond for column
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK, BC]
            b_kpt1 = tl.trans(b_kp1)  # k_precond transposed
            b_gt1 = tl.trans(b_g1)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn1 = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_qg1 = b_q1 * b_gqn1
            b_kg1 = b_k1 * b_gqn1  # Original k for Akk row
            # [BK, BC]
            b_kpgt = b_kpt0 * exp2(b_gn1[:, None] - b_gt0)  # k_precond for column
            # [BC, BC]
            b_Aqk10 += tl.dot(b_qg1, b_kpgt)
            b_Akk10 += tl.dot(b_kg1, b_kpgt)  # Asymmetric: k @ k_precond^T

        if i_tc2 < T:
            p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_kp2 = tl.make_block_ptr(k_precond, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))

            b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
            b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_kp2 = tl.load(p_kp2, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_kpt2 = tl.trans(b_kp2)
            b_gt2 = tl.trans(b_g2)

            b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            b_kpgt = b_kpt0 * exp2(b_gn2[:, None] - b_gt0)
            b_Aqk20 += tl.dot(b_qg2, b_kpgt)
            b_Akk20 += tl.dot(b_kg2, b_kpgt)

            b_kpgt = b_kpt1 * exp2(b_gn2[:, None] - b_gt1)
            b_Aqk21 += tl.dot(b_qg2, b_kpgt)
            b_Akk21 += tl.dot(b_kg2, b_kpgt)

        if i_tc3 < T:
            p_q3 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_k3 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_g3 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
            b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
            b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)

            b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
            b_qg3 = b_q3 * b_gqn3
            b_kg3 = b_k3 * b_gqn3
            b_kpgt = b_kpt0 * exp2(b_gn3[:, None] - b_gt0)
            b_Aqk30 += tl.dot(b_qg3, b_kpgt)
            b_Akk30 += tl.dot(b_kg3, b_kpgt)

            b_kpgt = b_kpt1 * exp2(b_gn3[:, None] - b_gt1)
            b_Aqk31 += tl.dot(b_qg3, b_kpgt)
            b_Akk31 += tl.dot(b_kg3, b_kpgt)

            b_kpgt = b_kpt2 * exp2(b_gn3[:, None] - b_gt2)
            b_Aqk32 += tl.dot(b_qg3, b_kpgt)
            b_Akk32 += tl.dot(b_kg3, b_kpgt)

    ################################################################################
    # 2. save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    ################################################################################
    # 3. load diagonal Akk blocks
    ################################################################################
    p_Akk00 = tl.make_block_ptr(Akk_diag, (T, BC), (H*BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk_diag, (T, BC), (H*BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk_diag, (T, BC), (H*BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk_diag, (T, BC), (H*BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

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
    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_precond_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,           # Original k (for Akk row side)
    k_precond,   # Preconditioned k (for column side)
    g,
    beta,
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
    """
    Asymmetric sub_chunk kernel for preconditioned KDA.
    Computes diagonal Aqk and Akk blocks using block-level dot products.
    Key difference from symmetric KDA: column side uses k_precond.
    """
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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
    k_precond = k_precond + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    beta = beta + bos * H + i_h
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_kp = tl.make_block_ptr(k_precond, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kp = tl.load(p_kp, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        # calculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    # Asymmetric: column side uses k_precond
    b_kpgt = tl.trans(b_kp * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kpgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kpgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H*BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H*BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def chunk_precond_kda_fwd_intra(
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
    chunk_precond_kda_fwd_kernel_inter_solve_fused[grid](
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


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_precond_kda_bwd_kernel_intra(
    q,              # [B, T, H, K] - queries
    k,              # [B, T, H, K] - original k (for Akk row side)
    k_precond,      # [B, T, H, K] - preconditioned k (for column side)
    g,              # [B, T, H, K] - gate cumsum
    beta,           # [B, T, H]
    dAqk,           # [B, T, H, BT] - gradient of Aqk
    dAkk,           # [B, T, H, BT] - gradient of Akk (asymmetric)
    dq,             # [B, T, H, K] - input: accumulated dq
    dq2,            # [B, T, H, K] - output: updated dq
    dk,             # [B, T, H, K] - input: accumulated dk (original k)
    dk2,            # [B, T, H, K] - output: updated dk
    dk_precond,     # [B, T, H, K] - input: accumulated dk_precond
    dk_precond2,    # [B, T, H, K] - output: updated dk_precond
    dg,             # [B, T, H, K] - input: accumulated dg
    dg2,            # [B, T, H, K] - output: updated dg
    db,             # [NK, B, T, H] - output: accumulated dbeta
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
    SAFE_GATE: tl.constexpr = False,
    USE_GATHER: tl.constexpr = False,
):
    """
    Asymmetric intra backward for preconditioned KDA.

    Key differences from symmetric KDA:
    - Row side of Akk uses original k
    - Column side uses k_precond
    - dAkk flows to both dk (row) and dk_precond (column)
    - dAqk flows to dq (row) and dk_precond (column)

    Forward:
        Aqk[t, s] = q[t] @ k_precond[s]^T * exp(g[t] - g[s]) * beta[s]  (for t >= s)
        Akk[t, s] = k[t] @ k_precond[s]^T * exp(g[t] - g[s]) * beta[s]  (for t > s)
    """
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k = i_kc // NC
    i_i = i_kc % NC

    all_BT = B * T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_i = tl.arange(0, BC)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    k_precond += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dk_precond += (bos * H + i_h) * K
    dk_precond2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all_BT + bos) * H + i_h  # Like KDA line 416

    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    p_gn_start = g + i_ti * H*K + o_k
    b_gn_start = tl.load(p_gn_start, mask=m_k, other=0).to(tl.float32)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

    for i_j in range(0, i_i):
        p_kp = tl.make_block_ptr(k_precond, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_kp = tl.load(p_kp, boundary_check=(0, 1)).to(tl.float32)
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kpg = b_kp * exp2(b_gn_start[None, :] - b_gk)
        # [BC, BC]
        b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)
        # [BC, BK]
        b_dq2 += tl.dot(b_dAqk, b_kpg)
        b_dk2 += tl.dot(b_dAkk, b_kpg)

    b_gqn = exp2(b_g - b_gn_start[None, :])
    b_dq2 *= b_gqn
    b_dk2 *= b_gqn

    o_dA = (i_ti + o_i) * H*BT + i_i * BC
    p_kpj = k_precond + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k
    m_dA = (i_ti + o_i) < T

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)

    if SAFE_GATE:
        # Vectorized upper diagonal (dq2/dk2): column side uses k_precond
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_kp_diag = tl.make_block_ptr(k_precond, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_kp_diag = tl.load(p_kp_diag, boundary_check=(0, 1)).to(tl.float32)

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        # Asymmetric: column side uses k_precond
        b_kp_exp_diag_qk = b_kp_diag * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_kp_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_kp_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk_val = tl.load(dAqk + o_dA + j, mask=m_dA, other=0).to(tl.float32)
            b_dAkk_val = tl.load(dAkk + o_dA + j, mask=m_dA, other=0).to(tl.float32)

            b_kpj = tl.load(p_kpj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)

            m_ij = o_i[:, None] >= j
            b_kpgj = b_kpj[None, :] * exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_ij, b_dAqk_val[:, None] * b_kpgj, 0.)
            b_dk2 += tl.where(m_ij, b_dAkk_val[:, None] * b_kpgj, 0.)

            p_kpj += H*K
            p_gkj += H*K

    b_db = tl.sum(b_dk2 * b_k, 1)
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_ti,), (BC,), (0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
    b_dk2 *= b_b[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)  # Final dk_precond (with beta from rows)

    NC_actual = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC_actual - 1:
        p_gn_t = g + (min(i_ti + BC, T) - 1) * H*K + o_k
        b_gn_t = tl.load(p_gn_t, mask=m_k, other=0).to(tl.float32)

        for i_j in range(i_i + 1, NC_actual):
            p_q_t = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k_t = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_gk_t = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_b_row = tl.make_block_ptr(beta, (T,), (H,), (i_t*BT+i_j*BC,), (BC,), (0,))  # beta at ROW positions
            p_dAqk_t = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i*BC, i_t*BT+i_j*BC), (BC, BC), (0, 1))
            p_dAkk_t = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i*BC, i_t*BT+i_j*BC), (BC, BC), (0, 1))

            # [BC]
            b_b_row = tl.load(p_b_row, boundary_check=(0,)).to(tl.float32)  # beta from ROW positions
            # [BC, BK]
            b_q_t = tl.load(p_q_t, boundary_check=(0, 1)).to(tl.float32)
            b_k_t = tl.load(p_k_t, boundary_check=(0, 1)).to(tl.float32)
            b_gk_t = tl.load(p_gk_t, boundary_check=(0, 1)).to(tl.float32)
            # [BC, BC]
            b_dAqk_t = tl.load(p_dAqk_t, boundary_check=(0, 1)).to(tl.float32)
            b_dAkk_t = tl.load(p_dAkk_t, boundary_check=(0, 1)).to(tl.float32)

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn_t = tl.where(m_j[:, None], exp2(b_gk_t - b_gn_t[None, :]), 0)
            b_qg_t = b_q_t * b_gkn_t
            b_kg_t = b_k_t * b_gkn_t * b_b_row[:, None]  # beta from ROW positions
            # [BC, BK]
            b_dkt += tl.dot(b_dAqk_t, b_qg_t)
            b_dkt += tl.dot(b_dAkk_t, b_kg_t)

        b_dkt *= exp2(b_gn_t[None, :] - b_g)

    if SAFE_GATE:
        # Vectorized lower diagonal (dkt): row side uses q and k*beta
        if USE_GATHER:
            b_gn_t2 = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn_t2 = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + o_k
            b_gn_t2 = tl.load(p_gn_t2, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q_diag = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_q_diag = tl.load(p_q_diag, boundary_check=(0, 1)).to(tl.float32)
        p_b_diag = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))
        b_b_diag = tl.load(p_b_diag, boundary_check=(0,)).to(tl.float32)

        p_dAqk_diag_kk = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk_diag_kk = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_diag_kk = tl.load(p_dAqk_diag_kk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk_diag_kk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        # ensure numerical stability
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn_t2, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        # Row-side values: q (for Aqk) and k*beta (for Akk)
        b_q_exp = b_q_diag * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b_diag[:, None] * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag_kk
    else:
        o_dA_t = i_ti * H*BT + i_i * BC + o_i
        p_qj = q + i_ti * H*K + o_k
        p_kj = k + i_ti * H*K + o_k
        p_gkj_t = g + i_ti * H*K + o_k
        p_bj = beta + i_ti * H

        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk_t = tl.load(dAqk + o_dA_t + j * H*BT).to(tl.float32)
            b_dAkk_t = tl.load(dAkk + o_dA_t + j * H*BT).to(tl.float32)
            # [BK]
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj_t = tl.load(p_gkj_t, mask=m_k, other=0).to(tl.float32)
            b_bj = tl.load(p_bj).to(tl.float32)  # beta at row position j
            # [BC, BK]
            m_ij = o_i[:, None] <= j
            b_gkq_t = exp2(b_gkj_t[None, :] - b_g)
            # k with beta from ROW position, q without beta (Aqk has no beta)
            b_dkt += tl.where(m_ij, (b_dAkk_t[:, None] * b_kj[None, :] * b_bj +
                              b_dAqk_t[:, None] * b_qj[None, :]) * b_gkq_t, 0.)

            p_qj += H*K
            p_kj += H*K
            p_gkj_t += H*K
            p_bj += H

    p_kp_local = tl.make_block_ptr(k_precond, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_kp_local = tl.load(p_kp_local, boundary_check=(0, 1)).to(tl.float32)

    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk_precond = tl.make_block_ptr(dk_precond, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk_precond2 = tl.make_block_ptr(dk_precond2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += b_dk2 * b_k - b_dkt * b_kp_local + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk_precond2 = b_dkt + tl.load(p_dk_precond, boundary_check=(0, 1))

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk_precond2, b_dk_precond2.to(p_dk_precond2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


def chunk_precond_kda_bwd_intra(
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
    """
    Asymmetric intra backward for preconditioned KDA.
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
    dk_precond2 = torch.empty_like(k_precond)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)

    grid = (NK * NC, NT, B * H)
    chunk_precond_kda_bwd_kernel_intra[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dk_precond=dk_precond,
        dk_precond2=dk_precond2,
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
    dk_precond = dk_precond2
    db = db2.sum(0).add_(db)
    dg = chunk_local_cumsum(
        dg2,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    return dq, dk, dk_precond, db, dg
