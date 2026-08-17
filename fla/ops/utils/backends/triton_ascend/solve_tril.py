# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""solve_tril adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import input_guard


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos*H + i_h) * BT
    Ai = Ai + (bos*H + i_h) * 16

    offset = (i_t * 16) % BT
    o_t = (i_t * 16 + o_i).to(tl.int64)
    m_t = o_t < T
    p_A = A + o_t[:, None] * (H*BT) + (o_i + offset)[None, :]
    # [16, 16]
    b_A = tl.load(p_A, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        # [16]
        b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    p_Ai = Ai + o_t[:, None] * (H*16) + o_i[None, :]
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_t[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_t = (i_t * BT + o_i).to(tl.int64)
    p_A_11 = A + o_t[:, None] * (H*BT) + o_i[None, :]
    p_A_22 = A + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
    b_Ai_11 = tl.load(p_A_11, mask=(o_t[:, None] < T), other=0.0).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    p_A_21 = A + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
    b_A_21 = tl.load(p_A_21, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)

    p_Ai_11 = Ai + o_t[:, None] * (H*BT) + o_i[None, :]
    p_Ai_21 = Ai + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
    p_Ai_22 = Ai + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] < T))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    o_j = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    r_off_11 = i_t * BT
    r_off_22 = i_t * BT + 16
    r_off_33 = i_t * BT + 32
    r_off_44 = i_t * BT + 48
    m_diag = (r_off_11 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_diag_22 = (r_off_22 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_diag_33 = (r_off_33 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    m_diag_44 = (r_off_44 + o_i[:, None] < T) & (48 + o_j[None, :] < BT)
    b_Ai_11 = tl.load(A + (r_off_11 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_diag, other=0.0).to(tl.float32)
    b_Ai_22 = tl.load(A + (r_off_22 + o_i[:, None]) * H * BT + 16 + o_j[None, :], mask=m_diag_22, other=0.0).to(tl.float32)
    b_Ai_33 = tl.load(A + (r_off_33 + o_i[:, None]) * H * BT + 32 + o_j[None, :], mask=m_diag_33, other=0.0).to(tl.float32)
    b_Ai_44 = tl.load(A + (r_off_44 + o_i[:, None]) * H * BT + 48 + o_j[None, :], mask=m_diag_44, other=0.0).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    # tri_inv_mch for diagonal block 11
    # X = I - A = I + b_Ai_11,  Y = A@A = b_Ai_11 @ b_Ai_11
    # 3 iterations: X += X@Y; Y = Y@Y

    remaining_rows = T - i_t * BT
    f_I = m_I.to(tl.float32)
    if (not IS_VARLEN) or remaining_rows >= 16:
        X_11 = f_I + b_Ai_11
        Y_11 = tl.dot(b_Ai_11, b_Ai_11, input_precision=DOT_PRECISION)
        X_11 = X_11 + tl.dot(X_11, Y_11, input_precision=DOT_PRECISION)
        Y_11 = tl.dot(Y_11, Y_11, input_precision=DOT_PRECISION)
        X_11 = X_11 + tl.dot(X_11, Y_11, input_precision=DOT_PRECISION)
        Y_11 = tl.dot(Y_11, Y_11, input_precision=DOT_PRECISION)
        X_11 = X_11 + tl.dot(X_11, Y_11, input_precision=DOT_PRECISION)
        b_Ai_11 = X_11
    else:
        for i in range(2, min(16, T - i_t * BT)):
            b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
            b_a_11 = tl.where(o_i < i, b_a_11, 0.)
            b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
            b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
        b_Ai_11 += m_I

    if (not IS_VARLEN) or remaining_rows >= 32:
        # tri_inv_mch for diagonal block 22
        X_22 = f_I + b_Ai_22
        Y_22 = tl.dot(b_Ai_22, b_Ai_22, input_precision=DOT_PRECISION)
        X_22 = X_22 + tl.dot(X_22, Y_22, input_precision=DOT_PRECISION)
        Y_22 = tl.dot(Y_22, Y_22, input_precision=DOT_PRECISION)
        X_22 = X_22 + tl.dot(X_22, Y_22, input_precision=DOT_PRECISION)
        Y_22 = tl.dot(Y_22, Y_22, input_precision=DOT_PRECISION)
        X_22 = X_22 + tl.dot(X_22, Y_22, input_precision=DOT_PRECISION)
        b_Ai_22 = X_22
    else:
        for i in range(16 + 2, min(32, T - i_t * BT)):
            b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
            b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
            b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
            b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
        b_Ai_22 += m_I

    if (not IS_VARLEN) or remaining_rows >= 48:
        # tri_inv_mch for diagonal block 33
        X_33 = f_I + b_Ai_33
        Y_33 = tl.dot(b_Ai_33, b_Ai_33, input_precision=DOT_PRECISION)
        X_33 = X_33 + tl.dot(X_33, Y_33, input_precision=DOT_PRECISION)
        Y_33 = tl.dot(Y_33, Y_33, input_precision=DOT_PRECISION)
        X_33 = X_33 + tl.dot(X_33, Y_33, input_precision=DOT_PRECISION)
        Y_33 = tl.dot(Y_33, Y_33, input_precision=DOT_PRECISION)
        X_33 = X_33 + tl.dot(X_33, Y_33, input_precision=DOT_PRECISION)
        b_Ai_33 = X_33
    else:
        for i in range(32 + 2, min(48, T - i_t * BT)):
            b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
            b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
            b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
            b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
        b_Ai_33 += m_I

    if (not IS_VARLEN) or remaining_rows >= 64:
        # tri_inv_mch for diagonal block 44
        X_44 = f_I + b_Ai_44
        Y_44 = tl.dot(b_Ai_44, b_Ai_44, input_precision=DOT_PRECISION)
        X_44 = X_44 + tl.dot(X_44, Y_44, input_precision=DOT_PRECISION)
        Y_44 = tl.dot(Y_44, Y_44, input_precision=DOT_PRECISION)
        X_44 = X_44 + tl.dot(X_44, Y_44, input_precision=DOT_PRECISION)
        Y_44 = tl.dot(Y_44, Y_44, input_precision=DOT_PRECISION)
        X_44 = X_44 + tl.dot(X_44, Y_44, input_precision=DOT_PRECISION)
        b_Ai_44 = X_44
    else:
        for i in range(48 + 2, min(64, T - i_t * BT)):
            b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
            b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
            b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
            b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
        b_Ai_44 += m_I


    m_off_21 = (i_t * BT + 16 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_off_31 = (i_t * BT + 32 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_off_32 = (i_t * BT + 32 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_off_41 = (i_t * BT + 48 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_off_42 = (i_t * BT + 48 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_off_43 = (i_t * BT + 48 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    b_A_21 = tl.load(A + (i_t * BT + 16 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_off_21, other=0.0).to(tl.float32)
    b_A_31 = tl.load(A + (i_t * BT + 32 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_off_31, other=0.0).to(tl.float32)
    b_A_32 = tl.load(A + (i_t * BT + 32 + o_i[:, None]) * H * BT + 16 + o_j[None, :], mask=m_off_32, other=0.0).to(tl.float32)
    b_A_41 = tl.load(A + (i_t * BT + 48 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_off_41, other=0.0).to(tl.float32)
    b_A_42 = tl.load(A + (i_t * BT + 48 + o_i[:, None]) * H * BT + 16 + o_j[None, :], mask=m_off_42, other=0.0).to(tl.float32)
    b_A_43 = tl.load(A + (i_t * BT + 48 + o_i[:, None]) * H * BT + 32 + o_j[None, :], mask=m_off_43, other=0.0).to(tl.float32)

    # MBH off-diagonal: recursive 2x2 block inversion
    # Level 2: 16x16 -> 32x32 pairs — X21 = -X22@A21@X11, X43 = -X44@A43@X33
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    # Level 1: 32x32 -> 64x64 cross terms — intermediates B21 @ X_UL
    P00 = tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION)
    P01 = tl.dot(b_A_32, b_Ai_22, input_precision=DOT_PRECISION)
    P10 = tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
    P11 = tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(b_Ai_33, P00, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(b_Ai_33, P01, input_precision=DOT_PRECISION)
    b_Ai_41 = -tl.dot(b_Ai_43, P00, input_precision=DOT_PRECISION) - tl.dot(b_Ai_44, P10, input_precision=DOT_PRECISION)
    b_Ai_42 = -tl.dot(b_Ai_43, P01, input_precision=DOT_PRECISION) - tl.dot(b_Ai_44, P11, input_precision=DOT_PRECISION)

    Ai_dt = Ai.dtype.element_ty
    m_s11 = (i_t * BT + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s22 = (i_t * BT + 16 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_s33 = (i_t * BT + 32 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    m_s44 = (i_t * BT + 48 + o_i[:, None] < T) & (48 + o_j[None, :] < BT)
    m_s21 = (i_t * BT + 16 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s31 = (i_t * BT + 32 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s32 = (i_t * BT + 32 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_s41 = (i_t * BT + 48 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s42 = (i_t * BT + 48 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_s43 = (i_t * BT + 48 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    tl.store(Ai + (i_t * BT + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_11.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s11)
    tl.store(Ai + (i_t * BT + 16 + o_i[:, None]) * H * BT + 16 + o_j[None, :], b_Ai_22.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s22)
    tl.store(Ai + (i_t * BT + 32 + o_i[:, None]) * H * BT + 32 + o_j[None, :], b_Ai_33.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s33)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + 48 + o_j[None, :], b_Ai_44.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s44)
    tl.store(Ai + (i_t * BT + 16 + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_21.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s21)
    tl.store(Ai + (i_t * BT + 32 + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_31.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s31)
    tl.store(Ai + (i_t * BT + 32 + o_i[:, None]) * H * BT + 16 + o_j[None, :], b_Ai_32.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s32)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_41.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s41)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + 16 + o_j[None, :], b_Ai_42.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s42)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + 32 + o_j[None, :], b_Ai_43.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s43)


@input_guard
def solve_tril_npu(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel_npu
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel_npu
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel_npu

    merge_fn[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=False,
        DOT_PRECISION="ieee",
    )
    return Ai
