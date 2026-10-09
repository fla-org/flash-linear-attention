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
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, max_grid_axis_chunks

_NUM_WARPS = 4


def _launch_solve_tril_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def solve_tril_16x16_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T
    i_t = tl.cast(i_t, tl.int64)

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    o_i_2 = i_t * 16 + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = offset + tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A = m_i_2[:, None] & m_i_3[None, :]
    p_A = A + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    b_A = tl.load(p_A, mask=m_p_A, other=0).to(tl.float32)
    b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I

    o_i_4 = tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < 16)
    m_p_Ai = m_i_2[:, None] & m_i_4[None, :]
    p_Ai = Ai + o_i_2[:, None] * (H * 16) + o_i_4[None, :]
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_Ai)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def merge_16x16_to_32x32_inverse_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_i_2 = tl.cast(i_t, tl.int64) * BT + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A_11 = m_i_2[:, None] & m_i_3[None, :]
    p_A_11 = A + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    o_i_4 = tl.cast(i_t, tl.int64) * BT + 16 + tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < T)
    o_i_5 = 16 + tl.arange(0, 16)
    m_i_5 = (o_i_5 >= 0) & (o_i_5 < BT)
    m_p_A_22 = m_i_4[:, None] & m_i_5[None, :]
    p_A_22 = A + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    b_Ai_11 = tl.load(p_A_11, mask=m_p_A_11, other=0).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, mask=m_p_A_22, other=0).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_11 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_22 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    m_p_A_21 = m_i_4[:, None] & m_i_3[None, :]
    p_A_21 = A + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    b_A_21 = tl.load(p_A_21, mask=m_p_A_21, other=0).to(tl.float32)
    b_Ai_22_c = b_Ai_22 + 0.0
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision='ieee'),
        b_Ai_11,
        input_precision='ieee',
    )

    p_Ai_11 = Ai + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_21 = Ai + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_22 = Ai + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_11)
    tl.store(p_Ai_22, b_Ai_22_c.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_22)
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_21)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def merge_16x16_to_64x64_inverse_kernel_npu(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_i_2 = tl.cast(i_t, tl.int64) * BT + tl.arange(0, 16)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < T)
    o_i_3 = tl.arange(0, 16)
    m_i_3 = (o_i_3 >= 0) & (o_i_3 < BT)
    m_p_A_11 = m_i_2[:, None] & m_i_3[None, :]
    p_A_11 = A + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    o_i_4 = tl.cast(i_t, tl.int64) * BT + 16 + tl.arange(0, 16)
    m_i_4 = (o_i_4 >= 0) & (o_i_4 < T)
    o_i_5 = 16 + tl.arange(0, 16)
    m_i_5 = (o_i_5 >= 0) & (o_i_5 < BT)
    m_p_A_22 = m_i_4[:, None] & m_i_5[None, :]
    p_A_22 = A + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    o_i_6 = tl.cast(i_t, tl.int64) * BT + 32 + tl.arange(0, 16)
    m_i_6 = (o_i_6 >= 0) & (o_i_6 < T)
    o_i_7 = 32 + tl.arange(0, 16)
    m_i_7 = (o_i_7 >= 0) & (o_i_7 < BT)
    m_p_A_33 = m_i_6[:, None] & m_i_7[None, :]
    p_A_33 = A + o_i_6[:, None] * (H * BT) + o_i_7[None, :]
    o_i_8 = tl.cast(i_t, tl.int64) * BT + 48 + tl.arange(0, 16)
    m_i_8 = (o_i_8 >= 0) & (o_i_8 < T)
    o_i_9 = 48 + tl.arange(0, 16)
    m_i_9 = (o_i_9 >= 0) & (o_i_9 < BT)
    m_p_A_44 = m_i_8[:, None] & m_i_9[None, :]
    p_A_44 = A + o_i_8[:, None] * (H * BT) + o_i_9[None, :]
    b_Ai_11 = tl.load(p_A_11, mask=m_p_A_11, other=0).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, mask=m_p_A_22, other=0).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, mask=m_p_A_33, other=0).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, mask=m_p_A_44, other=0).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_11 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_22 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_33 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - tl.cast(i_t, tl.int64) * BT)):
        b_a_44 = -tl.load(A + (tl.cast(i_t, tl.int64) * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    m_p_A_21 = m_i_4[:, None] & m_i_3[None, :]
    p_A_21 = A + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    m_p_A_31 = m_i_6[:, None] & m_i_3[None, :]
    p_A_31 = A + o_i_6[:, None] * (H * BT) + o_i_3[None, :]
    m_p_A_32 = m_i_6[:, None] & m_i_5[None, :]
    p_A_32 = A + o_i_6[:, None] * (H * BT) + o_i_5[None, :]
    m_p_A_41 = m_i_8[:, None] & m_i_3[None, :]
    p_A_41 = A + o_i_8[:, None] * (H * BT) + o_i_3[None, :]
    m_p_A_42 = m_i_8[:, None] & m_i_5[None, :]
    p_A_42 = A + o_i_8[:, None] * (H * BT) + o_i_5[None, :]
    m_p_A_43 = m_i_8[:, None] & m_i_7[None, :]
    p_A_43 = A + o_i_8[:, None] * (H * BT) + o_i_7[None, :]
    b_A_21 = tl.load(p_A_21, mask=m_p_A_21, other=0).to(tl.float32)
    b_A_31 = tl.load(p_A_31, mask=m_p_A_31, other=0).to(tl.float32)
    b_A_32 = tl.load(p_A_32, mask=m_p_A_32, other=0).to(tl.float32)
    b_A_41 = tl.load(p_A_41, mask=m_p_A_41, other=0).to(tl.float32)
    b_A_42 = tl.load(p_A_42, mask=m_p_A_42, other=0).to(tl.float32)
    b_A_43 = tl.load(p_A_43, mask=m_p_A_43, other=0).to(tl.float32)

    b_Ai_22_c = b_Ai_22 + 0.0
    b_Ai_33_c = b_Ai_33 + 0.0
    b_Ai_33_c2 = b_Ai_33 + 0.0
    b_Ai_44_c = b_Ai_44 + 0.0
    b_Ai_44_c2 = b_Ai_44 + 0.0
    b_Ai_44_store = b_Ai_44 + 0.0
    b_A_42_c = b_A_42 + 0.0
    b_A_43_c = b_A_43 + 0.0

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision='ieee'),
        b_Ai_11,
        input_precision='ieee',
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision='ieee'),
        b_Ai_22_c,
        input_precision='ieee',
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision='ieee'),
        b_Ai_33_c,
        input_precision='ieee',
    )
    b_Ai_31 = -tl.dot(
        b_Ai_33_c2,
        tl.dot(b_A_31, b_Ai_11, input_precision='ieee')
        + tl.dot(b_A_32, b_Ai_21, input_precision='ieee'),
        input_precision='ieee',
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44_c,
        tl.dot(b_A_42, b_Ai_22_c, input_precision='ieee')
        + tl.dot(b_A_43, b_Ai_32, input_precision='ieee'),
        input_precision='ieee',
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44_c2,
        tl.dot(b_A_41, b_Ai_11, input_precision='ieee')
        + tl.dot(b_A_42_c, b_Ai_21, input_precision='ieee')
        + tl.dot(b_A_43_c, b_Ai_31, input_precision='ieee'),
        input_precision='ieee',
    )

    p_Ai_11 = Ai + o_i_2[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_22 = Ai + o_i_4[:, None] * (H * BT) + o_i_5[None, :]
    p_Ai_33 = Ai + o_i_6[:, None] * (H * BT) + o_i_7[None, :]
    p_Ai_44 = Ai + o_i_8[:, None] * (H * BT) + o_i_9[None, :]
    p_Ai_21 = Ai + o_i_4[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_31 = Ai + o_i_6[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_32 = Ai + o_i_6[:, None] * (H * BT) + o_i_5[None, :]
    p_Ai_41 = Ai + o_i_8[:, None] * (H * BT) + o_i_3[None, :]
    p_Ai_42 = Ai + o_i_8[:, None] * (H * BT) + o_i_5[None, :]
    p_Ai_43 = Ai + o_i_8[:, None] * (H * BT) + o_i_7[None, :]
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_11)
    tl.store(p_Ai_22, b_Ai_22_c.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_22)
    tl.store(p_Ai_33, b_Ai_33_c.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_33)
    tl.store(p_Ai_44, b_Ai_44_store.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_44)
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_21)
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_31)
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_32)
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_41)
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_42)
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_p_A_43)


@input_guard
def solve_tril_npu(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
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
    else:
        merge_fn = merge_16x16_to_64x64_inverse_kernel_npu

    _launch_solve_tril_kernel(
        merge_fn,
        NT=NT,
        bh_total=B * H,
        kernel_kwargs={
            'A': A,
            'Ai': Ai,
            'cu_seqlens': cu_seqlens,
            'chunk_indices': chunk_indices,
            'T': T,
            'H': H,
            'BT': BT,
            'IS_VARLEN': cu_seqlens is not None,
            'NT_OFFSET': 0,
            'BH_OFFSET': 0,
        },
    )
    return Ai
