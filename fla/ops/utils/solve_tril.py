# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.common.utils import prepare_chunk_indices
from fla.utils import input_guard


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,
    Ad,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
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
        A = A + i_bh * T * BT
        Ad = Ad + i_bh * T * 16
        stride_16 = 16
        stride_BT = BT
    else:
        A = A + (bos*H + i_h) * BT
        Ad = Ad + (bos*H + i_h) * 16
        stride_16 = H*16
        stride_BT = H*BT

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(A, (T, BT), (stride_BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)

    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T-i_t*16)):
        b_a = -tl.load(A + (i_t * 16 + i) * stride_BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'HEAD_FIRST', 'USE_OFFSETS'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
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
        A += (i_bh * T * 32)
        Ad += (i_bh * T * 16)
        Ai += (i_bh * T * 32)
        stride_16 = 16
        stride_32 = 32
    else:
        A += (bos*H + i_h) * 32
        Ad += (bos*H + i_h) * 16
        Ai += (bos*H + i_h) * 32
        stride_16 = 16 * H
        stride_32 = 32 * H

    p_A_21 = tl.make_block_ptr(A, (T, 32), (stride_32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_Ai_11 = tl.make_block_ptr(Ai, (T, 32), (stride_32, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, 32), (stride_32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, 32), (stride_32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1))
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1))
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1))
    Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision='ieee'), Ai_11, input_precision='ieee')
    tl.store(p_Ai_11, Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT', 'HEAD_FIRST', 'USE_OFFSETS'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
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
        A += i_bh * T * 64
        Ad += i_bh * T * 16
        Ai += i_bh * T * 64
        stride_16 = 16
        stride_64 = 64
    else:
        A += (bos*H + i_h) * 64
        Ad += (bos*H + i_h) * 16
        Ai += (bos*H + i_h) * 64
        stride_16 = 16 * H
        stride_64 = 64 * H

    p_A_21 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 64, 0), (16, 16), (1, 0))
    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_Ad_33 = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_Ad_44 = tl.make_block_ptr(Ad, (T, 16), (stride_16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1))
    A_32 = tl.load(p_A_32, boundary_check=(0, 1))
    A_31 = tl.load(p_A_31, boundary_check=(0, 1))
    A_43 = tl.load(p_A_43, boundary_check=(0, 1))
    A_42 = tl.load(p_A_42, boundary_check=(0, 1))
    A_41 = tl.load(p_A_41, boundary_check=(0, 1))

    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1))
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1))
    Ai_33 = tl.load(p_Ad_33, boundary_check=(0, 1))
    Ai_44 = tl.load(p_Ad_44, boundary_check=(0, 1))

    Ai_21 = -tl.dot(tl.dot(Ai_22, A_21, input_precision='ieee'), Ai_11, input_precision='ieee')
    Ai_32 = -tl.dot(tl.dot(Ai_33, A_32, input_precision='ieee'), Ai_22, input_precision='ieee')
    Ai_43 = -tl.dot(tl.dot(Ai_44, A_43, input_precision='ieee'), Ai_33, input_precision='ieee')

    Ai_31 = -tl.dot(
        Ai_33,
        tl.dot(A_31, Ai_11, input_precision='ieee') +
        tl.dot(A_32, Ai_21, input_precision='ieee'),
        input_precision='ieee'
    )
    Ai_42 = -tl.dot(
        Ai_44,
        tl.dot(A_42, Ai_22, input_precision='ieee') +
        tl.dot(A_43, Ai_32, input_precision='ieee'),
        input_precision='ieee'
    )
    Ai_41 = -tl.dot(
        Ai_44,
        tl.dot(A_41, Ai_11, input_precision='ieee') +
        tl.dot(A_42, Ai_21, input_precision='ieee') +
        tl.dot(A_43, Ai_31, input_precision='ieee'),
        input_precision='ieee'
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, 64), (stride_64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_33, Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_44, Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_31, Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_32, Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_41, Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_42, Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_43, Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Compute the inverse of the lower triangular matrix
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, K] if head_first else [B, H, T, K]
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None.
        head_first (bool):
            If False, the input/output tensor is in the shape of [B, T, H, K].
            If True, the input/output tensor is in the shape of [B, H, T, K].
            Default: False
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]
    assert A.dtype == torch.float, "A should be float32."

    if head_first:
        B, H, T, BT = A.shape
        Ad = torch.empty(B, H, T, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype)
    else:
        B, T, H, BT = A.shape
        Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype)

    indices = prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    NT = len(indices) if cu_seqlens is not None else triton.cdiv(T, 16)
    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        offsets=cu_seqlens,
        indices=indices,
        T=T,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
    )
    if BT == 16:
        return Ad

    if head_first:
        Ai = torch.zeros(B, H, T, BT, device=A.device, dtype=output_dtype)
    else:
        Ai = torch.zeros(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        offsets=cu_seqlens,
        indices=indices,
        T=T,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        USE_OFFSETS=cu_seqlens is not None
    )
    return Ai
