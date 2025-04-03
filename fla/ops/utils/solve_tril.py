

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.common.utils import prepare_chunk_indices


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,  # (batch, head, T, BT)
    A_inv_diag,  # (batch, head, T, 16)
    offsets,
    indices,
    T,
    BT: tl.constexpr,
    H: tl.constexpr,
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
        A_inv_diag = A_inv_diag + i_bh * T * 16
        stride_16 = 16
        stride_BT = BT
    else:
        A = A + (bos*H + i_h) * BT
        A_inv_diag = A_inv_diag + (bos*H + i_h) * 16
        stride_16 = H*16
        stride_BT = H*BT

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(A, (T, BT), (stride_BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
    p_A_inv = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)

    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T-i_t*16)):
        p_A_i = A + (i_t * 16 + i) * stride_BT + o_i + offset
        b_a = -tl.load(p_A_i)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(p_A_inv, b_A.to(p_A_inv.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2]
        for num_stages in [2, 3, 4]
    ],
    key=['H'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    A_inv_diag,
    A_inv,
    offsets,
    indices,
    T,
    H: tl.constexpr,
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
        A_inv_diag += (i_bh * T * 16)
        A += (i_bh * T * 32)
        A_inv += (i_bh * T * 32)
        stride_16 = 16
        stride_32 = 32
    else:
        A_inv_diag += (bos*H + i_h) * 16
        A += (bos*H + i_h) * 32
        A_inv += (bos*H + i_h) * 32
        stride_16 = 16 * H
        stride_32 = 32 * H

    p_A_inv_11 = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_A_inv_22 = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))

    p_A_21 = tl.make_block_ptr(A, (T, 32), (stride_32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_A_final_11 = tl.make_block_ptr(A_inv, (T, 32), (stride_32, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_A_final_22 = tl.make_block_ptr(A_inv, (T, 32), (stride_32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0))
    p_A_final_21 = tl.make_block_ptr(A_inv, (T, 32), (stride_32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))

    A_inv_11 = tl.load(p_A_inv_11, boundary_check=(0, 1))
    A_inv_22 = tl.load(p_A_inv_22, boundary_check=(0, 1))
    A_21 = tl.load(p_A_21, boundary_check=(0, 1))
    A_inv_21 = -tl.dot(tl.dot(A_inv_22, A_21, input_precision='ieee'), A_inv_11, input_precision='ieee')
    tl.store(p_A_final_11, A_inv_11.to(p_A_final_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_22, A_inv_22.to(p_A_final_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_21, A_inv_21.to(p_A_final_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'BK', 'BC', 'HEAD_FIRST', 'USE_OFFSETS'],
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    A_inv_diag,
    A_inv,
    offsets,
    indices,
    T,
    H: tl.constexpr,
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
        A_inv_diag += i_bh * T * 16
        A += i_bh * T * 64
        A_inv += i_bh * T * 64
        stride_16 = 16
        stride_64 = 64
    else:
        A_inv_diag += (bos*H + i_h) * 16
        A += (bos*H + i_h) * 64
        A_inv += (bos*H + i_h) * 64
        stride_16 = 16 * H
        stride_64 = 64 * H

    p_A_inv_11 = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 64, 0), (16, 16), (1, 0))
    p_A_inv_22 = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_A_inv_33 = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_A_inv_44 = tl.make_block_ptr(A_inv_diag, (T, 16), (stride_16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p_A_21 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, 64), (stride_64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p_A_final_11 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64, 0), (16, 16), (1, 0))
    p_A_final_22 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0))
    p_A_final_33 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0))
    p_A_final_44 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0))
    p_A_final_21 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_A_final_31 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_A_final_32 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0))
    p_A_final_41 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p_A_final_42 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0))
    p_A_final_43 = tl.make_block_ptr(A_inv, (T, 64), (stride_64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0))

    A_inv_11 = tl.load(p_A_inv_11, boundary_check=(0, 1))
    A_inv_22 = tl.load(p_A_inv_22, boundary_check=(0, 1))
    A_inv_33 = tl.load(p_A_inv_33, boundary_check=(0, 1))
    A_inv_44 = tl.load(p_A_inv_44, boundary_check=(0, 1))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1))
    A_32 = tl.load(p_A_32, boundary_check=(0, 1))
    A_31 = tl.load(p_A_31, boundary_check=(0, 1))
    A_43 = tl.load(p_A_43, boundary_check=(0, 1))
    A_42 = tl.load(p_A_42, boundary_check=(0, 1))
    A_41 = tl.load(p_A_41, boundary_check=(0, 1))

    A_inv_21 = -tl.dot(tl.dot(A_inv_22, A_21, input_precision='ieee'), A_inv_11, input_precision='ieee')
    A_inv_32 = -tl.dot(tl.dot(A_inv_33, A_32, input_precision='ieee'), A_inv_22, input_precision='ieee')
    A_inv_43 = -tl.dot(tl.dot(A_inv_44, A_43, input_precision='ieee'), A_inv_33, input_precision='ieee')
    A_inv_31 = -tl.dot(A_inv_33, tl.dot(A_31, A_inv_11, input_precision='ieee') +
                       tl.dot(A_32, A_inv_21, input_precision='ieee'), input_precision='ieee')
    A_inv_42 = -tl.dot(A_inv_44, tl.dot(A_42, A_inv_22, input_precision='ieee') +
                       tl.dot(A_43, A_inv_32, input_precision='ieee'), input_precision='ieee')
    A_inv_41 = -tl.dot(A_inv_44, tl.dot(A_41, A_inv_11, input_precision='ieee') + tl.dot(A_42, A_inv_21,
                       input_precision='ieee') + tl.dot(A_43, A_inv_31, input_precision='ieee'), input_precision='ieee')
    tl.store(p_A_final_11, A_inv_11.to(p_A_final_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_22, A_inv_22.to(p_A_final_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_33, A_inv_33.to(p_A_final_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_44, A_inv_44.to(p_A_final_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_21, A_inv_21.to(p_A_final_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_31, A_inv_31.to(p_A_final_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_32, A_inv_32.to(p_A_final_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_41, A_inv_41.to(p_A_final_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_42, A_inv_42.to(p_A_final_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_final_43, A_inv_43.to(p_A_final_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def solve_tril(A, cu_seqlens=None, head_first=True, output_dtype=torch.float32):
    """
    Compute the inverse of the lower triangular matrix
    A should be strictly lower triangular. Please make sure A.triu() == 0.
    return: (I + A)^-1
    """
    assert A.shape[-1] in [16, 32, 64]
    assert A.dtype == torch.float32, "A should be float32."
    assert A.is_contiguous(), "A should be contiguous."
    if head_first is True:
        B, H, T, BT = A.shape
        A_inv_diag = torch.empty(B, H, T, 16, device=A.device, dtype=torch.float32 if BT != 16 else output_dtype)
    else:
        B, T, H, BT = A.shape
        A_inv_diag = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float32 if BT != 16 else output_dtype)

    indices1 = prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    NT = len(indices1) if cu_seqlens is not None else triton.cdiv(T, 16)
    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        A_inv_diag=A_inv_diag,
        offsets=cu_seqlens,
        indices=indices1,
        T=T,
        BT=BT,
        H=H,
        HEAD_FIRST=head_first,
    )
    if BT == 16:
        return A_inv_diag

    if head_first is True:
        A_inv = torch.zeros(B, H, T, BT, device=A.device, dtype=output_dtype)
    else:
        A_inv = torch.zeros(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel
    indices2 = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(indices2) if cu_seqlens is not None else triton.cdiv(T,BT)
    merge_fn[NT, B * H](
        A=A,
        A_inv_diag=A_inv_diag,
        A_inv=A_inv,
        offsets=cu_seqlens,
        indices=indices2,
        T=T,
        H=H,
        HEAD_FIRST=head_first,
        USE_OFFSETS=cu_seqlens is not None
    )
    return A_inv


if __name__ == "__main__":
    B = 2
    H = 16
    T = 200
    chunk_size = 64
    D = 64
    head_first = True
    k = torch.nn.functional.normalize(torch.randn((B, H, T, D), dtype=torch.float32, device="cuda"), dim=-1)
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = torch.nn.functional.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, D)
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1).reshape(B, H, -1, chunk_size)[:, :, :T, :].contiguous()
    A_inv = solve_tril(A, head_first=head_first)

    A_inv_ref = torch.zeros_like(A_inv)
    for i in range(0, T, chunk_size):
        actual_Size = min(chunk_size, T - i)
        A_inv_ref[:, :, i:i+actual_Size, :actual_Size] = torch.inverse(
            A[:, :, i:i+actual_Size, :actual_Size] + torch.eye(actual_Size, device=A.device, dtype=A.dtype)[None, None, ...])

    from fla.ops.utils.testing import assert_close

    # breakpoint()
    assert_close("solve_tril", A_inv, A_inv_ref, 0.0001)
