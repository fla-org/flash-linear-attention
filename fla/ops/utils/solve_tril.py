# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from fla.backends import dispatch
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_TF32_SUPPORTED, IS_TMA_SUPPORTED, autotune_cache_kwargs, input_guard

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
assert FLA_TRIL_PRECISION in ('ieee', 'tf32', 'tf32x3'), (
    f"FLA_TRIL_PRECISION must be one of 'ieee', 'tf32', or 'tf32x3', but got {FLA_TRIL_PRECISION}"
)

DOT_PRECISION_AUTOTUNE_LIST = ['ieee']
# require TF32 support explicitly; TMA is also available on non-Nvidia backends
if IS_TMA_SUPPORTED and IS_TF32_SUPPORTED and FLA_TRIL_PRECISION != "ieee":
    DOT_PRECISION_AUTOTUNE_LIST.append(FLA_TRIL_PRECISION)


@triton.jit
def solve16x16(A, T, H: tl.constexpr, BC: tl.constexpr, m_A, m_I, offset=None, USE_TMA: tl.constexpr = False):
    """Invert a 16x16 diagonal block of I + A; m_A=None requires strictly lower-triangular input."""
    o_i = tl.arange(0, 16)
    if offset is None:
        i_t, i_c = 0, 0
    else:
        i_t, i_c = offset
    if USE_TMA:
        desc = make_tensor_descriptor(A, [T, BC], [H*BC, 1], [16, 16])
        b_A = desc.load([i_t, i_c]).to(tl.float32)
    else:
        p_A = A + (i_t + o_i[:, None]) * (H*BC) + (i_c + o_i[None, :])
        b_A = tl.load(p_A, mask=(i_t + o_i[:, None]) < T, other=0.).to(tl.float32)
    if m_A is not None:
        b_A = tl.where(m_A, b_A, 0.)
    b_A = -b_A
    for i in range(2, min(16, T - i_t)):
        b_a = -tl.load(A + (i_t + i) * H*BC + i_c + o_i)
        if m_A is not None:
            b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    return b_A + m_I


@triton.jit
def solve32x32(A, T, H: tl.constexpr, m_A, m_I, DOT_PRECISION: tl.constexpr, offset=0, USE_TMA: tl.constexpr = False):
    """Return the three lower 16x16 blocks of (I + A)^-1 for a 32-token tile."""
    b_Ai00 = solve16x16(A=A, T=T, H=H, BC=32, m_A=m_A, m_I=m_I, offset=(offset, 0), USE_TMA=USE_TMA)
    b_Ai11 = solve16x16(A=A, T=T, H=H, BC=32, m_A=m_A, m_I=m_I, offset=(offset + 16, 16), USE_TMA=USE_TMA)
    if USE_TMA:
        desc = make_tensor_descriptor(A, [T, 32], [H*32, 1], [16, 16])
        b_A10 = desc.load([offset + 16, 0]).to(tl.float32)
    else:
        o_i = tl.arange(0, 16)
        p_A10 = A + (offset + 16 + o_i[:, None]) * (H*32) + o_i[None, :]
        b_A10 = tl.load(p_A10, mask=(offset + 16 + o_i[:, None]) < T, other=0.).to(tl.float32)
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=DOT_PRECISION),
        b_Ai00,
        input_precision=DOT_PRECISION,
    )
    return b_Ai00, b_Ai10, b_Ai11


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': 'ieee'}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
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
    b_A = solve16x16(A=A, T=T, H=H, BC=BT, m_A=m_A, m_I=m_I, offset=(i_t * 16, offset), USE_TMA=USE_TMA)
    if not USE_TMA:
        p_Ai = Ai + o_t[:, None] * (H*16) + o_i[None, :]
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_t[:, None])
    else:
        desc_o = make_tensor_descriptor(Ai, [T, 16], [H*16, 1], [16, 16])
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
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
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_t = (i_t * BT + o_i).to(tl.int64)
    b_Ai_11, b_Ai_21, b_Ai_22 = solve32x32(
        A=A,
        T=T,
        H=H,
        m_A=None,
        m_I=m_I,
        DOT_PRECISION=DOT_PRECISION,
        offset=i_t * BT,
        USE_TMA=USE_TMA,
    )

    if not USE_TMA:
        p_Ai_11 = Ai + o_t[:, None] * (H*BT) + o_i[None, :]
        p_Ai_21 = Ai + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
        p_Ai_22 = Ai + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] < T))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
    else:
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
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
    b_Ai_11 = solve16x16(A=A, T=T, H=H, BC=BT, m_A=m_A, m_I=m_I, offset=(i_t * BT + 0, 0), USE_TMA=USE_TMA)
    b_Ai_22 = solve16x16(A=A, T=T, H=H, BC=BT, m_A=m_A, m_I=m_I, offset=(i_t * BT + 16, 16), USE_TMA=USE_TMA)
    b_Ai_33 = solve16x16(A=A, T=T, H=H, BC=BT, m_A=m_A, m_I=m_I, offset=(i_t * BT + 32, 32), USE_TMA=USE_TMA)
    b_Ai_44 = solve16x16(A=A, T=T, H=H, BC=BT, m_A=m_A, m_I=m_I, offset=(i_t * BT + 48, 48), USE_TMA=USE_TMA)

    if not USE_TMA:
        p_A_21 = A + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
        p_A_31 = A + (o_t[:, None] + 32) * (H*BT) + o_i[None, :]
        p_A_32 = A + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 16)
        p_A_41 = A + (o_t[:, None] + 48) * (H*BT) + o_i[None, :]
        p_A_42 = A + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 16)
        p_A_43 = A + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 32)
        b_A_21 = tl.load(p_A_21, mask=(o_t[:, None] + 16 < T), other=0.0).to(tl.float32)
        b_A_31 = tl.load(p_A_31, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
        b_A_32 = tl.load(p_A_32, mask=(o_t[:, None] + 32 < T), other=0.0).to(tl.float32)
        b_A_41 = tl.load(p_A_41, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
        b_A_42 = tl.load(p_A_42, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
        b_A_43 = tl.load(p_A_43, mask=(o_t[:, None] + 48 < T), other=0.0).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = Ai + o_t[:, None] * (H*BT) + o_i[None, :]
        p_Ai_22 = Ai + (o_t[:, None] + 16) * (H*BT) + (o_i[None, :] + 16)
        p_Ai_33 = Ai + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 32)
        p_Ai_44 = Ai + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 48)
        p_Ai_21 = Ai + (o_t[:, None] + 16) * (H*BT) + o_i[None, :]
        p_Ai_31 = Ai + (o_t[:, None] + 32) * (H*BT) + o_i[None, :]
        p_Ai_32 = Ai + (o_t[:, None] + 32) * (H*BT) + (o_i[None, :] + 16)
        p_Ai_41 = Ai + (o_t[:, None] + 48) * (H*BT) + o_i[None, :]
        p_Ai_42 = Ai + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 16)
        p_Ai_43 = Ai + (o_t[:, None] + 48) * (H*BT) + (o_i[None, :] + 32)
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] < T))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 32 < T))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 16 < T))
        tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 32 < T))
        tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 32 < T))
        tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
        tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
        tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), mask=(o_t[:, None] + 48 < T))
    else:
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@dispatch
@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
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
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel

    merge_fn[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=IS_TMA_SUPPORTED,
    )
    return Ai
