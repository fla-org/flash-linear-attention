# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import gather
from fla.utils import IS_GATHER_SUPPORTED, autotune_cache_kwargs


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk32(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,  # placeholder, do not delete
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_Aab = A_ab + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    p_Aab_inv = A_ab_inv + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_A[None, :]
    b_A_ab = tl.load(p_Aab, mask=m_A, other=0.0)
    b_A_ab = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A_ab, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A_ab, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A_ab, 0) * (tl.arange(0, BT) < i)
        b_A_ab = tl.where(mask[:, None], b_a, b_A_ab)
    b_A_ab += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    tl.store(p_Aab_inv, b_A_ab.to(p_Aab_inv.dtype.element_ty), mask=m_A)


@triton.jit
def _inv_diag16(b_L):
    # (I - L)^-1 = (I + L)(I + L^2)(I + L^4)(I + L^8), exact for strictly
    # lower-triangular 16x16 L (L^16 = 0). fp16 operands with fp32 accumulate:
    # strictly-lower gated entries and their Neumann products are bounded.
    o_i = tl.arange(0, 16)
    b_I = (o_i[:, None] == o_i[None, :]).to(tl.float16)
    b_L2 = tl.dot(b_L, b_L).to(tl.float16)
    b_L4 = tl.dot(b_L2, b_L2).to(tl.float16)
    b_L8 = tl.dot(b_L4, b_L4).to(tl.float16)
    b_T = tl.dot(b_I + b_L, b_I + b_L2)
    b_T = tl.dot(b_T.to(tl.float16), b_I + b_L4)
    b_T = tl.dot(b_T.to(tl.float16), b_I + b_L8)
    return b_T


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_safe(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,  # placeholder, do not delete
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T

    NB: tl.constexpr = 16
    NS: tl.constexpr = BT // NB
    o_i = tl.arange(0, NB)
    # hierarchical (I - L)^-1 over 16x16 blocks: exact nilpotent base inverses
    # combined level by level with M21 = M22 L21 M11
    o_c1 = NB + o_i
    m_lo = o_i[:, None] > o_i[None, :]
    b_zero = tl.zeros([NB, NB], dtype=tl.float32)

    r_0 = i_t * BT + o_i
    m_r0 = r_0 < T
    o_A0 = A_ab + (bos * H + i_h) * BT + r_0[:, None] * (H * BT)
    o_I0 = A_ab_inv + (bos * H + i_h) * BT + r_0[:, None] * (H * BT)
    b_L00 = tl.where(m_lo, tl.load(o_A0 + o_i[None, :], mask=m_r0[:, None], other=0.0), 0.0).to(tl.float16)
    b_T00 = _inv_diag16(b_L00)
    tl.store(o_I0 + o_i[None, :], b_T00, mask=m_r0[:, None])
    if NS >= 2:
        r_1 = i_t * BT + NB + o_i
        m_r1 = r_1 < T
        o_A1 = A_ab + (bos * H + i_h) * BT + r_1[:, None] * (H * BT)
        o_I1 = A_ab_inv + (bos * H + i_h) * BT + r_1[:, None] * (H * BT)
        b_L11 = tl.where(m_lo, tl.load(o_A1 + o_c1[None, :], mask=m_r1[:, None], other=0.0), 0.0).to(tl.float16)
        b_L10 = tl.load(o_A1 + o_i[None, :], mask=m_r1[:, None], other=0.0).to(tl.float16)
        b_T11 = _inv_diag16(b_L11)
        b_T10 = tl.dot(b_T11.to(tl.float16), tl.dot(b_L10, b_T00.to(tl.float16)).to(tl.float16))
        tl.store(o_I1 + o_i[None, :], b_T10, mask=m_r1[:, None])
        tl.store(o_I1 + o_c1[None, :], b_T11, mask=m_r1[:, None])
        tl.store(o_I0 + o_c1[None, :], b_zero, mask=m_r0[:, None])
        if NS == 4:
            o_c2 = 2 * NB + o_i
            o_c3 = 3 * NB + o_i
            r_2 = i_t * BT + 2 * NB + o_i
            r_3 = i_t * BT + 3 * NB + o_i
            m_r2 = r_2 < T
            m_r3 = r_3 < T
            o_A2 = A_ab + (bos * H + i_h) * BT + r_2[:, None] * (H * BT)
            o_A3 = A_ab + (bos * H + i_h) * BT + r_3[:, None] * (H * BT)
            o_I2 = A_ab_inv + (bos * H + i_h) * BT + r_2[:, None] * (H * BT)
            o_I3 = A_ab_inv + (bos * H + i_h) * BT + r_3[:, None] * (H * BT)
            b_L22 = tl.where(m_lo, tl.load(o_A2 + o_c2[None, :], mask=m_r2[:, None], other=0.0), 0.0).to(tl.float16)
            b_L33 = tl.where(m_lo, tl.load(o_A3 + o_c3[None, :], mask=m_r3[:, None], other=0.0), 0.0).to(tl.float16)
            b_L20 = tl.load(o_A2 + o_i[None, :], mask=m_r2[:, None], other=0.0).to(tl.float16)
            b_L21 = tl.load(o_A2 + o_c1[None, :], mask=m_r2[:, None], other=0.0).to(tl.float16)
            b_L30 = tl.load(o_A3 + o_i[None, :], mask=m_r3[:, None], other=0.0).to(tl.float16)
            b_L31 = tl.load(o_A3 + o_c1[None, :], mask=m_r3[:, None], other=0.0).to(tl.float16)
            b_L32 = tl.load(o_A3 + o_c2[None, :], mask=m_r3[:, None], other=0.0).to(tl.float16)

            b_T22 = _inv_diag16(b_L22)
            b_T33 = _inv_diag16(b_L33)
            b_T32 = tl.dot(b_T33.to(tl.float16), tl.dot(b_L32, b_T22.to(tl.float16)).to(tl.float16))
            b_P1 = tl.dot(b_L20, b_T00.to(tl.float16)) + tl.dot(b_L21, b_T10.to(tl.float16))
            b_P2 = tl.dot(b_L30, b_T00.to(tl.float16)) + tl.dot(b_L31, b_T10.to(tl.float16))
            b_T20 = tl.dot(b_T22.to(tl.float16), b_P1.to(tl.float16))
            b_T21 = tl.dot(b_T22.to(tl.float16), tl.dot(b_L21, b_T11.to(tl.float16)).to(tl.float16))
            b_T30 = tl.dot(b_T32.to(tl.float16), b_P1.to(tl.float16)) + tl.dot(b_T33.to(tl.float16), b_P2.to(tl.float16))
            b_T31 = tl.dot(b_T32.to(tl.float16), tl.dot(b_L21, b_T11.to(tl.float16)).to(tl.float16)) \
                + tl.dot(b_T33.to(tl.float16), tl.dot(b_L31, b_T11.to(tl.float16)).to(tl.float16))

            tl.store(o_I0 + o_c2[None, :], b_zero, mask=m_r0[:, None])
            tl.store(o_I0 + o_c3[None, :], b_zero, mask=m_r0[:, None])
            tl.store(o_I1 + o_c2[None, :], b_zero, mask=m_r1[:, None])
            tl.store(o_I1 + o_c3[None, :], b_zero, mask=m_r1[:, None])
            tl.store(o_I2 + o_i[None, :], b_T20, mask=m_r2[:, None])
            tl.store(o_I2 + o_c1[None, :], b_T21, mask=m_r2[:, None])
            tl.store(o_I2 + o_c2[None, :], b_T22, mask=m_r2[:, None])
            tl.store(o_I2 + o_c3[None, :], b_zero, mask=m_r2[:, None])
            tl.store(o_I3 + o_i[None, :], b_T30, mask=m_r3[:, None])
            tl.store(o_I3 + o_c1[None, :], b_T31, mask=m_r3[:, None])
            tl.store(o_I3 + o_c2[None, :], b_T32, mask=m_r3[:, None])
            tl.store(o_I3 + o_c3[None, :], b_T33, mask=m_r3[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
    A_ab,
    A_ab_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATHER_SUPPORTED: tl.constexpr = IS_GATHER_SUPPORTED,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_c = tl.arange(0, BC)
    o_t1 = i_t * BT + o_c
    o_t2 = i_t * BT + BC + o_c
    m_t1 = o_t1 < T
    m_t2 = o_t2 < T
    p_A1 = A_ab + (bos*H + i_h) * BT + o_t1[:, None] * (H*BT) + o_c[None, :]
    p_A2 = A_ab + (bos*H + i_h) * BT + o_t2[:, None] * (H*BT) + (BC + o_c)[None, :]
    p_A3 = A_ab + (bos*H + i_h) * BT + o_t2[:, None] * (H*BT) + o_c[None, :]
    p_A_inv1 = A_ab_inv + (bos*H + i_h) * BT + o_t1[:, None] * (H*BT) + o_c[None, :]
    p_A_inv2 = A_ab_inv + (bos*H + i_h) * BT + o_t2[:, None] * (H*BT) + (BC + o_c)[None, :]
    p_A_inv3 = A_ab_inv + (bos*H + i_h) * BT + o_t2[:, None] * (H*BT) + o_c[None, :]
    p_A_inv4 = A_ab_inv + (bos*H + i_h) * BT + o_t1[:, None] * (H*BT) + (BC + o_c)[None, :]

    b_A = tl.load(p_A1, mask=m_t1[:, None] & (o_c[None, :] < BT), other=0.0)
    b_A2 = tl.load(p_A2, mask=m_t2[:, None] & ((BC + o_c)[None, :] < BT), other=0.0)
    b_A3 = tl.load(p_A3, mask=m_t2[:, None] & (o_c[None, :] < BT), other=0.0)
    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BC], i, dtype=tl.int16)
            # [1, BK] -> [BK]
            b_a = tl.sum(gather(b_A, row_idx, axis=0), 0)
            b_a2 = tl.sum(gather(b_A2, row_idx, axis=0), 0)
        else:
            mask = tl.arange(0, BC) == i
            b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
            b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        mask = tl.arange(0, BC) == i
        # b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        # b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3), b_A)
    # tl.debug_barrier()
    tl.store(p_A_inv1, b_A.to(p_A_inv1.dtype.element_ty, fp_downcast_rounding="rtne"),
             mask=m_t1[:, None] & (o_c[None, :] < BT))
    tl.store(p_A_inv2, b_A2.to(p_A_inv2.dtype.element_ty, fp_downcast_rounding="rtne"),
             mask=m_t2[:, None] & ((BC + o_c)[None, :] < BT))
    tl.store(p_A_inv3, b_A3.to(p_A_inv3.dtype.element_ty, fp_downcast_rounding="rtne"),
             mask=m_t2[:, None] & (o_c[None, :] < BT))
    # causal mask
    tl.store(p_A_inv4, tl.zeros([BC, BC], dtype=tl.float32).to(
        p_A_inv4.dtype.element_ty), mask=m_t1[:, None] & ((BC + o_c)[None, :] < BT))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(
    w,
    u,
    ag,
    v,
    A_ab_inv,
    A_ak,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_s = tl.arange(0, BT)

    o_t = i_t * BT + o_s
    m_t = o_t < T
    m_A = m_t[:, None] & (o_s[None, :] < BT)
    p_A_ab_inv = A_ab_inv + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_s[None, :]
    p_A_ak = A_ak + (bos*H + i_h) * BT + o_t[:, None] * (H*BT) + o_s[None, :]

    b_Aab_inv = tl.load(p_A_ab_inv, mask=m_A, other=0.0)
    b_Aak = tl.load(p_A_ak, mask=m_A, other=0.0)
    b_Aab_inv = tl.where(o_s[:, None] >= o_s[None, :], b_Aab_inv, 0)
    b_Aak = tl.where(o_s[:, None] > o_s[None, :], b_Aak, 0)
    # let's use tf32 here
    b_Aak = tl.dot(b_Aab_inv, b_Aak)
    # (SY 01/04) should be bf16 or tf32? To verify.
    b_Aak = b_Aak.to(v.dtype.element_ty, fp_downcast_rounding="rtne")
    b_Aab_inv = b_Aab_inv.to(ag.dtype.element_ty, fp_downcast_rounding="rtne")

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_ag = ag + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_w = w + (bos*H + i_h) * K + o_t[:, None] * (H*K) + o_k[None, :]
        b_ag = tl.load(p_ag, mask=m_k, other=0.0)
        b_w = tl.dot(b_Aab_inv, b_ag)  # both bf16 or fp16
        tl.store(p_w, b_w.to(p_w.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_k)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        p_u = u + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_u = tl.dot(b_Aak, b_v)  # both bf16 or fp16
        tl.store(p_u, b_u.to(p_u.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m_v)


def wu_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)

    w = torch.empty_like(ag)
    u = torch.empty_like(v)
    wu_fwd_kernel[(NT, B * H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


def prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, _ = ag.shape
    BT = chunk_size

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    if safe_gate:
        # the tensor-core inversion relies on bounded-gate decay to keep the
        # Neumann products small; unbounded gates fall back to the serial one
        fwd_fn = prepare_wy_repr_fwd_kernel_safe
    else:
        fwd_fn = prepare_wy_repr_fwd_kernel_chunk64 if BT == 64 else prepare_wy_repr_fwd_kernel_chunk32
    A_ab_inv = torch.empty_like(A_ab)
    fwd_fn[(NT, B * H)](
        A_ab=A_ab,
        A_ab_inv=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
    )
    w, u = wu_fwd(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
    )
    return w, u, A_ab_inv


fwd_prepare_wy_repr = prepare_wy_repr_fwd

fwd_wu = wu_fwd
