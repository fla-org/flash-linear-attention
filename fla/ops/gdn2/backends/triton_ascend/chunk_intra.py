# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN-2 forward intra kernels for triton-ascend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.backends import TritonAscendBackend, register
from fla.ops.common.backends.triton_ascend.chunk_delta_h import (
    _pad_dense_tail_tokens,
    need_dma_mask,
    time_axis_needs_mask,
)
from fla.ops.gdn2.chunk_intra import chunk_gdn2_fwd_intra
from fla.ops.gdn2.gate import gdn2_local_log_cumsum
from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.ops.utils.triton_ascend.addressing import mload, mstore, token_rows
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import compute_row_tile_block_size, iter_axis_launch_chunks

# the grouped diagonal kernel keeps the padded K dimension in one UB slab.
MAX_FWD_INTRA_BK = 256
SUB_CHUNK_SIZE = 16
_BC = SUB_CHUNK_SIZE
TOKEN_GROUP_SIZE = 8
INTER_MEM_MULT = 18.0
SAFETY_MARGIN = 0.80
FALLBACK_BK = 16
MAX_INTER_BK = 64
LAUNCH_BLOCK_BUDGET = 4096
# Disable auto-multi-buffer and AutoBlockify on the inter kernels for CANN 9.1.
INTER_COMPILE_KWARGS = ascend_compile_kwargs(blacklist_auto_blockify=True)


@triton.jit
def dma_rows(ptr, origin, rows, stride, MASK_DMA: tl.constexpr):
    """Block DMA along the token axis when ``MASK_DMA`` is false."""
    if MASK_DMA:
        return ptr + origin * stride + rows * stride
    return (ptr + origin * stride) + rows * stride


def _get_inter_bk(K: int) -> int:
    return compute_row_tile_block_size(
        SUB_CHUNK_SIZE,
        K,
        INTER_MEM_MULT,
        tiling_row=False,
        safety_margin=SAFETY_MARGIN,
        fallback=FALLBACK_BK,
        min_block=16,
        max_block=min(MAX_INTER_BK, triton.next_power_of_2(K)),
    )


def get_inter_bk(K: int) -> int:
    return _get_inter_bk(K)


def launch_diag_kernel(
    kernel,
    *,
    nt: int,
    nc: int,
    bh_total: int,
    kernel_kwargs: dict,
    sync_stream=None,
) -> None:
    budget = LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nt_step = nt if nt * nc * bh_total <= budget else max(1, budget // max(nc * bh_total, 1))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        for nc_off, nc_len in iter_axis_launch_chunks(nc, nt_len * bh_total):
            kernel_kwargs['NC_OFFSET'] = nc_off
            for bh_off, bh_len in iter_axis_launch_chunks(bh_total, nt_len * nc_len):
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nt_len, nc_len, bh_len)](**kernel_kwargs)
    if sync_stream is not None:
        sync_stream.synchronize()


def launch_inter_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    budget = LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nt_step = nt if nt * bh_total <= budget else max(1, budget // max(bh_total, 1))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        for bh_off, bh_len in iter_axis_launch_chunks(bh_total, nt_len):
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](**kernel_kwargs, **INTER_COMPILE_KWARGS)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_intra_grouped_npu(
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
    BR: tl.constexpr,
    ROW_GROUP: tl.constexpr,
    IS_HEAD_MAJOR: tl.constexpr,
    USE_RAW_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    """Build one group of causal Aqk/Akk rows without unstable gate factoring."""
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1,
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    # promote before token-stride products, including packed inputs and Aqk/Akk stores.
    i_ts = tl.cast(i_t, tl.int64) * BT + i_i * BC
    i_ti = i_ts + ROW_GROUP * BR
    if i_ti >= T:
        return

    o_r = tl.arange(0, BR)
    if MASK_DMA:
        _, _, m_r = token_rows(i_ti, T, BR)
    else:
        m_r = o_r < BR
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_rk = m_r[:, None] & m_k[None, :]

    if IS_HEAD_MAJOR:
        if IS_VARLEN:
            input_offset = tl.cast(i_h, tl.int64) * T_max * K + bos * K
        else:
            input_offset = ((tl.cast(i_b, tl.int64) * H + i_h) * T) * K
        input_stride = K
    else:
        input_offset = (bos * H + i_h) * K
        input_stride = H * K
    q += input_offset
    k += input_offset
    g += input_offset
    b += input_offset
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BC

    p_q = dma_rows(q, i_ti, o_r[:, None], input_stride, MASK_DMA) + o_k[None, :]
    p_k = dma_rows(k, i_ti, o_r[:, None], input_stride, MASK_DMA) + o_k[None, :]
    p_g = dma_rows(g, i_ti, o_r[:, None], input_stride, MASK_DMA) + o_k[None, :]
    p_b = dma_rows(b, i_ti, o_r[:, None], input_stride, MASK_DMA) + o_k[None, :]
    b_q = mload(p_q, m_rk, MASK_DMA).to(tl.float32)
    b_k = mload(p_k, m_rk, MASK_DMA).to(tl.float32)
    b_g = mload(p_g, m_rk, MASK_DMA).to(tl.float32)
    b_b = mload(p_b, m_rk, MASK_DMA).to(tl.float32)
    b_k *= b_b

    b_interval = tl.zeros([BR, BK], tl.float32)
    for j in tl.static_range((ROW_GROUP + 1) * BR - 1, -1, -1):
        i_j = i_ts + j
        m_j = i_j < T
        p_kj = k + i_j * input_stride + o_k
        p_gj = g + i_j * input_stride + o_k
        b_kj = tl.load(p_kj, mask=m_j & m_k, other=0.0).to(tl.float32)
        b_gj = tl.load(p_gj, mask=m_j & m_k, other=0.0).to(tl.float32)
        b_decay = exp2(b_g - b_gj[None, :])
        if USE_RAW_G:
            b_decay = exp2(b_interval * 1.4426950408889634)
            b_interval += tl.where((ROW_GROUP * BR + o_r)[:, None] >= j, b_gj[None, :], 0.)
        b_kgj = tl.where(m_k[None, :], b_kj[None, :] * b_decay, 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1)
        row_in_subchunk = ROW_GROUP * BR + o_r
        p_Aqk = dma_rows(Aqk, i_ti, o_r, H * BT, MASK_DMA) + i_i * BC + j
        p_Akk = dma_rows(Akk, i_ti, o_r, H * BC, MASK_DMA) + j
        m_store = m_r & m_j & (j <= row_in_subchunk)
        mstore(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), m_store, MASK_DMA)
        mstore(p_Akk, b_Akk.to(Akk.dtype.element_ty), m_r & m_j & (j < row_in_subchunk), MASK_DMA)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_diag_solve_npu(
    Akkd,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    Akkd += (bos * H + i_h).to(tl.int64) * BC
    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    o_t = i_ti + tl.arange(0, BC)
    m_t = (o_t >= 0) & (o_t < T)
    o_i_2 = tl.arange(0, BC)
    m_i_2 = (o_i_2 >= 0) & (o_i_2 < BC)
    m_p_Akk = m_t[:, None] & m_i_2[None, :]
    p_Akk = Akkd + o_t[:, None] * (H * BC) + o_i_2[None, :]
    b_Akk = tl.load(p_Akk, mask=m_p_Akk, other=0).to(tl.float32)
    b_Ai = -tl.where(m_A, b_Akk, 0)
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akkd + (i_ti + i).to(tl.int64) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akkd.dtype.element_ty), mask=m_p_Akk)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_inter_products_npu(
    q,
    k,
    g,
    g_raw,
    b,
    Aqk,
    Akkx,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    DST_BLOCK: tl.constexpr,
    SRC_BLOCK: tl.constexpr,
    IS_HEAD_MAJOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
    USE_RAW_G: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1,
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    i_tc_dst = i_t * BT + DST_BLOCK * BC
    i_tc_src = i_t * BT + SRC_BLOCK * BC

    if IS_HEAD_MAJOR:
        if IS_VARLEN:
            input_offset = tl.cast(i_h, tl.int64) * T_max * K + bos * K
        else:
            input_offset = ((tl.cast(i_b, tl.int64) * H + i_h) * T) * K
        input_stride = K
    else:
        input_offset = (bos * H + i_h) * K
        input_stride = H * K
    q += input_offset
    k += input_offset
    g += input_offset
    b += input_offset
    Aqk += (bos * H + i_h) * BT
    Akkx += (bos * H + i_h) * BT

    src_rows = tl.arange(0, BC)
    dst_rows = tl.arange(0, BC)
    if MASK_DMA:
        _, _, m_src = token_rows(i_tc_src, T, BC)
        _, _, m_dst = token_rows(i_tc_dst, T, BC)
    else:
        m_src = src_rows < BC
        m_dst = dst_rows < BC
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        m_p_k_src = m_src[:, None] & m_k[None, :]
        p_k_src = dma_rows(k, i_tc_src, src_rows[:, None], input_stride, MASK_DMA) + o_k[None, :]
        p_g_src = dma_rows(g, i_tc_src, src_rows[:, None], input_stride, MASK_DMA) + o_k[None, :]
        b_k_src = mload(p_k_src, m_p_k_src, MASK_DMA).to(tl.float32)
        b_g_src = mload(p_g_src, m_p_k_src, MASK_DMA).to(tl.float32)

        m_p_q_dst = m_dst[:, None] & m_k[None, :]
        p_q_dst = dma_rows(q, i_tc_dst, dst_rows[:, None], input_stride, MASK_DMA) + o_k[None, :]
        p_k_dst = dma_rows(k, i_tc_dst, dst_rows[:, None], input_stride, MASK_DMA) + o_k[None, :]
        p_g_dst = dma_rows(g, i_tc_dst, dst_rows[:, None], input_stride, MASK_DMA) + o_k[None, :]
        p_b_dst = dma_rows(b, i_tc_dst, dst_rows[:, None], input_stride, MASK_DMA) + o_k[None, :]
        b_q_dst = mload(p_q_dst, m_p_q_dst, MASK_DMA).to(tl.float32)
        b_k_dst = mload(p_k_dst, m_p_q_dst, MASK_DMA).to(tl.float32)
        b_g_dst = mload(p_g_dst, m_p_q_dst, MASK_DMA).to(tl.float32)
        b_b_dst = mload(p_b_dst, m_p_q_dst, MASK_DMA).to(tl.float32)
        if IS_HEAD_MAJOR:
            p_gn = g + tl.cast(i_tc_dst, tl.int64) * K + o_k
        else:
            p_gn = g + tl.cast(i_tc_dst, tl.int64) * H * K + o_k
        b_gn_dst = mload(
            p_gn,
            m_k & (i_tc_dst < T),
            MASK_DMA,
        ).to(tl.float32)
        b_gq = tl.where(m_dst[:, None], exp2(b_g_dst - b_gn_dst[None, :]), 0.0)
        b_ks = b_gn_dst[None, :] - b_g_src
        if USE_RAW_G:
            g_raw_ptr = g_raw + (bos * H + i_h) * K
            b_gp = gdn2_local_log_cumsum(g_raw_ptr, i_tc_dst.to(tl.int64), min(i_tc_dst + BC, T), o_k, H, K, BC)
            b_ks = gdn2_local_log_cumsum(g_raw_ptr, i_tc_src.to(tl.int64), min(i_tc_src + BC, T), o_k, H, K, BC, True)
            o_gap = i_t.to(tl.int64) * BT + tl.arange(0, BT)
            b_gap = tl.load(
                g_raw_ptr + o_gap[:, None] * H * K + o_k[None, :],
                (o_gap[:, None] >= i_tc_src + BC) & (o_gap[:, None] < min(i_tc_dst, T)) & m_k[None, :], other=0,
            ).to(tl.float32)
            b_ks += tl.sum(b_gap, axis=0)[None, :] * 1.4426950408889634
            b_gq = tl.where(m_dst[:, None], exp2(b_gp), 0.)
        b_kgt = tl.trans(b_k_src * exp2(b_ks))
        b_Aqk += tl.dot(b_q_dst * b_gq, b_kgt, allow_tf32=False)
        b_Akk += tl.dot((b_b_dst * b_k_dst) * b_gq, b_kgt, allow_tf32=False)

    o_col = SRC_BLOCK * BC + tl.arange(0, BC)
    m_col = (o_col >= 0) & (o_col < BT)
    m_p_Aqk = m_dst[:, None] & m_col[None, :]
    p_Aqk = dma_rows(Aqk, i_tc_dst, dst_rows[:, None], H * BT, MASK_DMA) + o_col[None, :]
    p_Akkx = dma_rows(Akkx, i_tc_dst, dst_rows[:, None], H * BT, MASK_DMA) + o_col[None, :]
    mstore(p_Aqk, (b_Aqk * scale).to(Aqk.dtype.element_ty), m_p_Aqk, MASK_DMA)
    mstore(p_Akkx, b_Akk, m_p_Aqk, MASK_DMA)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_inter_solve_npu(
    Akkd,
    Akkx,
    Akk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1,
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_tc0 + BC
    i_tc2 = i_tc0 + 2 * BC
    i_tc3 = i_tc0 + 3 * BC

    base = bos * H + i_h
    Akkd += base * BC
    Akkx += base * BT
    Akk += base * BT

    o_t = i_tc1 + tl.arange(0, BC)
    m_t = (o_t >= 0) & (o_t < T)
    o_i = tl.arange(0, BC)
    m_i = (o_i >= 0) & (o_i < BT)
    m_p_Akkx10 = m_t[:, None] & m_i[None, :]
    p_Akkx10 = Akkx + o_t[:, None] * (H * BT) + o_i[None, :]
    b_Akk10 = tl.load(p_Akkx10, mask=m_p_Akkx10, other=0).to(tl.float32)
    if NC >= 3:
        o_t_2 = i_tc2 + tl.arange(0, BC)
        m_t_2 = (o_t_2 >= 0) & (o_t_2 < T)
        m_p_Akkx20 = m_t_2[:, None] & m_i[None, :]
        p_Akkx20 = Akkx + o_t_2[:, None] * (H * BT) + o_i[None, :]
        o_t_3 = BC + tl.arange(0, BC)
        m_t_3 = (o_t_3 >= 0) & (o_t_3 < BT)
        m_p_Akkx21 = m_t_2[:, None] & m_t_3[None, :]
        p_Akkx21 = Akkx + o_t_2[:, None] * (H * BT) + o_t_3[None, :]
        b_Akk20 = tl.load(p_Akkx20, mask=m_p_Akkx20, other=0).to(tl.float32)
        b_Akk21 = tl.load(p_Akkx21, mask=m_p_Akkx21, other=0).to(tl.float32)
    if NC >= 4:
        o_t_2 = i_tc3 + tl.arange(0, BC)
        m_t_2 = (o_t_2 >= 0) & (o_t_2 < T)
        m_p_Akkx30 = m_t_2[:, None] & m_i[None, :]
        p_Akkx30 = Akkx + o_t_2[:, None] * (H * BT) + o_i[None, :]
        o_t_3 = BC + tl.arange(0, BC)
        m_t_3 = (o_t_3 >= 0) & (o_t_3 < BT)
        m_p_Akkx31 = m_t_2[:, None] & m_t_3[None, :]
        p_Akkx31 = Akkx + o_t_2[:, None] * (H * BT) + o_t_3[None, :]
        o_t_4 = 2 * BC + tl.arange(0, BC)
        m_t_4 = (o_t_4 >= 0) & (o_t_4 < BT)
        m_p_Akkx32 = m_t_2[:, None] & m_t_4[None, :]
        p_Akkx32 = Akkx + o_t_2[:, None] * (H * BT) + o_t_4[None, :]
        b_Akk30 = tl.load(p_Akkx30, mask=m_p_Akkx30, other=0).to(tl.float32)
        b_Akk31 = tl.load(p_Akkx31, mask=m_p_Akkx31, other=0).to(tl.float32)
        b_Akk32 = tl.load(p_Akkx32, mask=m_p_Akkx32, other=0).to(tl.float32)

    o_t_2 = i_tc0 + tl.arange(0, BC)
    m_t_2 = (o_t_2 >= 0) & (o_t_2 < T)
    m_i_2 = (o_i >= 0) & (o_i < BC)
    m_p_Akk00 = m_t_2[:, None] & m_i_2[None, :]
    p_Akk00 = Akkd + o_t_2[:, None] * (H * BC) + o_i[None, :]
    m_p_Akk11 = m_t[:, None] & m_i_2[None, :]
    p_Akk11 = Akkd + o_t[:, None] * (H * BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_p_Akk00, other=0).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, mask=m_p_Akk11, other=0).to(tl.float32)
    if NC >= 3:
        o_t_3 = i_tc2 + tl.arange(0, BC)
        m_t_3 = (o_t_3 >= 0) & (o_t_3 < T)
        m_p_Akk22 = m_t_3[:, None] & m_i_2[None, :]
        p_Akk22 = Akkd + o_t_3[:, None] * (H * BC) + o_i[None, :]
        b_Ai22 = tl.load(p_Akk22, mask=m_p_Akk22, other=0).to(tl.float32)
    if NC >= 4:
        o_t_3 = i_tc3 + tl.arange(0, BC)
        m_t_3 = (o_t_3 >= 0) & (o_t_3 < T)
        m_p_Akk33 = m_t_3[:, None] & m_i_2[None, :]
        p_Akk33 = Akkd + o_t_3[:, None] * (H * BC) + o_i[None, :]
        b_Ai33 = tl.load(p_Akk33, mask=m_p_Akk33, other=0).to(tl.float32)

    # preserve Ai11 before the Ai10 Ascend tl.dot clobbers it; later rhs dots and the store need the original
    b_Ai11_pristine = b_Ai11 + 0.0
    if NC >= 3:
        # Ai21 consumes Ai22 as lhs. Ai20 needs its own lhs copy.
        # Ai32 reads Ai22 as rhs, which leaves the store copy intact.
        b_Ai22_for_store = b_Ai22 + 0.0
        b_Ai22_for_Ai20 = b_Ai22 + 0.0
    if NC >= 4:
        # preserve Ai33 before the Ai32 Ascend tl.dot clobbers it; Ai31, Ai30, and the store need independent values
        b_Ai33_for_store = b_Ai33 + 0.0
        b_Ai33_for_Ai31 = b_Ai33 + 0.0
        b_Ai33_for_Ai30 = b_Ai33 + 0.0
        # preserve Akk31/Akk32 before their Ai31 Ascend tl.dot lhs uses; Ai30 needs fresh lhs tiles
        b_Akk31_for_Ai30 = b_Akk31 + 0.0
        b_Akk32_for_Ai30 = b_Akk32 + 0.0

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, allow_tf32=False), b_Ai00, allow_tf32=False)
    if NC >= 3:
        b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, allow_tf32=False), b_Ai11_pristine, allow_tf32=False)
        b_Ai20 = -tl.dot(
            b_Ai22_for_Ai20,
            tl.dot(b_Akk20, b_Ai00, allow_tf32=False) + tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
            allow_tf32=False,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, allow_tf32=False), b_Ai22_for_store, allow_tf32=False)
        b_Ai31 = -tl.dot(
            b_Ai33_for_Ai31,
            tl.dot(b_Akk31, b_Ai11_pristine, allow_tf32=False) + tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
            allow_tf32=False,
        )
        b_Ai30 = -tl.dot(
            b_Ai33_for_Ai30,
            tl.dot(b_Akk30, b_Ai00, allow_tf32=False)
            + tl.dot(b_Akk31_for_Ai30, b_Ai10, allow_tf32=False)
            + tl.dot(b_Akk32_for_Ai30, b_Ai20, allow_tf32=False),
            allow_tf32=False,
        )

    m_p_Akk00 = m_t_2[:, None] & m_i[None, :]
    p_Akk00 = Akk + o_t_2[:, None] * (H * BT) + o_i[None, :]
    p_Akk10 = Akk + o_t[:, None] * (H * BT) + o_i[None, :]
    o_t_3 = BC + tl.arange(0, BC)
    m_t_3 = (o_t_3 >= 0) & (o_t_3 < BT)
    m_p_Akk11 = m_t[:, None] & m_t_3[None, :]
    p_Akk11 = Akk + o_t[:, None] * (H * BT) + o_t_3[None, :]
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), mask=m_p_Akk00)
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), mask=m_p_Akkx10)
    tl.store(p_Akk11, b_Ai11_pristine.to(Akk.dtype.element_ty), mask=m_p_Akk11)
    if NC >= 3:
        o_t_4 = i_tc2 + tl.arange(0, BC)
        m_t_4 = (o_t_4 >= 0) & (o_t_4 < T)
        m_p_Akk20 = m_t_4[:, None] & m_i[None, :]
        p_Akk20 = Akk + o_t_4[:, None] * (H * BT) + o_i[None, :]
        m_p_Akk21 = m_t_4[:, None] & m_t_3[None, :]
        p_Akk21 = Akk + o_t_4[:, None] * (H * BT) + o_t_3[None, :]
        o_t_5 = 2 * BC + tl.arange(0, BC)
        m_t_5 = (o_t_5 >= 0) & (o_t_5 < BT)
        m_p_Akk22 = m_t_4[:, None] & m_t_5[None, :]
        p_Akk22 = Akk + o_t_4[:, None] * (H * BT) + o_t_5[None, :]
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), mask=m_p_Akk20)
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), mask=m_p_Akk21)
        tl.store(p_Akk22, b_Ai22_for_store.to(Akk.dtype.element_ty), mask=m_p_Akk22)
    if NC >= 4:
        o_t_4 = i_tc3 + tl.arange(0, BC)
        m_t_4 = (o_t_4 >= 0) & (o_t_4 < T)
        m_p_Akk30 = m_t_4[:, None] & m_i[None, :]
        p_Akk30 = Akk + o_t_4[:, None] * (H * BT) + o_i[None, :]
        m_p_Akk31 = m_t_4[:, None] & m_t_3[None, :]
        p_Akk31 = Akk + o_t_4[:, None] * (H * BT) + o_t_3[None, :]
        o_t_5 = 2 * BC + tl.arange(0, BC)
        m_t_5 = (o_t_5 >= 0) & (o_t_5 < BT)
        m_p_Akk32 = m_t_4[:, None] & m_t_5[None, :]
        p_Akk32 = Akk + o_t_4[:, None] * (H * BT) + o_t_5[None, :]
        o_t_6 = 3 * BC + tl.arange(0, BC)
        m_t_6 = (o_t_6 >= 0) & (o_t_6 < BT)
        m_p_Akk33 = m_t_4[:, None] & m_t_6[None, :]
        p_Akk33 = Akk + o_t_4[:, None] * (H * BT) + o_t_6[None, :]
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), mask=m_p_Akk30)
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), mask=m_p_Akk31)
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), mask=m_p_Akk32)
        tl.store(p_Akk33, b_Ai33_for_store.to(Akk.dtype.element_ty), mask=m_p_Akk33)


def chunk_gdn2_fwd_intra_verifier(
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
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    g_raw: torch.Tensor | None = None,
) -> tuple[bool, str | None]:
    if chunk_size != 64:
        return False, f"GDN-2 Ascend intra requires chunk_size=64, got {chunk_size}"
    K = int(k.shape[-1])
    BK = triton.next_power_of_2(K)
    if BK > MAX_FWD_INTRA_BK:
        return False, (
            f"GDN-2 Ascend intra requires next_power_of_2(K) <= {MAX_FWD_INTRA_BK} "
            f"for UB capacity, got K={K} (BK={BK})"
        )
    float_tensors = (q, k, v, gk, b, w_gate) + ((g_raw,) if g_raw is not None else ())
    tensors = (*float_tensors, *(t for t in (cu_seqlens, chunk_indices) if t is not None))
    if any(t.device.type != "npu" for t in tensors):
        return False, "GDN-2 Ascend intra requires NPU tensors"
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if any(t.dtype not in supported for t in float_tensors):
        return False, "GDN-2 Ascend intra received an unsupported dtype"
    return True, None


@register(chunk_gdn2_fwd_intra, backend=TritonAscendBackend, verifier=chunk_gdn2_fwd_intra_verifier)
@input_guard
def chunk_gdn2_fwd_intra_npu(
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
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    g_raw: torch.Tensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size
    if (
        cu_seqlens is None
        and T >= 4 * BT
        and T % BT
        and triton.next_power_of_2(K) == K
        and triton.next_power_of_2(v.shape[-1]) == v.shape[-1]
    ):
        T_pad = triton.cdiv(T, BT) * BT
        outputs = chunk_gdn2_fwd_intra_npu(
            q=_pad_dense_tail_tokens(q, T_pad),
            k=_pad_dense_tail_tokens(k, T_pad),
            v=_pad_dense_tail_tokens(v, T_pad),
            gk=_pad_dense_tail_tokens(gk, T_pad, repeat_last=True),
            g_raw=_pad_dense_tail_tokens(g_raw, T_pad),
            b=_pad_dense_tail_tokens(b, T_pad),
            w_gate=_pad_dense_tail_tokens(w_gate, T_pad),
            scale=scale,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            chunk_indices=chunk_indices,
        )
        return tuple(x[:, :T].contiguous() if x is not None else None for x in outputs)
    BC = SUB_CHUNK_SIZE
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None

    # Keep the WY factors in fp32. Rounding A and u to bf16 before A @ u
    # costs about 1.1% RMS on the gated bf16 inputs this op is tested with.
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Akkd = torch.zeros(B, T, H, BC, device=k.device, dtype=torch.float32)
    Akkx = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)

    sync_stream = torch.npu.current_stream(k.device)
    use_head_major_intra = not is_varlen
    # serialize split packed launches as well as dense launches to avoid CANN queue stalls.
    sync_each_launch = use_head_major_intra or NT * NC * B * H > LAUNCH_BLOCK_BUDGET
    bk_intra = triton.next_power_of_2(K)
    mask_grouped = time_axis_needs_mask(T, BT, cu_seqlens) or K % bk_intra != 0
    g_diag = g_raw if g_raw is not None else gk
    if use_head_major_intra:
        q_intra = q.transpose(1, 2).contiguous()
        k_intra = k.transpose(1, 2).contiguous()
        g_intra = g_diag.transpose(1, 2).contiguous()
        b_intra = b.transpose(1, 2).contiguous()
    else:
        q_intra, k_intra, g_intra, b_intra = q, k, g_diag, b
    for row_group in range(BC // TOKEN_GROUP_SIZE):
        launch_diag_kernel(
            chunk_gdn2_fwd_kernel_intra_grouped_npu,
            nt=NT,
            nc=NC,
            bh_total=B * H,
            sync_stream=sync_stream if sync_each_launch else None,
            kernel_kwargs=dict(
                q=q_intra,
                k=k_intra,
                g=g_intra,
                b=b_intra,
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
                BK=triton.next_power_of_2(K),
                BR=TOKEN_GROUP_SIZE,
                ROW_GROUP=row_group,
                IS_HEAD_MAJOR=use_head_major_intra,
                USE_RAW_G=g_raw is not None,
                IS_VARLEN=is_varlen,
                MASK_DMA=mask_grouped,
                NT_OFFSET=0,
                NC_OFFSET=0,
                BH_OFFSET=0,
            ),
        )
        if not sync_each_launch and row_group == 0:
            # CANN can stall when the two grouped row kernels are queued together.
            sync_stream.synchronize()

    launch_diag_kernel(
        chunk_gdn2_fwd_kernel_diag_solve_npu,
        nt=NT,
        nc=NC,
        bh_total=B * H,
        kernel_kwargs=dict(
            Akkd=Akkd,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            BC=BC,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            NC_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    inter_bk = _get_inter_bk(K)
    mask_inter = need_dma_mask(T, BC, K, inter_bk, K, inter_bk, is_varlen, cu_seqlens)
    product_kwargs = dict(
        q=q_intra if use_head_major_intra else q,
        k=k_intra if use_head_major_intra else k,
        # with raw gates the product kernel reads `g_raw` and ignores `g`
        g=g_intra if use_head_major_intra else gk,
        b=b_intra if use_head_major_intra else b,
        g_raw=g_raw,
        USE_RAW_G=g_raw is not None,
        Aqk=Aqk,
        Akkx=Akkx,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=inter_bk,
        IS_HEAD_MAJOR=use_head_major_intra,
        IS_VARLEN=is_varlen,
        MASK_DMA=mask_inter,
        NT_OFFSET=0,
        BH_OFFSET=0,
    )
    for dst_block in range(1, NC):
        for src_block in range(dst_block):
            launch_inter_kernel(
                chunk_gdn2_fwd_kernel_inter_products_npu,
                nt=NT,
                bh_total=B * H,
                kernel_kwargs=dict(product_kwargs, DST_BLOCK=dst_block, SRC_BLOCK=src_block),
            )
    launch_inter_kernel(
        chunk_gdn2_fwd_kernel_inter_solve_npu,
        nt=NT,
        bh_total=B * H,
        kernel_kwargs=dict(
            Akkd=Akkd,
            Akkx=Akkx,
            Akk=Akk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            BC=BC,
            NC=NC,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    w, u, qg, kg = recompute_w_u_fwd_gdn2(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        g_raw=g_raw,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # The stream-aware allocator keeps scratch storage alive for queued consumers.
    return w, u, qg, kg, Aqk, Akk
