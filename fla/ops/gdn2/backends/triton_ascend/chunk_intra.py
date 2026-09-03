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

from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, compute_row_tile_block_size, max_grid_axis_chunks

_BC = 16
_TOKEN_GROUP = 8
_INTER_MEM_MULT = 18.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_MAX_INTER_BK = 64
_LAUNCH_BLOCK_BUDGET = 4096
# Disable auto-multi-buffer and AutoBlockify on the inter kernels for CANN 9.1.
_INTER_COMPILE_KWARGS = ascend_compile_kwargs(blacklist_auto_blockify=True)


def _get_inter_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _INTER_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_INTER_BK, triton.next_power_of_2(K)),
    )


def _launch_diag_kernel(
    kernel,
    *,
    nt: int,
    nc: int,
    bh_total: int,
    kernel_kwargs: dict,
    sync_stream=None,
) -> None:
    budget = _LAUNCH_BLOCK_BUDGET
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
        max_nc = max_grid_axis_chunks(nc, nt_len * bh_total, max_grid=ASCEND_MAX_GRID_DIM)
        for nc_off in range(0, nc, max_nc):
            nc_len = min(max_nc, nc - nc_off)
            kernel_kwargs['NC_OFFSET'] = nc_off
            max_bh = max_grid_axis_chunks(bh_total, nt_len * nc_len, max_grid=ASCEND_MAX_GRID_DIM)
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nt_len, nc_len, bh_len)](**kernel_kwargs)
                if sync_stream is not None:
                    sync_stream.synchronize()


def _launch_inter_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    budget = _LAUNCH_BLOCK_BUDGET
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
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](**kernel_kwargs, **_INTER_COMPILE_KWARGS)


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
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    """Build one group of causal Aqk/Akk rows without unstable gate factoring."""
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1,
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    i_ts = i_t * BT + i_i * BC
    i_ti = i_ts + ROW_GROUP * BR
    if i_ti >= T:
        return

    o_r = tl.arange(0, BR)
    o_k = tl.arange(0, BK)
    o_c = i_ti + o_r
    m_r = o_c < T
    m_k = o_k < K
    m_rk = m_r[:, None] & m_k[None, :]

    if IS_HEAD_MAJOR:
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

    p_q = q + o_c[:, None] * input_stride + o_k[None, :]
    p_k = k + o_c[:, None] * input_stride + o_k[None, :]
    p_g = g + o_c[:, None] * input_stride + o_k[None, :]
    p_b = b + o_c[:, None] * input_stride + o_k[None, :]
    b_q = tl.load(p_q, mask=m_rk, other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=m_rk, other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=m_rk, other=0.0).to(tl.float32)
    b_b = tl.load(p_b, mask=m_rk, other=0.0).to(tl.float32)
    b_k *= b_b

    for j in range(0, (ROW_GROUP + 1) * BR):
        i_j = i_ts + j
        m_j = i_j < T
        p_kj = k + i_j * input_stride + o_k
        p_gj = g + i_j * input_stride + o_k
        b_kj = tl.load(p_kj, mask=m_j & m_k, other=0.0).to(tl.float32)
        b_gj = tl.load(p_gj, mask=m_j & m_k, other=0.0).to(tl.float32)
        b_kgj = tl.where(m_k[None, :], b_kj[None, :] * exp2(b_g - b_gj[None, :]), 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1)
        row_in_subchunk = ROW_GROUP * BR + o_r
        tl.store(
            Aqk + o_c * (H * BT) + i_i * BC + j,
            b_Aqk.to(Aqk.dtype.element_ty),
            mask=m_r & m_j & (j <= row_in_subchunk),
        )
        tl.store(
            Akk + o_c * (H * BC) + j,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_r & m_j & (j < row_in_subchunk),
        )


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
    p_Akk = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    b_Akk = tl.load(p_Akk, boundary_check=(0, 1)).to(tl.float32)
    b_Ai = -tl.where(m_A, b_Akk, 0)
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akkd + (i_ti + i).to(tl.int64) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akkd.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gdn2_fwd_kernel_inter_products_npu(
    q,
    k,
    g,
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

    i_tc_dst = i_t * BT + DST_BLOCK * BC
    i_tc_src = i_t * BT + SRC_BLOCK * BC

    base = bos * H + i_h
    q += base * K
    k += base * K
    g += base * K
    b += base * K
    Aqk += base * BT
    Akkx += base * BT

    o_i = tl.arange(0, BC)
    m_dst = (i_tc_dst + o_i) < T
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k_src = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_src, i_k * BK), (BC, BK), (1, 0))
        p_g_src = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc_src, i_k * BK), (BC, BK), (1, 0))
        b_k_src = tl.load(p_k_src, boundary_check=(0, 1)).to(tl.float32)
        b_g_src = tl.load(p_g_src, boundary_check=(0, 1)).to(tl.float32)

        p_q_dst = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_dst, i_k * BK), (BC, BK), (1, 0))
        p_k_dst = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_dst, i_k * BK), (BC, BK), (1, 0))
        p_g_dst = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc_dst, i_k * BK), (BC, BK), (1, 0))
        p_b_dst = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc_dst, i_k * BK), (BC, BK), (1, 0))
        b_q_dst = tl.load(p_q_dst, boundary_check=(0, 1)).to(tl.float32)
        b_k_dst = tl.load(p_k_dst, boundary_check=(0, 1)).to(tl.float32)
        b_g_dst = tl.load(p_g_dst, boundary_check=(0, 1)).to(tl.float32)
        b_b_dst = tl.load(p_b_dst, boundary_check=(0, 1)).to(tl.float32)
        b_gn_dst = tl.load(
            g + tl.cast(i_tc_dst, tl.int64) * H * K + o_k,
            mask=m_k & (i_tc_dst < T),
            other=0.0,
        ).to(tl.float32)
        b_gq = tl.where(m_dst[:, None], exp2(b_g_dst - b_gn_dst[None, :]), 0.0)
        b_kgt = tl.trans(b_k_src * exp2(b_gn_dst[None, :] - b_g_src))
        b_Aqk += tl.dot(b_q_dst * b_gq, b_kgt, allow_tf32=False)
        b_Akk += tl.dot((b_b_dst * b_k_dst) * b_gq, b_kgt, allow_tf32=False)

    p_Aqk = tl.make_block_ptr(
        Aqk,
        (T, BT),
        (H * BT, 1),
        (i_tc_dst, SRC_BLOCK * BC),
        (BC, BC),
        (1, 0),
    )
    p_Akkx = tl.make_block_ptr(
        Akkx,
        (T, BT),
        (H * BT, 1),
        (i_tc_dst, SRC_BLOCK * BC),
        (BC, BC),
        (1, 0),
    )
    tl.store(p_Aqk, (b_Aqk * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akkx, b_Akk, boundary_check=(0, 1))


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

    p_Akkx10 = tl.make_block_ptr(Akkx, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    b_Akk10 = tl.load(p_Akkx10, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 3:
        p_Akkx20 = tl.make_block_ptr(Akkx, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akkx21 = tl.make_block_ptr(Akkx, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        b_Akk20 = tl.load(p_Akkx20, boundary_check=(0, 1)).to(tl.float32)
        b_Akk21 = tl.load(p_Akkx21, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 4:
        p_Akkx30 = tl.make_block_ptr(Akkx, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Akkx31 = tl.make_block_ptr(Akkx, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Akkx32 = tl.make_block_ptr(Akkx, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        b_Akk30 = tl.load(p_Akkx30, boundary_check=(0, 1)).to(tl.float32)
        b_Akk31 = tl.load(p_Akkx31, boundary_check=(0, 1)).to(tl.float32)
        b_Akk32 = tl.load(p_Akkx32, boundary_check=(0, 1)).to(tl.float32)

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 3:
        p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 4:
        p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    # tl.dot may clobber its lhs on Ascend; materialize copies for later dot and store uses.
    b_Ai11_c = b_Ai11 + 0.0
    if NC >= 3:
        b_Ai22_c = b_Ai22 + 0.0
        b_Ai22_c2 = b_Ai22 + 0.0
        b_Ai22_c3 = b_Ai22 + 0.0
    if NC >= 4:
        b_Ai33_c = b_Ai33 + 0.0
        b_Ai33_c2 = b_Ai33 + 0.0
        b_Ai33_c3 = b_Ai33 + 0.0
        b_Akk31_c = b_Akk31 + 0.0
        b_Akk32_c = b_Akk32 + 0.0

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10, allow_tf32=False), b_Ai00, allow_tf32=False)
    if NC >= 3:
        b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_Akk21, allow_tf32=False), b_Ai11_c, allow_tf32=False)
        b_Ai20 = -tl.dot(
            b_Ai22_c2,
            tl.dot(b_Akk20, b_Ai00, allow_tf32=False) + tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
            allow_tf32=False,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_Akk32, allow_tf32=False), b_Ai22_c3, allow_tf32=False)
        b_Ai31 = -tl.dot(
            b_Ai33_c2,
            tl.dot(b_Akk31, b_Ai11_c, allow_tf32=False) + tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
            allow_tf32=False,
        )
        b_Ai30 = -tl.dot(
            b_Ai33_c3,
            tl.dot(b_Akk30, b_Ai00, allow_tf32=False)
            + tl.dot(b_Akk31_c, b_Ai10, allow_tf32=False)
            + tl.dot(b_Akk32_c, b_Ai20, allow_tf32=False),
            allow_tf32=False,
        )

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 3:
        p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 4:
        p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))


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
    disable_recompute: bool = False,
):
    # gk is already activated and accumulated; pairwise gate differences serve both gate modes.
    del safe_gate

    B, T, H, K = k.shape
    BT = chunk_size
    BC = _BC
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None

    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.zeros(B, T, H, BC, device=k.device, dtype=torch.float32)
    Akkx = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)

    sync_stream = torch.npu.current_stream(k.device)
    use_head_major_intra = not is_varlen
    # serialize split dense launches to avoid CANN queue stalls.
    dense_sync_stream = sync_stream if use_head_major_intra else None
    if use_head_major_intra:
        q_intra = q.transpose(1, 2).contiguous()
        k_intra = k.transpose(1, 2).contiguous()
        g_intra = gk.transpose(1, 2).contiguous()
        b_intra = b.transpose(1, 2).contiguous()
    else:
        q_intra, k_intra, g_intra, b_intra = q, k, gk, b
    for row_group in range(BC // _TOKEN_GROUP):
        _launch_diag_kernel(
            chunk_gdn2_fwd_kernel_intra_grouped_npu,
            nt=NT,
            nc=NC,
            bh_total=B * H,
            sync_stream=dense_sync_stream,
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
                BR=_TOKEN_GROUP,
                ROW_GROUP=row_group,
                IS_HEAD_MAJOR=use_head_major_intra,
                IS_VARLEN=is_varlen,
                NT_OFFSET=0,
                NC_OFFSET=0,
                BH_OFFSET=0,
            ),
        )
        if not use_head_major_intra and row_group == 0:
            # CANN can stall when the two grouped row kernels are queued together.
            sync_stream.synchronize()

    _launch_diag_kernel(
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
    product_kwargs = dict(
        q=q,
        k=k,
        g=gk,
        b=b,
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
        BK=_get_inter_bk(K),
        IS_VARLEN=is_varlen,
        NT_OFFSET=0,
        BH_OFFSET=0,
    )
    for dst_block in range(1, NC):
        for src_block in range(dst_block):
            _launch_inter_kernel(
                chunk_gdn2_fwd_kernel_inter_products_npu,
                nt=NT,
                bh_total=B * H,
                kernel_kwargs=dict(product_kwargs, DST_BLOCK=dst_block, SRC_BLOCK=src_block),
            )
    _launch_inter_kernel(
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
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # Keep intra scratch alive until its last asynchronous consumer completes.
    sync_stream.synchronize()
    return w, u, qg, kg, Aqk, Akk
