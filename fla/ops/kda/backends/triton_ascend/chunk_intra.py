# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA chunk intra kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_kda_npu as _recompute_w_u_fwd_npu
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_BC = 16
_NUM_WARPS_SUB = 2
_NUM_WARPS_INTER = 2
_SUB_CHUNK_MEM_MULT = 6.0
_INTER_MEM_MULT = 14.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_MAX_INTER_BK = 64
# limit programs per launch to stay within Ascend AICore task time.
_KDA_LAUNCH_BLOCK_BUDGET = 4096


def _get_sub_chunk_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _SUB_CHUNK_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=triton.next_power_of_2(K),
    )


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


def _recompute_w_u_fwd(*args, **kwargs):
    return _recompute_w_u_fwd_npu(*args, **kwargs)


def _launch_sub_chunk_kernel(
    kernel,
    *,
    nt: int,
    nc: int,
    bh_total: int,
    kernel_kwargs: dict,
) -> None:
    budget = _KDA_LAUNCH_BLOCK_BUDGET
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
        nc_budget = max(1, budget // max(nt_len * bh_total, 1))
        max_nc = min(
            nc_budget,
            max_grid_axis_chunks(nc, nt_len * bh_total, max_grid=ASCEND_MAX_GRID_DIM),
        )
        for nc_off in range(0, nc, max_nc):
            nc_len = min(max_nc, nc - nc_off)
            kernel_kwargs['NC_OFFSET'] = nc_off
            bh_budget = max(1, budget // max(nt_len * nc_len, 1))
            max_bh = min(
                bh_budget,
                max_grid_axis_chunks(bh_total, nt_len * nc_len, max_grid=ASCEND_MAX_GRID_DIM),
            )
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nt_len, nc_len, bh_len)](num_warps=_NUM_WARPS_SUB, **kernel_kwargs)


def _launch_inter_kernel(
    kernel,
    *,
    nt: int,
    bh_total: int,
    kernel_kwargs: dict,
) -> None:
    budget = _KDA_LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nt_step = nt if nt * bh_total <= budget else max(1, min(nt, budget // max(bh_total, 1)))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        bh_budget = max(1, budget // max(nt_len, 1))
        max_bh = min(
            bh_budget,
            max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM),
        )
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS_INTER, **kernel_kwargs)


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_kda_fwd_kernel_diag_solve_npu(
    Akkd,
    cu_seqlens,
    chunk_indices,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    """Per-subchunk lower-triangular forward substitution into Akkd.

    Run before inter_solve so the fused inter kernel only merges off-diagonal
    blocks, keeping scalar BC loops off the large (NT, BH) grid.
    """
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    Akkd = Akkd + (bos * HV + i_hv) * BC
    o_i = tl.arange(0, BC)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    p_Akk = tl.make_block_ptr(Akkd, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    b_Akk = tl.load(p_Akk, boundary_check=(0, 1)).to(tl.float32)
    b_Ai = -tl.where(m_A, b_Akk, 0)
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akkd + (i_ti + i) * HV * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akkd.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_kda_fwd_kernel_intra_sub_chunk_npu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    NC_OFFSET,
    BH_OFFSET,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

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
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + tl.arange(0, BK)
    b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
    b_gn = b_gn[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt, allow_tf32=False) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_kda_fwd_kernel_inter_solve_fused_npu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET,
    BH_OFFSET,
):
    # Diagonal Akkd blocks are inverted by diag_solve before this kernel.
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

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
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

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

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        # Ascend cannot compile dynamic `if i_tc* < T` around dots (scf.if shape mismatch);
        # block_ptr uses boundary_check, and bare g loads mask out-of-range rows.
        p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        p_g1 = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
        b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
        b_gn1 = tl.load(g + i_tc1 * HV * K + o_k, mask=m_k & (i_tc1 < T), other=0).to(tl.float32)
        b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
        b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
        b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt, allow_tf32=False)
        b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt, allow_tf32=False)

        if NC >= 3:
            p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
            b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            b_gn2 = tl.load(g + i_tc2 * HV * K + o_k, mask=m_k & (i_tc2 < T), other=0).to(tl.float32)
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
            b_Aqk20 += tl.dot(b_qg2, b_kgt, allow_tf32=False)
            b_Akk20 += tl.dot(b_kg2, b_kgt, allow_tf32=False)
            b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
            b_Aqk21 += tl.dot(b_qg2, b_kgt, allow_tf32=False)
            b_Akk21 += tl.dot(b_kg2, b_kgt, allow_tf32=False)

            if NC >= 4:
                p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_g3 = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                b_gn3 = tl.load(g + i_tc3 * HV * K + o_k, mask=m_k & (i_tc3 < T), other=0).to(tl.float32)
                b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                b_qg3 = b_q3 * b_gqn3
                b_kg3 = b_k3 * b_gqn3
                b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                b_Aqk30 += tl.dot(b_qg3, b_kgt, allow_tf32=False)
                b_Akk30 += tl.dot(b_kg3, b_kgt, allow_tf32=False)
                b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                b_Aqk31 += tl.dot(b_qg3, b_kgt, allow_tf32=False)
                b_Akk31 += tl.dot(b_kg3, b_kgt, allow_tf32=False)
                b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                b_Aqk32 += tl.dot(b_qg3, b_kgt, allow_tf32=False)
                b_Akk32 += tl.dot(b_kg3, b_kgt, allow_tf32=False)

    p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    p_b1 = tl.make_block_ptr(beta + bos * HV + i_hv, (T,), (HV,), (i_tc1,), (BC,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + bos * HV + i_hv, (T,), (HV,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if NC >= 4:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b3 = tl.make_block_ptr(beta + bos * HV + i_hv, (T,), (HV,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (HV * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (HV * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 3:
        p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (HV * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    if NC >= 4:
        p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (HV * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, allow_tf32=False),
        b_Ai00,
        allow_tf32=False,
    )

    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, allow_tf32=False),
            b_Ai11,
            allow_tf32=False,
        )
        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_Akk20, b_Ai00, allow_tf32=False) +
            tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
            allow_tf32=False,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, allow_tf32=False),
            b_Ai22,
            allow_tf32=False,
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk31, b_Ai11, allow_tf32=False) +
            tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
            allow_tf32=False,
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk30, b_Ai00, allow_tf32=False) +
            tl.dot(b_Akk31, b_Ai10, allow_tf32=False) +
            tl.dot(b_Akk32, b_Ai20, allow_tf32=False),
            allow_tf32=False,
        )

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 3:
        p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 4:
        p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_kda_fwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    BC = _BC
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    is_varlen = cu_seqlens is not None

    Aqk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.zeros(B, T, HV, BC, device=k.device, dtype=torch.float32)

    if safe_gate:
        sub_bk = _get_sub_chunk_bk(K)
        _launch_sub_chunk_kernel(
            chunk_kda_fwd_kernel_intra_sub_chunk_npu,
            nt=NT,
            nc=NC,
            bh_total=B * HV,
            kernel_kwargs=dict(
                q=q,
                k=k,
                g=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                H=H,
                HV=HV,
                K=K,
                BT=BT,
                BC=BC,
                BK=sub_bk,
                IS_VARLEN=is_varlen,
                NT_OFFSET=0,
                NC_OFFSET=0,
                BH_OFFSET=0,
            ),
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Invert diagonal Akkd blocks first; inter then only merges off-diagonals.
    _launch_sub_chunk_kernel(
        chunk_kda_fwd_kernel_diag_solve_npu,
        nt=NT,
        nc=NC,
        bh_total=B * HV,
        kernel_kwargs=dict(
            Akkd=Akkd,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            HV=HV,
            BT=BT,
            BC=BC,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            NC_OFFSET=0,
            BH_OFFSET=0,
        ),
    )

    inter_bk = _get_inter_bk(K)
    _launch_inter_kernel(
        chunk_kda_fwd_kernel_inter_solve_fused_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akkd=Akkd,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            NC=NC,
            BK=inter_bk,
            IS_VARLEN=is_varlen,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    w, u, qg, kg = _recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


_NUM_WARPS_BWD = 2
# Vectorized diagonal path keeps BC×BK tiles live; keep BK small for Ascend UB.
_BWD_INTRA_MEM_MULT = 18.0
_FALLBACK_BK = 16
_MAX_BK = 16
# limit programs per launch to stay within Ascend AICore task time.
_KDA_BWD_INTRA_LAUNCH_BLOCK_BUDGET = 4096


def _get_bwd_intra_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _BWD_INTRA_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_BK, triton.next_power_of_2(K)),
    )


def _launch_bwd_intra_kernel(
    kernel,
    *,
    nk_nc: int,
    nt: int,
    bh_total: int,
    kernel_kwargs: dict,
) -> None:
    # limit programs per launch to stay within Ascend AICore task time.
    budget = _KDA_BWD_INTRA_LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nk_step = nk_nc if nk_nc * nt * bh_total <= budget else max(1, budget // max(nt * bh_total, 1))
    for nknc_off in range(0, nk_nc, nk_step):
        nknc_len = min(nk_step, nk_nc - nknc_off)
        kernel_kwargs['NKNC_OFFSET'] = nknc_off
        nt_budget = max(1, budget // max(nknc_len * bh_total, 1))
        max_nt = min(
            nt_budget,
            max_grid_axis_chunks(nt, nknc_len * bh_total, max_grid=ASCEND_MAX_GRID_DIM),
        )
        for nt_off in range(0, nt, max_nt):
            nt_len = min(max_nt, nt - nt_off)
            if cu_seqlens is not None and chunk_indices is not None:
                kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
                kernel_kwargs['NT_OFFSET'] = 0
            else:
                kernel_kwargs['NT_OFFSET'] = nt_off
            bh_budget = max(1, budget // max(nknc_len * nt_len, 1))
            max_bh = min(
                bh_budget,
                max_grid_axis_chunks(bh_total, nknc_len * nt_len, max_grid=ASCEND_MAX_GRID_DIM),
            )
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nknc_len, nt_len, bh_len)](num_warps=_NUM_WARPS_BWD, **kernel_kwargs)


@triton.jit(do_not_specialize=['B', 'T', 'NKNC_OFFSET', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_kda_bwd_kernel_intra_npu(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    NKNC_OFFSET,
    NT_OFFSET,
    BH_OFFSET,
):
    i_kc = tl.program_id(0) + NKNC_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += (i_k * all + bos) * HV + i_hv

    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (HV * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (HV * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn - b_gk)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            b_dq2 += tl.dot(b_dAqk, b_kg, allow_tf32=False)
            b_dk2 += tl.dot(b_dAkk, b_kg, allow_tf32=False)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * HV * BT + i_i * BC
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * HV * K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    if SAFE_GATE:
        # Midpoint-offset vectorized path (bounded under lower_bound=-5).
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk, allow_tf32=False) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk, allow_tf32=False) * exp_b_g_diag_qk
    else:
        # Pairwise scalar path required for unbounded non-safe gates (avoids Inf/NaN).
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H * K
            p_gkj += HV * K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (HV,), (i_ti,), (BC,), (0,))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    # synchronize before the second half of the kernel that reuses the same tiles.
    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, HV * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, HV * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            b_b = tl.load(p_b, boundary_check=(0,))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_kb = tl.load(p_k, boundary_check=(0, 1)) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            b_dkt += tl.dot(b_dAqk, b_qg, allow_tf32=False)
            b_dkt += tl.dot(b_dAkk, b_kbg, allow_tf32=False)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * HV * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * HV * K + o_k
    p_bj = beta + i_ti * HV

    if SAFE_GATE:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
        b_b = tl.load(p_b, boundary_check=(0,))

        p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, HV * BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, HV * BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b[:, None] * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp, allow_tf32=False) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp, allow_tf32=False) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j * HV * BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV * BT)
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H * K
            p_kj += H * K
            p_gkj += HV * K
            p_bj += HV

    p_dk = tl.make_block_ptr(dk, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_kda_bwd_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BC = min(_BC, BT)
    BK = _get_bwd_intra_bk(K)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    is_varlen = cu_seqlens is not None

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)

    _launch_bwd_intra_kernel(
        chunk_kda_bwd_kernel_intra_npu,
        nk_nc=NK * NC,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=g,
            beta=beta,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dq2=dq2,
            dk=dk,
            dk2=dk2,
            dg=dg,
            dg2=dg2,
            db=db2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            NC=NC,
            IS_VARLEN=is_varlen,
            SAFE_GATE=safe_gate,
            NKNC_OFFSET=0,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = dg2

    return dq, dk, db, dg
