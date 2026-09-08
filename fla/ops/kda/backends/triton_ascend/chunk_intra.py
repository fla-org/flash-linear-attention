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
from triton.runtime import driver

from fla.ops.kda.backends.triton_ascend.wy_fast import recompute_w_u_fwd_kda_npu as _recompute_w_u_fwd_npu
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
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


# disable auto-multi-buffer and AutoBlockify on the fused inter launch
_INTER_COMPILE_KWARGS = ascend_compile_kwargs(blacklist_auto_blockify=True)


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
            kernel[(nt_len, bh_len)](
                num_warps=_NUM_WARPS_INTER,
                **kernel_kwargs,
                **_INTER_COMPILE_KWARGS,
            )


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
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

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
        b_a = -tl.load(Akkd + (i_ti + i).to(tl.int64) * HV * BC + o_i)
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
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + (bos * HV + i_hv)
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

    p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)).to(tl.int64) * HV * K + tl.arange(0, BK)
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
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

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
        b_gn1 = tl.load(g + i_tc1.to(tl.int64) * HV * K + o_k, mask=m_k & (i_tc1 < T), other=0).to(tl.float32)
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
            b_gn2 = tl.load(g + i_tc2.to(tl.int64) * HV * K + o_k, mask=m_k & (i_tc2 < T), other=0).to(tl.float32)
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            b_qg2_c = b_qg2 + 0.0
            b_kg2_c = b_kg2 + 0.0
            b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
            b_Aqk20 += tl.dot(b_qg2, b_kgt, allow_tf32=False)
            b_Akk20 += tl.dot(b_kg2, b_kgt, allow_tf32=False)
            b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
            b_Aqk21 += tl.dot(b_qg2_c, b_kgt, allow_tf32=False)
            b_Akk21 += tl.dot(b_kg2_c, b_kgt, allow_tf32=False)

            if NC >= 4:
                p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                p_g3 = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                b_gn3 = tl.load(g + i_tc3.to(tl.int64) * HV * K + o_k, mask=m_k & (i_tc3 < T), other=0).to(tl.float32)
                b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                b_qg3 = b_q3 * b_gqn3
                b_kg3 = b_k3 * b_gqn3
                b_qg3_c1 = b_qg3 + 0.0
                b_kg3_c1 = b_kg3 + 0.0
                b_qg3_c2 = b_qg3 + 0.0
                b_kg3_c2 = b_kg3 + 0.0
                b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                b_Aqk30 += tl.dot(b_qg3, b_kgt, allow_tf32=False)
                b_Akk30 += tl.dot(b_kg3, b_kgt, allow_tf32=False)
                b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                b_Aqk31 += tl.dot(b_qg3_c1, b_kgt, allow_tf32=False)
                b_Akk31 += tl.dot(b_kg3_c1, b_kgt, allow_tf32=False)
                b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                b_Aqk32 += tl.dot(b_qg3_c2, b_kgt, allow_tf32=False)
                b_Akk32 += tl.dot(b_kg3_c2, b_kgt, allow_tf32=False)

    p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    p_b1 = tl.make_block_ptr(beta + (bos * HV + i_hv), (T,), (HV,), (i_tc1,), (BC,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + (bos * HV + i_hv), (T,), (HV,), (i_tc2,), (BC,), (0,))
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

        p_b3 = tl.make_block_ptr(beta + (bos * HV + i_hv), (T,), (HV,), (i_tc3,), (BC,), (0,))
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

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, allow_tf32=False),
        b_Ai00,
        allow_tf32=False,
    )

    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, allow_tf32=False),
            b_Ai11_c,
            allow_tf32=False,
        )
        b_Ai20 = -tl.dot(
            b_Ai22_c2,
            tl.dot(b_Akk20, b_Ai00, allow_tf32=False) +
            tl.dot(b_Akk21, b_Ai10, allow_tf32=False),
            allow_tf32=False,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, allow_tf32=False),
            b_Ai22_c3,
            allow_tf32=False,
        )
        b_Ai31 = -tl.dot(
            b_Ai33_c2,
            tl.dot(b_Akk31, b_Ai11_c, allow_tf32=False) +
            tl.dot(b_Akk32, b_Ai21, allow_tf32=False),
            allow_tf32=False,
        )
        b_Ai30 = -tl.dot(
            b_Ai33_c3,
            tl.dot(b_Akk30, b_Ai00, allow_tf32=False) +
            tl.dot(b_Akk31_c, b_Ai10, allow_tf32=False) +
            tl.dot(b_Akk32_c, b_Ai20, allow_tf32=False),
            allow_tf32=False,
        )

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 3:
        p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    if NC >= 4:
        p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33_c.to(Akk.dtype.element_ty), boundary_check=(0, 1))


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
    use_graph: bool = False,
):
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
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


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit(do_not_specialize=['B', 'T', 'NT_TOTAL'])
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
    SAFE_GATE: tl.constexpr,
    NT_TOTAL,
):
    NC = tl.cdiv(BT, BC)
    core_id = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    task_num = tl.cast(NT_TOTAL, tl.int64) * NC * B * HV
    BH_TOTAL = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT_TOTAL
        i_i0 = rem // NT_TOTAL
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if cu_seqlens is not None:
            i_n, i_t = tl.load(chunk_indices + i_t * 2), tl.load(chunk_indices + i_t * 2 + 1)
            # int64 guarantees: cu_seqlens may arrive as int32 and chunk_indices
            # follows its dtype, but the global offsets must stay 64-bit
            bos, eos = tl.cast(tl.load(cu_seqlens + i_n), tl.int64), tl.cast(tl.load(cu_seqlens + i_n + 1), tl.int64)
        else:
            bos, eos = i_b * T, i_b * T + T
        # T is a loop-carried arg (int32); the reassignment must keep its type
        T = tl.cast(eos - bos, tl.int32)

        # rebind pointers per task (ptr-arg += inside the task loop would both
        # accumulate offsets across tasks and break loop-carried typecheck)
        off_h = (bos * H + i_h) * K
        off_hv = bos * HV + i_hv
        q_l = q + off_h
        k_l = k + off_h
        g_l = g + off_hv * K
        beta_l = beta + off_hv
        dAqk_l = dAqk + off_hv * BT
        dAkk_l = dAkk + off_hv * BT
        dq_l = dq + off_hv * K
        dq2_l = dq2 + off_hv * K
        dk_l = dk + off_hv * K
        dk2_l = dk2 + off_hv * K
        dg_l = dg + off_hv * K
        dg2_l = dg2 + off_hv * K
        db_l = db + off_hv

        o_k = tl.arange(0, BK)
        o_i = tl.arange(0, BC)
        m_k = o_k < K
        NC_LOC = min(NC, tl.cdiv(T - i_t * BT, BC))
        i_i = i_i0
        if i_i0 < NC_LOC:
            # no-op cast when i_t is int64; guards int32 chunk_indices tables
            i_ti = tl.cast(i_t * BT + i_i * BC, tl.int64)
            m_row = (i_ti + o_i) < T
            m_ik = m_row[:, None] & m_k[None, :]
            a_row = i_i * BC + o_i

            b_g = tl.load(g_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_ik, other=0.0).to(tl.float32)
            b_b = tl.load(beta_l + i_ti * HV + o_i * HV, mask=m_row, other=0.0)
            b_q = tl.load(q_l + i_ti * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ik, other=0.0)
            b_k = tl.load(k_l + i_ti * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ik, other=0.0)

            b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
            b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

            # ---- inter blocks (j < i) ----
            if i_i > 0:
                b_gn = tl.load(g_l + i_ti * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                for i_j in range(0, i_i):
                    row_j = tl.cast(i_t * BT + i_j * BC, tl.int64)
                    m_rowj = (row_j + o_i) < T
                    m_ikj = m_rowj[:, None] & m_k[None, :]
                    b_kj = tl.load(k_l + row_j * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ikj, other=0.0)
                    b_gkj = tl.load(g_l + row_j * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_ikj, other=0.0)
                    b_kg = b_kj * exp2(b_gn - b_gkj.to(tl.float32))
                    m_ij = m_row[:, None] & ((i_j * BC + o_i)[None, :] < BT)
                    b_dAqk = tl.load(dAqk_l + i_ti * (HV * BT) + o_i[:, None] * (HV * BT) +
                                     (i_j * BC + o_i)[None, :], mask=m_ij, other=0.0)
                    b_dAkk = tl.load(dAkk_l + i_ti * (HV * BT) + o_i[:, None] * (HV * BT) +
                                     (i_j * BC + o_i)[None, :], mask=m_ij, other=0.0)
                    b_dq2 += tl.dot(b_dAqk.to(tl.float32), b_kg.to(tl.float32), allow_tf32=False)
                    b_dk2 += tl.dot(b_dAkk.to(tl.float32), b_kg.to(tl.float32), allow_tf32=False)
                b_gqn = exp2(b_g - b_gn)
                b_dq2 *= b_gqn
                b_dk2 *= b_gqn

            # ---- diagonal (SAFE_GATE midpoint path) ----
            if SAFE_GATE:
                i_gm = i_ti + min(BC // 2, T - i_ti - 1)
                b_gm = tl.load(g_l + i_gm * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                m_ij_d = m_row[:, None] & ((i_i * BC + o_i)[None, :] < BT)
                b_dAqk_d = tl.load(dAqk_l + i_ti * (HV * BT) + o_i[:, None] * (HV * BT) +
                                   (i_i * BC + o_i)[None, :], mask=m_ij_d, other=0.0).to(tl.float32)
                b_dAkk_d = tl.load(dAkk_l + i_ti * (HV * BT) + o_i[:, None] * (HV * BT) +
                                   (i_i * BC + o_i)[None, :], mask=m_ij_d, other=0.0).to(tl.float32)
                m_i_d = (o_i[:, None] >= o_i[None, :]) & m_row[:, None] & m_row[None, :]
                b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.)
                b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.)
                b_g_d = tl.where(m_row[:, None], b_g - b_gm, 0.)
                exp_p = tl.where(m_row[:, None], exp2(b_g_d), 0.)
                exp_n = tl.where(m_row[:, None], exp2(-b_g_d), 0.)
                b_k_exp = b_k.to(tl.float32) * exp_n
                b_dq2 += tl.dot(b_dAqk_d, b_k_exp, allow_tf32=False) * exp_p
                b_dk2 += tl.dot(b_dAkk_d, b_k_exp, allow_tf32=False) * exp_p
            else:
                # pairwise per-column diag for unbounded (non-safe) gates
                o_dA_c = i_ti * (HV * BT) + o_i * (HV * BT) + i_i * BC
                for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                    b_dAqk_j = tl.load(dAqk_l + o_dA_c + j, mask=m_row, other=0.0)
                    b_dAkk_j = tl.load(dAkk_l + o_dA_c + j, mask=m_row, other=0.0)
                    b_kj = tl.load(k_l + i_ti * (H * K) + j * (H * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    b_gkj = tl.load(g_l + i_ti * (HV * K) + j * (HV * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    m_i = o_i[:, None] >= j
                    b_gqk = exp2(b_g - b_gkj[None, :])
                    b_dq2 += tl.where(m_i, b_dAqk_j[:, None] * b_kj[None, :] * b_gqk, 0.)
                    b_dk2 += tl.where(m_i, b_dAkk_j[:, None] * b_kj[None, :] * b_gqk, 0.)

            # ---- first-half outputs: dq2/db (past + diag contributions) ----
            b_db = tl.sum(b_dk2 * b_k.to(tl.float32), 1)
            b_dk2 = b_dk2 * b_b.to(tl.float32)[:, None]
            b_dg2 = b_q.to(tl.float32) * b_dq2
            b_dq2 += tl.load(dq_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                             o_k[None, :], mask=m_ik, other=0.0).to(tl.float32)
            tl.store(dq2_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                     o_k[None, :], b_dq2.to(dq2.dtype.element_ty), mask=m_ik)
            tl.store(db_l + i_ti * HV + o_i * HV, b_db.to(tl.float32), mask=m_row)

            # ---- future blocks (j > i) and diag-kk: dkt contribution ----
            b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
            if i_i < NC_LOC - 1:
                b_gn_f = tl.load(g_l + (min(i_ti + BC, T) - 1) * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                for i_j in range(i_i + 1, NC_LOC):
                    row_j = tl.cast(i_t * BT + i_j * BC, tl.int64)
                    m_rowj = (row_j + o_i) < T
                    m_ikj = m_rowj[:, None] & m_k[None, :]
                    b_qf = tl.load(q_l + row_j * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ikj, other=0.0)
                    b_kf = tl.load(k_l + row_j * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ikj, other=0.0)
                    b_gkf = tl.load(g_l + row_j * (HV * K) + o_i[:, None] * (HV * K) +
                                    o_k[None, :], mask=m_ikj, other=0.0).to(tl.float32)
                    b_bf = tl.load(beta_l + row_j * HV + o_i * HV, mask=m_rowj, other=0.0)
                    # transposed dA tiles: element (a, b) at dA + a + b*(HV*BT)
                    b_col = row_j + o_i
                    m_t = (b_col[None, :] < T) & (a_row[:, None] < BT)
                    b_dAqk_f = tl.load(dAqk_l + row_j * (HV * BT) +
                                       a_row[:, None] + o_i[None, :] * (HV * BT), mask=m_t, other=0.0)
                    b_dAkk_f = tl.load(dAkk_l + row_j * (HV * BT) +
                                       a_row[:, None] + o_i[None, :] * (HV * BT), mask=m_t, other=0.0)
                    b_gkn = exp2(b_gkf - b_gn_f)
                    b_qg = b_qf * tl.where(m_rowj[:, None], b_gkn, 0)
                    b_kbg = b_kf * b_bf.to(tl.float32)[:, None] * tl.where(m_rowj[:, None], b_gkn, 0)
                    b_dkt += tl.dot(b_dAqk_f.to(tl.float32), b_qg.to(tl.float32), allow_tf32=False)
                    b_dkt += tl.dot(b_dAkk_f.to(tl.float32), b_kbg.to(tl.float32), allow_tf32=False)
                b_dkt *= exp2(b_gn_f - b_g)

            if SAFE_GATE:
                i_gm = i_ti + min(BC // 2, T - i_ti - 1)
                b_gm = tl.load(g_l + i_gm * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                b_col = i_ti + o_i
                m_t = (b_col[None, :] < T)
                b_dAqk_t = tl.load(dAqk_l + i_ti * (HV * BT) + a_row[:, None] +
                                   o_i[None, :] * (HV * BT), mask=m_t, other=0.0).to(tl.float32)
                b_dAkk_t = tl.load(dAkk_l + i_ti * (HV * BT) + a_row[:, None] +
                                   o_i[None, :] * (HV * BT), mask=m_t, other=0.0).to(tl.float32)
                m_i_t = (o_i[:, None] <= o_i[None, :]) & m_row[:, None] & m_row[None, :]
                b_dAqk_t = tl.where(m_i_t, b_dAqk_t, 0.)
                b_dAkk_t = tl.where(m_i_t, b_dAkk_t, 0.)
                b_g_d = tl.where(m_row[:, None], b_g - b_gm, 0.)
                exp_p = tl.where(m_row[:, None], exp2(b_g_d), 0.)
                exp_n = tl.where(m_row[:, None], exp2(-b_g_d), 0.)
                b_q_exp = b_q.to(tl.float32) * exp_p
                b_kb_exp = b_k.to(tl.float32) * b_b.to(tl.float32)[:, None] * exp_p
                b_dkt += tl.dot(b_dAqk_t, b_q_exp, allow_tf32=False) * exp_n
                b_dkt += tl.dot(b_dAkk_t, b_kb_exp, allow_tf32=False) * exp_n
            else:
                # pairwise per-column diag (future side) for unbounded gates
                for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                    b_dAqk_j = tl.load(dAqk_l + i_ti * (HV * BT) + j * (HV * BT) + i_i * BC + o_i)
                    b_dAkk_j = tl.load(dAkk_l + i_ti * (HV * BT) + j * (HV * BT) + i_i * BC + o_i)
                    b_qj = tl.load(q_l + i_ti * (H * K) + j * (H * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    b_kbj = tl.load(k_l + i_ti * (H * K) + j * (H * K) + o_k, mask=m_k,
                                    other=0).to(tl.float32) * tl.load(beta_l + (i_ti + j) * HV)
                    b_gkj = tl.load(g_l + i_ti * (HV * K) + j * (HV * K) + o_k, mask=m_k, other=0).to(tl.float32)
                    m_i = o_i[:, None] <= j
                    b_gkq = exp2(b_gkj[None, :] - b_g)
                    b_dkt += tl.where(m_i, b_dAqk_j[:, None] * b_qj[None, :] * b_gkq, 0.)
                    b_dkt += tl.where(m_i, b_dAkk_j[:, None] * b_kbj[None, :] * b_gkq, 0.)

            # ---- second-half outputs: dk2/dg2 (adds future/dkt contributions) ----
            b_dg2 += (b_dk2 - b_dkt) * b_k.to(tl.float32) + tl.load(
                dg_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_ik, other=0.0)
            b_dk2 += tl.load(dk_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                             o_k[None, :], mask=m_ik, other=0.0).to(tl.float32)
            b_dk2 += b_dkt
            tl.store(dk2_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                     o_k[None, :], b_dk2.to(dk2.dtype.element_ty), mask=m_ik)
            tl.store(dg2_l + i_ti * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], b_dg2, mask=m_ik)


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
    use_graph: bool = False,
):
    if use_graph:
        raise NotImplementedError("use_graph is not supported on the Ascend NPU backend")
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)
    if (safe_gate and BK > 512) or (not safe_gate and BK > 256):
        # Outside the validated single-block UB envelope. The dispatch verifier
        # routes these calls to the mainline kernel already; this covers direct
        # callers of the backend function.
        from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as _mainline_bwd_intra
        # __wrapped__ skips re-dispatch back into this backend; bare fn when dispatch is disabled
        return getattr(_mainline_bwd_intra, '__wrapped__', _mainline_bwd_intra)(
            q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size, safe_gate=safe_gate)
    # The 192KB UB budget caps how many BCxBK fp32 tiles stay live, so BC must
    # follow the padded BK (non-pow2 K like 192 lands in the BK=256 tier).
    # SAFE_GATE dot path uses 32x32 cube tiles; the pairwise per-column diag
    # compiles only at BC<=16, and only at BC=8 once BK reaches 256.
    if safe_gate:
        BC = min(32, BT) if BK <= 256 else min(16, BT)
    else:
        BC = min(16, BT) if BK <= 128 else min(8, BT)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(1, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    num_core = get_npu_properties()['num_aicore']
    chunk_kda_bwd_kernel_intra_npu[(num_core,)](
        NT_TOTAL=NT,
        q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK,
        SAFE_GATE=safe_gate,
    )
    return dq2, dk2, db2.sum(0).add_(db), dg2
