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

    Akkd = Akkd + (bos * HV + i_hv).to(tl.int64) * BC
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

    q = q + (bos * H + i_h).to(tl.int64) * K
    k = k + (bos * H + i_h).to(tl.int64) * K
    g = g + (bos * HV + i_hv).to(tl.int64) * K
    beta = beta + (bos * HV + i_hv).to(tl.int64)
    Aqk = Aqk + (bos * HV + i_hv).to(tl.int64) * BT
    Akk = Akk + (bos * HV + i_hv).to(tl.int64) * BC

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

    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    g += (bos * HV + i_hv).to(tl.int64) * K
    Aqk += (bos * HV + i_hv).to(tl.int64) * BT
    Akk += (bos * HV + i_hv).to(tl.int64) * BT
    Akkd += (bos * HV + i_hv).to(tl.int64) * BC

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

    p_b1 = tl.make_block_ptr(beta + (bos * HV + i_hv).to(tl.int64), (T,), (HV,), (i_tc1,), (BC,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + (bos * HV + i_hv).to(tl.int64), (T,), (HV,), (i_tc2,), (BC,), (0,))
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

        p_b3 = tl.make_block_ptr(beta + (bos * HV + i_hv).to(tl.int64), (T,), (HV,), (i_tc3,), (BC,), (0,))
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


# Split at debug_barrier: dq/db half has lower peak live set than dk/dg half.
# BC=32 cuts NC 4→2 (chunk_size=64) so the future-subchunk loop in dkt_future
# drops from 6 pair-blocks to 1; Cube tiles are 32×32 instead of 16×16.
# SAFE_GATE diag keeps many concurrent BC×BK fp32 tiles live. mem_mult=9 lets
# host pick BK=128 (NK 4→2 on D256) while dq_db (past+diag) still fits 192KB UB.
# dk_dg has a smaller live set and can use BK=256 (NK=1). dkt_future compiles at
# BK=256 but faults at runtime (UB address OOB), so it stays on BK=128.
_BWD_INTRA_BC = 32
_BWD_INTRA_DQ_MEM_MULT = 9.0
_BWD_INTRA_DK_MEM_MULT = 4.5
_MAX_BK_DQ = 128
_MAX_BK_DK = 256


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _get_bwd_intra_bk(K: int, BC: int = _BWD_INTRA_BC, *, element_size: int = 2) -> int:
    # fp32 I/O roughly doubles compile-time element-tile UB vs bf16 in dq_db SAFE_GATE.
    max_bk = _MAX_BK_DQ if element_size <= 2 else min(_MAX_BK_DQ, 32)
    return compute_row_tile_block_size(
        BC,
        K,
        _BWD_INTRA_DQ_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(max_bk, triton.next_power_of_2(K)),
    )


def _get_bwd_intra_bk_dk(K: int, BC: int = _BWD_INTRA_BC, *, element_size: int = 2) -> int:
    """Wider K tile for dk_dg (no past-subchunk live set). dkt_future cannot use this."""
    max_bk = _MAX_BK_DK if element_size <= 2 else min(_MAX_BK_DK, 32)
    return compute_row_tile_block_size(
        BC,
        K,
        _BWD_INTRA_DK_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(max_bk, triton.next_power_of_2(K)),
    )


def _launch_bwd_intra_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **kernel_kwargs)


def _gk_npu_arg(g: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    """Transpose g to [B, HV, T, K] for stride-1 row loads along T."""
    if HV == 1:
        return g, False
    return g.transpose(1, 2).contiguous(), True


def _hv_t_npu_arg(x: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    """Transpose [B, T, HV] to [B, HV, T] for stride-1 loads along T."""
    if HV == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


@triton.jit
def _bwd_intra_beta_base(beta, bos, i_b, i_hv, T_seq, HV, IS_VARLEN: tl.constexpr, BETA_T_CONTIG: tl.constexpr):
    if BETA_T_CONTIG:
        if IS_VARLEN:
            return beta + (bos + i_hv.to(tl.int64) * T_seq)
        return beta + (tl.cast(i_b, tl.int64) * HV + i_hv) * T_seq
    return beta + (bos * HV + i_hv).to(tl.int64)


@triton.jit
def _bwd_intra_beta_row_stride(BETA_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if BETA_T_CONTIG:
        return 1
    return HV


@triton.jit
def _bwd_intra_g_base(g, bos, i_b, i_hv, T_seq, K, HV, IS_VARLEN: tl.constexpr, G_T_CONTIG: tl.constexpr):
    if G_T_CONTIG:
        if IS_VARLEN:
            return g + (bos * K).to(tl.int64) + i_hv.to(tl.int64) * T_seq * K
        return g + tl.cast(i_b, tl.int64) * HV * T_seq * K + i_hv.to(tl.int64) * T_seq * K
    return g + (bos * HV + i_hv).to(tl.int64) * K


@triton.jit
def _bwd_intra_g_row_stride(G_T_CONTIG: tl.constexpr, HV: tl.constexpr, K: tl.constexpr):
    if G_T_CONTIG:
        return K
    return HV * K


@triton.jit
def _bwd_intra_g_block_ptr(g_base, T, row, col, BC, BK, g_row_stride, K: tl.constexpr):
    return tl.make_block_ptr(g_base, (T, K), (g_row_stride, 1), (row, col), (BC, BK), (1, 0))


@triton.jit(do_not_specialize=['B', 'T', 'NT', 'BH_TOTAL', 'task_num', 'num_core'])
def chunk_kda_bwd_kernel_intra_dq_db_npu(
    q, k, g, beta, dAqk, dAkk, dq, dq2, dk2, dg2, db,
    cu_seqlens, chunk_indices, B, T, NT, BH_TOTAL, task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr, SAFE_GATE: tl.constexpr,
    G_T_CONTIG: tl.constexpr, BETA_T_CONTIG: tl.constexpr,
):
    core_id = tl.program_id(0)
    g_row_stride = _bwd_intra_g_row_stride(G_T_CONTIG, HV, K)
    beta_row_stride = _bwd_intra_beta_row_stride(BETA_T_CONTIG, HV)
    o_i = tl.arange(0, BC)
    # One (i_k, i_i) tile per task; NK*NC encoded in i_kc to keep parallel granularity.
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT
        i_kc = rem // NT
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)
        i_k, i_i = i_kc // NC, i_kc % NC
        T_seq = T
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
        T_cur = (eos - bos).to(tl.int32)
        i_ti = i_t * BT + i_i * BC
        if i_ti < T_cur:
            all = tl.cast(B, tl.int64) * T
            q_ptr = q + (bos * H + i_h).to(tl.int64) * K
            k_ptr = k + (bos * H + i_h).to(tl.int64) * K
            g_base = _bwd_intra_g_base(g, bos, i_b, i_hv, T_seq, K, HV, IS_VARLEN, G_T_CONTIG)
            beta_base = _bwd_intra_beta_base(beta, bos, i_b, i_hv, T_seq, HV, IS_VARLEN, BETA_T_CONTIG)
            dAqk_ptr = dAqk + (bos * HV + i_hv).to(tl.int64) * BT
            dAkk_ptr = dAkk + (bos * HV + i_hv).to(tl.int64) * BT
            dq_ptr = dq + (bos * HV + i_hv).to(tl.int64) * K
            dq2_ptr = dq2 + (bos * HV + i_hv).to(tl.int64) * K
            dk2_ptr = dk2 + (bos * HV + i_hv).to(tl.int64) * K
            dg2_ptr = dg2 + (bos * HV + i_hv).to(tl.int64) * K
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            db_ptr = db + (i_k * all + bos).to(tl.int64) * HV + i_hv
            p_g = _bwd_intra_g_block_ptr(g_base, T_cur, i_ti, i_k * BK, BC, BK, g_row_stride, K)
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            p_b = tl.make_block_ptr(beta_base, (T_cur,), (beta_row_stride,), (i_ti,), (BC,), (0,))
            b_b = tl.load(p_b, boundary_check=(0,))
            b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
            b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
            if i_i > 0:
                p_gn = g_base + i_ti.to(tl.int64) * g_row_stride + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
                for i_j in range(0, i_i):
                    p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1),
                                            (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                    p_gk = _bwd_intra_g_block_ptr(g_base, T_cur, i_t * BT + i_j * BC,
                                                  i_k * BK, BC, BK, g_row_stride, K)
                    p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (HV * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
                    p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (HV * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_gk = tl.load(p_gk, boundary_check=(0, 1))
                    b_kg = b_k * exp2(b_gn - b_gk.to(tl.float32))
                    b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
                    b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
                    b_dq2 += tl.dot(b_dAqk.to(tl.float32), b_kg.to(tl.float32), allow_tf32=False)
                    b_dk2 += tl.dot(b_dAkk.to(tl.float32), b_kg.to(tl.float32), allow_tf32=False)
                b_gqn = exp2(b_g - b_gn)
                b_dq2 *= b_gqn
                b_dk2 *= b_gqn
            m_dA = (i_ti + o_i) < T_cur
            o_dA = (i_ti + o_i).to(tl.int64) * HV * BT + i_i * BC
            p_kj = k_ptr + i_ti.to(tl.int64) * H * K + o_k
            p_gkj = g_base + i_ti.to(tl.int64) * g_row_stride + o_k
            p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if SAFE_GATE:
                p_gn = g_base + (i_ti + min(BC // 2, T_cur - i_ti - 1)).to(tl.int64) * g_row_stride + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
                p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
                p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
                b_dAqk_diag = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
                b_dAkk_diag = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)
                m_i_diag = (o_i[:, None] >= o_i[None, :]) & (
                    (i_ti + o_i[:, None]) < T_cur) & ((i_ti + o_i[None, :]) < T_cur)
                m_j_diag = (i_ti + o_i[:, None]) < T_cur
                b_dAqk_diag = tl.where(m_i_diag, b_dAqk_diag, 0.)
                b_dAkk_diag = tl.where(m_i_diag, b_dAkk_diag, 0.)
                b_g_diag = tl.where(m_j_diag, b_g - b_gn, 0.)
                exp_b_g_diag = tl.where(m_j_diag, exp2(b_g_diag), 0.)
                exp_neg_b_g_diag = tl.where(m_j_diag, exp2(-b_g_diag), 0.)
                b_k_exp = b_k * exp_neg_b_g_diag
                b_dq2 += tl.dot(b_dAqk_diag, b_k_exp, allow_tf32=False) * exp_b_g_diag
                b_dk2 += tl.dot(b_dAkk_diag, b_k_exp, allow_tf32=False) * exp_b_g_diag
            else:
                for j in range(0, min(BC, T_cur - i_t * BT - i_i * BC)):
                    b_dAqk = tl.load(dAqk_ptr + o_dA + j, mask=m_dA, other=0)
                    b_dAkk = tl.load(dAkk_ptr + o_dA + j, mask=m_dA, other=0)
                    b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
                    b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
                    m_i = o_i[:, None] >= j
                    b_gqk = exp2(b_g - b_gkj[None, :])
                    b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
                    b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)
                    p_kj += H * K
                    p_gkj += g_row_stride
            b_db = tl.sum(b_dk2 * b_k, 1)
            b_dk2 *= b_b[:, None]
            p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            p_dq = tl.make_block_ptr(dq_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dq2 = tl.make_block_ptr(dq2_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dk2 = tl.make_block_ptr(dk2_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dg2 = tl.make_block_ptr(dg2_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_db = tl.make_block_ptr(db_ptr, (T_cur,), (HV,), (i_ti,), (BC,), (0,))
            b_dg2 = b_q * b_dq2
            b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
            tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T', 'NT', 'BH_TOTAL', 'task_num', 'num_core'])
def chunk_kda_bwd_kernel_intra_dkt_future_npu(
    q, k, g, beta, dAqk, dAkk, dkt_part,
    cu_seqlens, chunk_indices, T, NT, BH_TOTAL, task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, NC_FUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr, BETA_T_CONTIG: tl.constexpr,
):
    core_id = tl.program_id(0)
    g_row_stride = _bwd_intra_g_row_stride(G_T_CONTIG, HV, K)
    beta_row_stride = _bwd_intra_beta_row_stride(BETA_T_CONTIG, HV)
    o_i = tl.arange(0, BC)
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT
        i_kc = rem // NT
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)
        i_k, i_i = i_kc // NC_FUT, i_kc % NC_FUT
        T_seq = T
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
        T_cur = (eos - bos).to(tl.int32)
        i_ti = i_t * BT + i_i * BC
        if i_ti < T_cur:
            q_ptr = q + (bos * H + i_h).to(tl.int64) * K
            k_ptr = k + (bos * H + i_h).to(tl.int64) * K
            g_base = _bwd_intra_g_base(g, bos, i_b, i_hv, T_seq, K, HV, IS_VARLEN, G_T_CONTIG)
            beta_base = _bwd_intra_beta_base(beta, bos, i_b, i_hv, T_seq, HV, IS_VARLEN, BETA_T_CONTIG)
            dAqk_ptr = dAqk + (bos * HV + i_hv).to(tl.int64) * BT
            dAkk_ptr = dAkk + (bos * HV + i_hv).to(tl.int64) * BT
            dkt_part_ptr = dkt_part + (bos * HV + i_hv).to(tl.int64) * K
            nc_eff = min(NC, tl.cdiv(T_cur - i_t * BT, BC))
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_g = _bwd_intra_g_block_ptr(g_base, T_cur, i_ti, i_k * BK, BC, BK, g_row_stride, K)
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
            if i_i < nc_eff - 1:
                p_gn = g_base + (min(i_ti + BC, T_cur) - 1).to(tl.int64) * g_row_stride + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
                for i_j in range(i_i + 1, nc_eff):
                    p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                    p_kj = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                    p_gk = _bwd_intra_g_block_ptr(g_base, T_cur, i_t * BT + i_j * BC, i_k * BK, BC, BK, g_row_stride, K)
                    p_bj = tl.make_block_ptr(beta_base, (T_cur,), (beta_row_stride,), (i_t * BT + i_j * BC,), (BC,), (0,))
                    p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (HV * BT, 1),
                                               (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
                    p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (HV * BT, 1),
                                               (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
                    b_bj = tl.load(p_bj, boundary_check=(0,))
                    b_qj = tl.load(p_q, boundary_check=(0, 1))
                    b_kj = tl.load(p_kj, boundary_check=(0, 1))
                    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
                    b_dAqk = tl.trans(tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32))
                    b_dAkk = tl.trans(tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32))
                    o_j = i_t * BT + i_j * BC + o_i
                    m_j = o_j < T_cur
                    b_gkn = exp2(b_gk - b_gn)
                    b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
                    b_kbg = b_kj * b_bj[:, None] * tl.where(m_j[:, None], b_gkn, 0)
                    b_dkt += tl.dot(b_dAqk, b_qg.to(tl.float32), allow_tf32=False)
                    b_dkt += tl.dot(b_dAkk, b_kbg.to(tl.float32), allow_tf32=False)
                b_dkt *= exp2(b_gn - b_g)
            p_dkt_part = tl.make_block_ptr(dkt_part_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dkt_part, b_dkt.to(p_dkt_part.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'NT', 'BH_TOTAL', 'task_num', 'num_core'])
def chunk_kda_bwd_kernel_intra_dk_dg_npu(
    q, k, g, beta, dAqk, dAkk, dk, dk2, dg, dg2, dkt_part,
    cu_seqlens, chunk_indices, T, NT, BH_TOTAL, task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr, SAFE_GATE: tl.constexpr,
    G_T_CONTIG: tl.constexpr, BETA_T_CONTIG: tl.constexpr,
):
    core_id = tl.program_id(0)
    g_row_stride = _bwd_intra_g_row_stride(G_T_CONTIG, HV, K)
    beta_row_stride = _bwd_intra_beta_row_stride(BETA_T_CONTIG, HV)
    o_i = tl.arange(0, BC)
    for task_id in tl.range(core_id, task_num, num_core):
        i_bh = task_id % BH_TOTAL
        rem = task_id // BH_TOTAL
        i_t = rem % NT
        i_kc = rem // NT
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)
        i_k, i_i = i_kc // NC, i_kc % NC
        T_seq = T
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        else:
            bos = tl.cast(i_b, tl.int64) * T
            eos = bos + T
        T_cur = (eos - bos).to(tl.int32)
        i_ti = i_t * BT + i_i * BC
        if i_ti < T_cur:
            q_ptr = q + (bos * H + i_h).to(tl.int64) * K
            k_ptr = k + (bos * H + i_h).to(tl.int64) * K
            g_base = _bwd_intra_g_base(g, bos, i_b, i_hv, T_seq, K, HV, IS_VARLEN, G_T_CONTIG)
            beta_base = _bwd_intra_beta_base(beta, bos, i_b, i_hv, T_seq, HV, IS_VARLEN, BETA_T_CONTIG)
            dAqk_ptr = dAqk + (bos * HV + i_hv).to(tl.int64) * BT
            dAkk_ptr = dAkk + (bos * HV + i_hv).to(tl.int64) * BT
            dk_ptr = dk + (bos * HV + i_hv).to(tl.int64) * K
            dk2_ptr = dk2 + (bos * HV + i_hv).to(tl.int64) * K
            dg_ptr = dg + (bos * HV + i_hv).to(tl.int64) * K
            dg2_ptr = dg2 + (bos * HV + i_hv).to(tl.int64) * K
            dkt_part_ptr = dkt_part + (bos * HV + i_hv).to(tl.int64) * K
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p_g = _bwd_intra_g_block_ptr(g_base, T_cur, i_ti, i_k * BK, BC, BK, g_row_stride, K)
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_b = tl.make_block_ptr(beta_base, (T_cur,), (beta_row_stride,), (i_ti,), (BC,), (0,))
            b_b = tl.load(p_b, boundary_check=(0,))
            p_dk2 = tl.make_block_ptr(dk2_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dg2 = tl.make_block_ptr(dg2_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dkt_part = tl.make_block_ptr(dkt_part_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_dk2 = tl.load(p_dk2, boundary_check=(0, 1)).to(tl.float32)
            b_dg2 = tl.load(p_dg2, boundary_check=(0, 1)).to(tl.float32)
            b_dkt = tl.load(p_dkt_part, boundary_check=(0, 1)).to(tl.float32)
            o_dA = i_ti.to(tl.int64) * HV * BT + i_i * BC + o_i
            p_qj = q_ptr + i_ti.to(tl.int64) * H * K + o_k
            p_kj = k_ptr + i_ti.to(tl.int64) * H * K + o_k
            p_gkj = g_base + i_ti.to(tl.int64) * g_row_stride + o_k
            p_bj = beta_base + i_ti.to(tl.int64) * beta_row_stride
            if SAFE_GATE:
                p_gn = g_base + (i_ti + min(BC // 2, T_cur - i_ti - 1)).to(tl.int64) * g_row_stride + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
                p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                p_dAqk = tl.make_block_ptr(dAqk_ptr, (T_cur, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
                p_dAkk = tl.make_block_ptr(dAkk_ptr, (T_cur, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
                b_dAqk_diag = tl.trans(tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32))
                b_dAkk_diag = tl.trans(tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32))
                m_i_diag = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T_cur) & ((i_ti + o_i[None, :]) < T_cur)
                m_j_diag = (i_ti + o_i[:, None]) < T_cur
                b_dAqk_diag = tl.where(m_i_diag, b_dAqk_diag, 0.)
                b_dAkk_diag = tl.where(m_i_diag, b_dAkk_diag, 0.)
                b_g_diag = tl.where(m_j_diag, b_g - b_gn, 0.)
                exp_b_g_diag = tl.where(m_j_diag, exp2(b_g_diag), 0.)
                exp_neg_b_g_diag = tl.where(m_j_diag, exp2(-b_g_diag), 0.)
                b_q_exp = b_q * exp_b_g_diag
                b_kb_exp = b_k * b_b[:, None] * exp_b_g_diag
                b_dkt += tl.dot(b_dAqk_diag, b_q_exp, allow_tf32=False) * exp_neg_b_g_diag
                b_dkt += tl.dot(b_dAkk_diag, b_kb_exp, allow_tf32=False) * exp_neg_b_g_diag
            else:
                for j in range(0, min(BC, T_cur - i_t * BT - i_i * BC)):
                    b_dAqk = tl.load(dAqk_ptr + o_dA + j * HV * BT)
                    b_dAkk = tl.load(dAkk_ptr + o_dA + j * HV * BT)
                    b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
                    b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
                    b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
                    m_i = o_i[:, None] <= j
                    b_gkq = exp2(b_gkj[None, :] - b_g)
                    b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
                    b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)
                    p_qj += H * K
                    p_kj += H * K
                    p_gkj += g_row_stride
                    p_bj += beta_row_stride
            p_dk = tl.make_block_ptr(dk_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            p_dg = tl.make_block_ptr(dg_ptr, (T_cur, K), (HV * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
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
    BC = min(_BWD_INTRA_BC, BT)
    # dq_db writes past+diag into fp32 dk2/dg2; tile for fp32 I/O even if dq is bf16.
    elem = max(dq.element_size(), 4)
    BK = _get_bwd_intra_bk(K, BC, element_size=elem)
    BK_dk = _get_bwd_intra_bk_dk(K, BC, element_size=elem)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    NK_dk = triton.cdiv(K, BK_dk)
    is_varlen = cu_seqlens is not None
    g_arg, g_t_contig = _gk_npu_arg(g, HV)
    beta_arg, beta_t_contig = _hv_t_npu_arg(beta, HV)

    dq2 = torch.empty_like(dq)
    # Stream-ordered: dq_db writes past+diag here; dk_dg loads then stores in place.
    # fp32 until the final store so dkt is fused without an intermediate bf16 round.
    dk2 = torch.empty_like(dk, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    # Last subchunk has no future blocks; keep zeros and skip those tasks.
    dkt_part = torch.zeros_like(dk, dtype=torch.float)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)

    bh_total = B * HV
    task_num = NK * NC * NT * bh_total
    nc_fut = max(NC - 1, 1)
    task_num_fut = NK * nc_fut * NT * bh_total
    task_num_dk = NK_dk * NC * NT * bh_total
    common_launch = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NT=NT,
        BH_TOTAL=bh_total,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        IS_VARLEN=is_varlen,
        G_T_CONTIG=g_t_contig,
        BETA_T_CONTIG=beta_t_contig,
    )

    _launch_bwd_intra_core_grid(
        chunk_kda_bwd_kernel_intra_dq_db_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=g_arg,
            beta=beta_arg,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dq2=dq2,
            dk2=dk2,
            dg2=dg2,
            db=db2,
            B=B,
            SAFE_GATE=safe_gate,
            **common_launch,
        ),
    )
    if NC > 1:
        _launch_bwd_intra_core_grid(
            chunk_kda_bwd_kernel_intra_dkt_future_npu,
            task_num=task_num_fut,
            kernel_kwargs=dict(
                q=q,
                k=k,
                g=g_arg,
                beta=beta_arg,
                dAqk=dAqk,
                dAkk=dAkk,
                dkt_part=dkt_part,
                NC_FUT=nc_fut,
                **common_launch,
            ),
        )
    _launch_bwd_intra_core_grid(
        chunk_kda_bwd_kernel_intra_dk_dg_npu,
        task_num=task_num_dk,
        kernel_kwargs=dict(
            q=q,
            k=k,
            g=g_arg,
            beta=beta_arg,
            dAqk=dAqk,
            dAkk=dAkk,
            dk=dk,
            dk2=dk2,
            dg=dg,
            dg2=dg2,
            dkt_part=dkt_part,
            SAFE_GATE=safe_gate,
            **{**common_launch, 'BK': BK_dk},
        ),
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = dg2

    return dq, dk, db, dg
