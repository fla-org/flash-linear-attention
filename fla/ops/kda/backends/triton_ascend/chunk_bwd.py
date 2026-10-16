# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA chunk backward kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from fla.ops.common.backends.triton_ascend.chunk_delta_h import mload, mstore, need_dma_mask
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.graph import get_static_buffer
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    get_ub_manager,
    max_grid_axis_chunks,
)


@triton.jit
def _local_rows(start, T_cur, N: tl.constexpr):
    """In-tile token offsets as int32.

    ``start`` is the int32 origin of the tile. An int64 offset vector lowers
    to a scalar loop on Ascend, so the int64 chunk base stays on the pointer.
    """
    rows = tl.arange(0, N)
    return rows, rows < (T_cur - start)


@triton.jit
def _add_rows(ptr, row_base, rows, stride, SPLIT: tl.constexpr):
    """Token-axis pointer offset.

    SPLIT adds the int64 chunk base to the pointer first, so the row vector
    stays int32. Masked launches keep one offset vector; splitting it there
    does not fit in UB.
    """
    if SPLIT:
        return (ptr + row_base * stride) + rows * stride
    return ptr + rows * stride


_DAV_NUM_WARPS = 2
_DAV_MEM_MULT = 8.0
_DAV_SAFETY_MARGIN = 0.75
_DAV_FALLBACK_TILE = 8
_DAV_MAX_TILE = 64


def _get_dAv_bv(BT: int, V: int) -> int:
    return compute_row_tile_block_size(
        BT, V, _DAV_MEM_MULT,
        tiling_row=False,
        safety_margin=_DAV_SAFETY_MARGIN,
        fallback=_DAV_FALLBACK_TILE,
        min_block=8,
        max_block=min(_DAV_MAX_TILE, triton.next_power_of_2(V)),
    )


def _launch_dAv_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, nt, max_nt):
        nt_len = min(max_nt, nt - nt_off)
        kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_DAV_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv_npu(
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT

    o_i = tl.arange(0, BT)
    m_i = (o_i >= 0) & (o_i < BT)
    # Masked and varlen tiles keep one int64 offset. Splitting it mismatches
    # the tail mask on Ascend and leaves those rows unstored.
    if not MASK_DMA:
        rows, m_t = _local_rows(i_t * BT, T, BT)
        row_base = tl.cast(i_t, tl.int64) * BT
    else:
        rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (rows >= 0) & (rows < T)
        row_base = tl.cast(i_t, tl.int64) * BT
    # A lives as [T, BT]. One contiguous load plus a transpose; the old
    # strided [BT, T] gather reloaded that tile on every V slab.
    m_p_A = m_t[:, None] & m_i[None, :]
    p_A = _add_rows(A + (bos * HV + i_hv) * BT, row_base, rows[:, None], HV * BT, not MASK_DMA) + o_i[None, :]
    m_A = (rows[:, None] <= rows[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, tl.trans(mload(p_A, m_p_A, MASK_DMA)), 0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = (o_v >= 0) & (o_v < V)
        m_p_do = m_t[:, None] & m_v[None, :]
        p_v = _add_rows(v, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
        p_do = _add_rows(do, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
        p_dv = _add_rows(dv, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
        b_v = mload(p_v, m_p_do, MASK_DMA)
        b_do = mload(p_do, m_p_do, MASK_DMA)
        b_do_c = b_do + 0.0
        b_dA = tl.dot(b_do, tl.trans(b_v), b_dA, allow_tf32=False)
        b_A_i = (b_A + 0.0).to(b_do.dtype)
        b_dv = tl.dot(b_A_i, b_do_c, allow_tf32=False)
        mstore(p_dv, b_dv.to(p_dv.dtype.element_ty), m_p_do, MASK_DMA)

    m_p_dA = m_t[:, None] & m_i[None, :]
    p_dA = _add_rows(dA, row_base, rows[:, None], HV * BT, not MASK_DMA) + o_i[None, :]
    b_dA = tl.where(rows[:, None] >= rows, b_dA * scale, 0.)
    mstore(p_dA, b_dA.to(p_dA.dtype.element_ty), m_p_dA, MASK_DMA)


@input_guard
def chunk_kda_bwd_dAv_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, HV, V = k.shape[0], k.shape[1], do.shape[2], do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BV = _get_dAv_bv(BT, V)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if use_graph:
        dA = get_static_buffer("dAv_dA", (B, T, HV, BT), torch.float, v.device, zero=True)
        dv = get_static_buffer("dAv_dv", tuple(do.shape), do.dtype, do.device, zero=True)
    else:
        dA = v.new_empty(B, T, HV, BT, dtype=torch.float)
        # dv stays zero-initialized on Ascend eager as well: boundary_check stores
        # can read-modify-write destination lanes of NaN-poisoned empty buffers.
        dv = torch.zeros_like(do)

    _launch_dAv_2d_kernel(
        chunk_kda_bwd_kernel_dAv_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            v=v,
            A=A,
            do=do,
            dv=dv,
            dA=dA,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            HV=HV,
            V=V,
            BT=BT,
            BV=BV,
            IS_VARLEN=cu_seqlens is not None,
            MASK_DMA=need_dma_mask(T, BT, V, BV, V, BV, cu_seqlens is not None),
            USE_GRAPH=use_graph,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    return dA, dv


_BC = 16
# 128×128 at row BT=64 is 160 KiB of fp32 tiles, under 192 KiB.
_WY_SAFETY = 0.90
_FALLBACK_TILE = 16
_MAX_TILE = 128


def _k_part_bc(bt: int) -> int:
    # wy_k_part splits BT into sub-tiles. The launch uses 32 whenever the
    # chunk is large enough, which is wider than the intra _BC of 16.
    return 32 if bt >= 32 else _BC


def _wy_live_bytes(row: int, bk: int, bv: int) -> int:
    """fp32 bytes for a contiguous [BK, BV] state, the [row, BV] operand, its transpose, and the [BK, row] acc."""
    return 4 * (bk * bv + 2 * row * bv + bk * row)


def _select_wy_tiles(K: int, V: int, row: int) -> tuple[int, int]:
    """Largest BK/BV whose matmul tile fits. dw_part's row is BT, which is the wider of the two kernels."""
    cap = int(get_ub_manager().ub_capacity_bytes * _WY_SAFETY)
    bk_lim = min(_MAX_TILE, triton.next_power_of_2(K))
    bv_lim = min(_MAX_TILE, triton.next_power_of_2(V))
    best: tuple[int, int, int, int] | None = None
    bk = 16
    while bk <= bk_lim:
        bv = 16
        while bv <= bv_lim:
            if _wy_live_bytes(row, bk, bv) <= cap:
                nlaunch = triton.cdiv(K, bk) * triton.cdiv(V, bv)
                area = bk * bv
                if best is None or nlaunch < best[0] or (nlaunch == best[0] and area > best[1]):
                    best = (nlaunch, area, bk, bv)
            bv *= 2
        bk *= 2
    if best is None:
        return _FALLBACK_TILE, _FALLBACK_TILE
    return best[2], best[3]


def _t_contig_arg(x: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, bool]:
    """Transpose [B, T, H*, ...] → [B, H*, T, ...] so T-loads are stride-1."""
    if head_dim == 1:
        return x, False
    return x.transpose(1, 2).contiguous(), True


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _launch_wy_dA_finalize(
    kernel,
    *,
    nt: int,
    bh_total: int,
    T: int,
    BT: int,
    is_varlen: bool,
    num_core: int,
    kernel_kwargs: dict,
) -> None:
    """Host-split aligned bulk vs tail so TAIL_MODE is constexpr per launch."""
    kwargs = dict(kernel_kwargs)
    kwargs['num_core'] = num_core
    if is_varlen:
        kwargs['TAIL_MODE'] = 1
        kwargs['NT_OFFSET'] = 0
        kwargs['task_num'] = nt * bh_total
        kernel[(num_core,)](**kwargs)
        return
    n_bulk = nt if T % BT == 0 else max(nt - 1, 0)
    if n_bulk > 0:
        kwargs['TAIL_MODE'] = 0
        kwargs['NT_OFFSET'] = 0
        kwargs['task_num'] = n_bulk * bh_total
        kernel[(num_core,)](**kwargs)
    if T % BT != 0 and nt > 0:
        kwargs['TAIL_MODE'] = 1
        kwargs['NT_OFFSET'] = n_bulk
        kwargs['task_num'] = bh_total
        kernel[(num_core,)](**kwargs)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_v_part_npu(
    v,
    beta,
    A,
    dv,
    dv2,
    dA_acc,
    db_acc,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    MASK_DMA: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        is_valid = True
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            if USE_GRAPH:
                # graph padding uses [-1, 0]; exclude it before sequence-dependent memory accesses.
                is_valid = i_n >= 0
        # skip only this task: returning would drop later valid tasks assigned to this core.
        if is_valid:
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                T = (eos - bos).to(tl.int32)
            else:
                bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

            if G_T_CONTIG:
                if IS_VARLEN:
                    v_ptr = v + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                    dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                    A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                    beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                else:
                    hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                    v_ptr = v + hv_off * T_seq * V
                    dv_ptr = dv + hv_off * T_seq * V
                    A_ptr = A + hv_off * T_seq * BT
                    beta_ptr = beta + hv_off * T_seq
                v_stride_t = V
                a_stride_t = BT
                beta_stride = 1
            else:
                v_ptr = v + (bos * HV + i_hv) * V
                dv_ptr = dv + (bos * HV + i_hv) * V
                A_ptr = A + (bos * HV + i_hv) * BT
                beta_ptr = beta + bos * HV + i_hv
                v_stride_t = HV * V
                a_stride_t = HV * BT
                beta_stride = HV

            dv2_ptr = dv2 + (bos * HV + i_hv) * V
            dA_ptr = dA_acc + (bos * HV + i_hv) * BT
            db_ptr = db_acc + bos * HV + i_hv

            o_i = tl.arange(0, BT)
            m_i = (o_i >= 0) & (o_i < BT)
            if not MASK_DMA:
                rows, m_t = _local_rows(i_t * BT, T, BT)
                row_base = tl.cast(i_t, tl.int64) * BT
            else:
                rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
                m_t = (rows >= 0) & (rows < T)
                row_base = tl.cast(i_t, tl.int64) * BT
            m_p_A = m_i[:, None] & m_t[None, :]
            p_A = _add_rows(A_ptr, row_base, rows[None, :], a_stride_t, not MASK_DMA) + o_i[:, None]
            p_beta = _add_rows(beta_ptr, row_base, rows, beta_stride, not MASK_DMA)
            b_A = mload(p_A, m_p_A, MASK_DMA)
            b_beta = mload(p_beta, m_t, MASK_DMA)

            b_dA = tl.zeros([BT, BT], dtype=tl.float32)
            b_db = tl.zeros([BT], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                o_v = i_v * BV + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_dv = m_t[:, None] & m_v[None, :]
                p_dv = _add_rows(dv_ptr, row_base, rows[:, None], v_stride_t, not MASK_DMA) + o_v[None, :]
                p_v = _add_rows(v_ptr, row_base, rows[:, None], v_stride_t, not MASK_DMA) + o_v[None, :]
                b_dv = mload(p_dv, m_p_dv, MASK_DMA)
                b_v = mload(p_v, m_p_dv, MASK_DMA)
                b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA, allow_tf32=False)
                # Ascend tl.dot clobbers lhs; copy A before every V-slab use.
                b_A_c = b_A + 0.0
                b_dvb = tl.dot(b_A_c, b_dv, allow_tf32=False)
                b_db += tl.sum(b_dvb * b_v, 1)
                p_dv2 = _add_rows(dv2_ptr, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
                mstore(p_dv2,  (b_dvb * b_beta[:, None]).to(p_dv2.dtype.element_ty), m_p_dv, MASK_DMA)

            m_p_dA = m_t[:, None] & m_i[None, :]
            p_dA = _add_rows(dA_ptr, row_base, rows[:, None], HV * BT, not MASK_DMA) + o_i[None, :]
            p_db = _add_rows(db_ptr, row_base, rows, HV, not MASK_DMA)
            mstore(p_dA,  b_dA.to(p_dA.dtype.element_ty), m_p_dA, MASK_DMA)
            mstore(p_db,  b_db.to(p_db.dtype.element_ty), m_t, MASK_DMA)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_k_part_npu(
    q,
    k,
    v_new,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
    K_OFFSET: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        is_valid = True
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            if USE_GRAPH:
                # graph padding uses [-1, 0]; exclude it before sequence-dependent memory accesses.
                is_valid = i_n >= 0
        # skip only this task: returning would drop later valid tasks assigned to this core.
        if is_valid:
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                T = (eos - bos).to(tl.int32)
                i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + tl.cast(i_t, tl.int64)
            else:
                i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
                bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

            q_ptr = q + (bos * H + i_h) * K
            k_ptr = k + (bos * H + i_h) * K
            v_new_ptr = v_new + (bos * HV + i_hv) * V
            g_ptr = g + (bos * HV + i_hv) * K
            h_ptr = h + (i_tg * HV + i_hv) * K * V
            do_ptr = do + (bos * HV + i_hv) * V
            dh_ptr = dh + (i_tg * HV + i_hv) * K * V
            dq_ptr = dq + (bos * HV + i_hv) * K
            dk_ptr = dk + (bos * HV + i_hv) * K
            dg_ptr = dg + (bos * HV + i_hv) * K

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K

            p_gn = g_ptr + (min(T, tl.cast(i_t, tl.int64) * BT + BT) - 1).to(tl.int64) * HV * K + o_k
            b_gn = mload(p_gn, m_k, MASK_DMA).to(tl.float32)

            n_sub = BT // BC
            b_dgk = tl.zeros([BK], dtype=tl.float32)

            # A full fp32 [BK, BV] product is 64 KiB at head dim 128 and does not fit
            # next to the two state tiles. Reduce V in 16-wide slices instead.
            DG_STEP: tl.constexpr = 16
            for i_v in range(tl.cdiv(V, BV)):
                o_k_2 = i_k * BK + tl.arange(0, BK)
                m_k_2 = (o_k_2 >= 0) & (o_k_2 < K)
                for i_s in range(BV // DG_STEP):
                    o_vs = i_v * BV + i_s * DG_STEP + tl.arange(0, DG_STEP)
                    m_vs = (o_vs >= 0) & (o_vs < V)
                    if STATE_V_FIRST:
                        m_p_h = m_vs[:, None] & m_k_2[None, :]
                        p_h = h_ptr + o_vs[:, None] * K + o_k_2[None, :]
                        p_dh = dh_ptr + o_vs[:, None] * K + o_k_2[None, :]
                        b_h = mload(p_h, m_p_h, MASK_DMA)
                        b_dh = mload(p_dh, m_p_h, MASK_DMA)
                        b_dgk += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32), axis=0)
                    else:
                        m_p_h = m_k_2[:, None] & m_vs[None, :]
                        p_h = h_ptr + o_k_2[:, None] * V + o_vs[None, :]
                        p_dh = dh_ptr + o_k_2[:, None] * V + o_vs[None, :]
                        b_h = mload(p_h, m_p_h, MASK_DMA)
                        b_dh = mload(p_dh, m_p_h, MASK_DMA)
                        b_dgk += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32), axis=1)

            b_dgk *= exp2(b_gn)

            b_kdk_sum = tl.zeros([BK], dtype=tl.float32)
            for s in range(n_sub):
                if not MASK_DMA:
                    sub = i_t * BT + s * BC
                    rows, m_t = _local_rows(sub, T, BC)
                    row_base = tl.cast(i_t, tl.int64) * BT + s * BC
                else:
                    row_base = tl.cast(i_t, tl.int64) * BT + s * BC
                    rows = row_base + tl.arange(0, BC)
                    m_t = (rows >= 0) & (rows < T)
                m_s = m_t

                o_k_2 = i_k * BK + tl.arange(0, BK)
                m_k_2 = (o_k_2 >= 0) & (o_k_2 < K)
                m_p_k = m_t[:, None] & m_k_2[None, :]
                p_k = _add_rows(k_ptr, row_base, rows[:, None], H * K, not MASK_DMA) + o_k_2[None, :]
                p_g = _add_rows(g_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k_2[None, :]
                b_k = mload(p_k, m_p_k, MASK_DMA)
                b_g = mload(p_g, m_p_k, MASK_DMA).to(tl.float32)

                if STATE_V_FIRST:
                    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
                    for i_v in range(tl.cdiv(V, BV)):
                        o_v = i_v * BV + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_v_new = m_t[:, None] & m_v[None, :]
                        p_v_new = _add_rows(v_new_ptr, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
                        m_p_dh = m_v[:, None] & m_k_2[None, :]
                        p_dh = dh_ptr + o_v[:, None] * K + o_k_2[None, :]
                        b_dh = mload(p_dh, m_p_dh, MASK_DMA)
                        b_v_new = mload(p_v_new, m_p_v_new, MASK_DMA)
                        b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk, allow_tf32=False)
                else:
                    # dk^T = dh[BK, BV] @ v^T so the accumulator stays contiguous in BC.
                    b_dk_t = tl.zeros([BK, BC], dtype=tl.float32)
                    for i_v in range(tl.cdiv(V, BV)):
                        o_v = i_v * BV + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_v_new = m_t[:, None] & m_v[None, :]
                        p_v_new = _add_rows(v_new_ptr, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
                        m_p_dh = m_k_2[:, None] & m_v[None, :]
                        p_dh = dh_ptr + o_k_2[:, None] * V + o_v[None, :]
                        b_dh = mload(p_dh, m_p_dh, MASK_DMA)
                        b_v_new = mload(p_v_new, m_p_v_new, MASK_DMA)
                        b_dk_t = tl.dot(b_dh.to(b_v_new.dtype), tl.trans(b_v_new), b_dk_t, allow_tf32=False)
                    b_dk = tl.trans(b_dk_t)

                b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
                b_kdk_sum += tl.sum(b_k * b_dk, axis=0)
                p_dk = _add_rows(dk_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k_2[None, :]
                mstore(p_dk,  b_dk.to(p_dk.dtype.element_ty), m_p_k, MASK_DMA)

            b_dgk_total = b_dgk + b_kdk_sum

            for s in range(n_sub):
                if not MASK_DMA:
                    sub = i_t * BT + s * BC
                    rows, m_t = _local_rows(sub, T, BC)
                    row_base = tl.cast(i_t, tl.int64) * BT + s * BC
                    m_last_s = (sub + rows) == min(T, i_t * BT + BT) - 1
                else:
                    row_base = tl.cast(i_t, tl.int64) * BT + s * BC
                    rows = row_base + tl.arange(0, BC)
                    m_t = (rows >= 0) & (rows < T)
                    m_last_s = rows == min(T, tl.cast(i_t, tl.int64) * BT + BT) - 1

                o_k_2 = i_k * BK + tl.arange(0, BK)
                m_k_2 = (o_k_2 >= 0) & (o_k_2 < K)
                m_p_k = m_t[:, None] & m_k_2[None, :]
                p_k = _add_rows(k_ptr, row_base, rows[:, None], H * K, not MASK_DMA) + o_k_2[None, :]
                p_g = _add_rows(g_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k_2[None, :]
                p_q = _add_rows(q_ptr, row_base, rows[:, None], H * K, not MASK_DMA) + o_k_2[None, :]
                p_dk = _add_rows(dk_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k_2[None, :]
                b_k = mload(p_k, m_p_k, MASK_DMA)
                b_g = mload(p_g, m_p_k, MASK_DMA).to(tl.float32)
                b_q = mload(p_q, m_p_k, MASK_DMA)
                b_dk = mload(p_dk, m_p_k, MASK_DMA).to(tl.float32)

                if STATE_V_FIRST:
                    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
                    for i_v in range(tl.cdiv(V, BV)):
                        o_v = i_v * BV + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_do = m_t[:, None] & m_v[None, :]
                        p_do = _add_rows(do_ptr, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
                        m_p_h = m_v[:, None] & m_k_2[None, :]
                        p_h = h_ptr + o_v[:, None] * K + o_k_2[None, :]
                        b_h = mload(p_h, m_p_h, MASK_DMA)
                        b_do = mload(p_do, m_p_do, MASK_DMA)
                        b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq, allow_tf32=False)
                else:
                    b_dq_t = tl.zeros([BK, BC], dtype=tl.float32)
                    for i_v in range(tl.cdiv(V, BV)):
                        o_v = i_v * BV + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_do = m_t[:, None] & m_v[None, :]
                        p_do = _add_rows(do_ptr, row_base, rows[:, None], HV * V, not MASK_DMA) + o_v[None, :]
                        m_p_h = m_k_2[:, None] & m_v[None, :]
                        p_h = h_ptr + o_k_2[:, None] * V + o_v[None, :]
                        b_h = mload(p_h, m_p_h, MASK_DMA)
                        b_do = mload(p_do, m_p_do, MASK_DMA)
                        b_dq_t = tl.dot(b_h.to(b_do.dtype), tl.trans(b_do), b_dq_t, allow_tf32=False)
                    b_dq = tl.trans(b_dq_t)

                b_dq = b_dq * exp2(b_g) * scale
                b_dg = b_q * b_dq - b_k * b_dk + m_last_s[:, None] * b_dgk_total

                p_dq = _add_rows(dq_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k_2[None, :]
                p_dg = _add_rows(dg_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k_2[None, :]
                mstore(p_dq,  b_dq.to(p_dq.dtype.element_ty), m_p_k, MASK_DMA)
                mstore(p_dg,  b_dg.to(p_dg.dtype.element_ty), m_p_k, MASK_DMA)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def chunk_kda_bwd_kernel_wy_dw_part_npu(
    k,
    g,
    beta,
    A,
    h,
    dv,
    dA_acc,
    db_acc,
    dg,
    dk,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    BH,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_T_CONTIG: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    MASK_DMA: tl.constexpr,
    K_OFFSET: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        is_valid = True
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            if USE_GRAPH:
                # graph padding uses [-1, 0]; exclude it before sequence-dependent memory accesses.
                is_valid = i_n >= 0
        # skip only this task: returning would drop later valid tasks assigned to this core.
        if is_valid:
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                T = (eos - bos).to(tl.int32)
                i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + tl.cast(i_t, tl.int64)
            else:
                i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
                bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

            if K_T_CONTIG:
                if IS_VARLEN:
                    k_ptr = k + tl.cast(i_h, tl.int64) * T_seq * K + bos * K
                else:
                    k_ptr = k + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * K
                k_stride_t = K
            else:
                k_ptr = k + (bos * H + i_h) * K
                k_stride_t = H * K

            if G_T_CONTIG:
                if IS_VARLEN:
                    g_ptr = g + tl.cast(i_hv, tl.int64) * T_seq * K + bos * K
                    beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                    A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                    dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
                else:
                    hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                    g_ptr = g + hv_off * T_seq * K
                    beta_ptr = beta + hv_off * T_seq
                    A_ptr = A + hv_off * T_seq * BT
                    dv_ptr = dv + hv_off * T_seq * V
                g_stride_t = K
                a_stride_t = BT
                dv_stride_t = V
                beta_stride = 1
            else:
                g_ptr = g + (bos * HV + i_hv) * K
                beta_ptr = beta + bos * HV + i_hv
                A_ptr = A + (bos * HV + i_hv) * BT
                dv_ptr = dv + (bos * HV + i_hv) * V
                g_stride_t = HV * K
                a_stride_t = HV * BT
                dv_stride_t = HV * V
                beta_stride = HV

            h_ptr = h + (i_tg * HV + i_hv) * K * V
            dA_ptr = dA_acc + (bos * HV + i_hv) * BT
            db_ptr = db_acc + bos * HV + i_hv
            dg_ptr = dg + (bos * HV + i_hv) * K
            dk_ptr = dk + (bos * HV + i_hv) * K

            if not MASK_DMA:
                rows, m_t = _local_rows(i_t * BT, T, BT)
                row_base = tl.cast(i_t, tl.int64) * BT
            else:
                rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
                m_t = (rows >= 0) & (rows < T)
                row_base = tl.cast(i_t, tl.int64) * BT
            if STATE_V_FIRST:
                b_dw = tl.zeros([BT, BK], dtype=tl.float32)
                for i_v in range(tl.cdiv(V, BV)):
                    o_v = i_v * BV + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dv = m_t[:, None] & m_v[None, :]
                    p_dv = _add_rows(dv_ptr, row_base, rows[:, None], dv_stride_t, not MASK_DMA) + o_v[None, :]
                    o_k = i_k * BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h = m_v[:, None] & m_k[None, :]
                    p_h = h_ptr + o_v[:, None] * K + o_k[None, :]
                    b_dv = mload(p_dv, m_p_dv, MASK_DMA)
                    b_h = mload(p_h, m_p_h, MASK_DMA)
                    b_dw = tl.dot(b_dv, b_h.to(b_dv.dtype), b_dw, allow_tf32=False)
            else:
                # dw^T = h[BK, BV] @ dv^T. Avoids gathering [V, K] out of [K, V].
                b_dw_t = tl.zeros([BK, BT], dtype=tl.float32)
                for i_v in range(tl.cdiv(V, BV)):
                    o_v = i_v * BV + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dv = m_t[:, None] & m_v[None, :]
                    p_dv = _add_rows(dv_ptr, row_base, rows[:, None], dv_stride_t, not MASK_DMA) + o_v[None, :]
                    o_k = i_k * BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h = m_k[:, None] & m_v[None, :]
                    p_h = h_ptr + o_k[:, None] * V + o_v[None, :]
                    b_dv = mload(p_dv, m_p_dv, MASK_DMA)
                    b_h = mload(p_h, m_p_h, MASK_DMA)
                    b_dw_t = tl.dot(b_h.to(b_dv.dtype), tl.trans(b_dv), b_dw_t, allow_tf32=False)
                b_dw = tl.trans(b_dw_t)

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = (o_k >= 0) & (o_k < K)
            m_p_k = m_t[:, None] & m_k[None, :]
            p_k = _add_rows(k_ptr, row_base, rows[:, None], k_stride_t, not MASK_DMA) + o_k[None, :]
            p_g = _add_rows(g_ptr, row_base, rows[:, None], g_stride_t, not MASK_DMA) + o_k[None, :]
            p_beta = _add_rows(beta_ptr, row_base, rows, beta_stride, not MASK_DMA)
            o_i = tl.arange(0, BT)
            m_i = (o_i >= 0) & (o_i < BT)
            m_p_A = m_i[:, None] & m_t[None, :]
            p_A = _add_rows(A_ptr, row_base, rows[None, :], a_stride_t, not MASK_DMA) + o_i[:, None]
            b_k = mload(p_k, m_p_k, MASK_DMA)
            b_g = mload(p_g, m_p_k, MASK_DMA).to(tl.float32)
            b_beta = mload(p_beta, m_t, MASK_DMA)
            b_A = mload(p_A, m_p_A, MASK_DMA)
            b_gk_exp = exp2(b_g)
            b_kg = b_k * b_gk_exp
            b_gb = b_gk_exp * b_beta[:, None]
            # Cube dots leave NaN in K-padding lanes. Those lanes are the reduction
            # axis of dA and db. Aligned tiles have no padding, so skip the masks.
            if K % BK != 0:
                b_dw = tl.where(m_k[None, :], -b_dw.to(b_A.dtype), 0)
                b_kg = tl.where(m_k[None, :], b_kg, 0)
            else:
                b_dw = -b_dw.to(b_A.dtype)
            b_kg_a = b_kg.to(b_A.dtype)
            b_dkgb = tl.dot(b_A, b_dw, allow_tf32=False)
            if K % BK != 0:
                b_dkgb = tl.where(m_k[None, :], b_dkgb, 0)

            m_p_dA_acc = m_t[:, None] & m_i[None, :]
            p_dA_acc = _add_rows(dA_ptr, row_base, rows[:, None], HV * BT, not MASK_DMA) + o_i[None, :]
            b_dA = mload(p_dA_acc, m_p_dA_acc, MASK_DMA).to(tl.float32)
            if K % BK != 0:
                b_dw_c = tl.where(m_k[None, :], b_dw + 0.0, 0)
            else:
                b_dw_c = b_dw + 0.0
            b_dA = tl.dot(b_dw_c, tl.trans(b_kg_a), b_dA, allow_tf32=False)
            mstore(p_dA_acc,  b_dA.to(p_dA_acc.dtype.element_ty), m_p_dA_acc, MASK_DMA)

            p_db_acc = _add_rows(db_ptr, row_base, rows, HV, not MASK_DMA)
            b_db = mload(p_db_acc, m_t, MASK_DMA).to(tl.float32)
            b_db += tl.sum(b_dkgb * b_kg, 1)
            mstore(p_db_acc,  b_db.to(p_db_acc.dtype.element_ty), m_t, MASK_DMA)

            p_dk = _add_rows(dk_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k[None, :]
            b_dk = mload(p_dk, m_p_k, MASK_DMA).to(tl.float32)
            b_dk = b_dk + b_dkgb * b_gb
            mstore(p_dk,  b_dk.to(p_dk.dtype.element_ty), m_p_k, MASK_DMA)

            p_dg = _add_rows(dg_ptr, row_base, rows[:, None], HV * K, not MASK_DMA) + o_k[None, :]
            b_dg = mload(p_dg, m_p_k, MASK_DMA).to(tl.float32)
            b_dg = b_dg + b_kg * b_dkgb * b_beta[:, None]
            mstore(p_dg,  b_dg.to(p_dg.dtype.element_ty), m_p_k, MASK_DMA)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH', 'NT_OFFSET'])
def chunk_kda_bwd_kernel_wy_dA_finalize_npu(
    A,
    beta,
    dA_acc,
    db_acc,
    dA,
    db,
    cu_seqlens,
    chunk_indices,
    T,
    BH,
    task_num,
    num_core,
    NT_OFFSET,
    HV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    TAIL_MODE: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    """dA = mask(-A @ ((mask * dA_acc * beta) @ A)); copy db_acc into db.

    Fuses the old mask kernel into mid+finalize so dA_acc stays in UB.
    TAIL_MODE 0 = aligned bulk (tile is in bounds, no mask). TAIL_MODE 1 = tail/varlen.
    First tl.dot clobbers masked dA (dead). Second uses b_A as lhs (dead after store).
    """
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = NT_OFFSET + task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        is_valid = True
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            if USE_GRAPH:
                # graph padding uses [-1, 0]; exclude it before sequence-dependent memory accesses.
                is_valid = i_n >= 0
        # skip only this task: returning would drop later valid tasks assigned to this core.
        if is_valid:
            if IS_VARLEN:
                bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                T = (eos - bos).to(tl.int32)
            else:
                bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

            if G_T_CONTIG:
                if IS_VARLEN:
                    A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                    beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                else:
                    hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                    A_ptr = A + hv_off * T_seq * BT
                    beta_ptr = beta + hv_off * T_seq
                a_stride_t = BT
                beta_stride = 1
            else:
                A_ptr = A + (bos * HV + i_hv) * BT
                beta_ptr = beta + bos * HV + i_hv
                a_stride_t = HV * BT
                beta_stride = HV

            dA_acc_ptr = dA_acc + (bos * HV + i_hv) * BT
            db_acc_ptr = db_acc + bos * HV + i_hv
            dA_ptr = dA + (bos * HV + i_hv) * BT
            db_ptr = db + bos * HV + i_hv

            o_i = tl.arange(0, BT)
            m_i = (o_i >= 0) & (o_i < BT)
            if TAIL_MODE == 0:
                rows, m_rows = _local_rows(i_t * BT, T, BT)
                row_base = tl.cast(i_t, tl.int64) * BT
            else:
                rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
                m_rows = (rows >= 0) & (rows < T)
                row_base = tl.cast(i_t, tl.int64) * BT
            m_p_A = m_i[:, None] & m_rows[None, :]
            p_A = _add_rows(A_ptr, row_base, rows[None, :], a_stride_t, TAIL_MODE == 0) + o_i[:, None]
            m_p_dA_acc = m_rows[:, None] & m_i[None, :]
            p_dA_acc = _add_rows(dA_acc_ptr, row_base, rows[:, None], HV * BT, TAIL_MODE == 0) + o_i[None, :]
            p_dA = _add_rows(dA_ptr, row_base, rows[:, None], HV * BT, TAIL_MODE == 0) + o_i[None, :]
            p_beta = _add_rows(beta_ptr, row_base, rows, beta_stride, TAIL_MODE == 0)
            p_db_acc = _add_rows(db_acc_ptr, row_base, rows, HV, TAIL_MODE == 0)
            p_db = _add_rows(db_ptr, row_base, rows, HV, TAIL_MODE == 0)

            if TAIL_MODE == 0:
                b_A = tl.load(p_A)
                b_dA = tl.load(p_dA_acc).to(tl.float32)
                b_beta = tl.load(p_beta)
                m_A = rows[:, None] > rows[None, :]
            else:
                b_A = tl.load(p_A, mask=m_p_A, other=0)
                b_dA = tl.load(p_dA_acc, mask=m_p_dA_acc, other=0).to(tl.float32)
                b_beta = tl.load(p_beta, mask=m_rows, other=0)
                m_A = (rows[:, None] > rows[None, :]) & (m_rows[:, None] & m_rows[None, :])

            b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
            # mid: (mask * dA_acc * beta) @ A. lhs clobbers b_dA; A is rhs then lhs.
            b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
            b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
            b_fin = tl.where(m_A, -b_fin, 0)

            if TAIL_MODE == 0:
                tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
                tl.store(p_db, tl.load(p_db_acc).to(p_db.dtype.element_ty))
            else:
                tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), mask=m_p_dA_acc)
                tl.store(p_db, tl.load(p_db_acc, mask=m_rows, other=0).to(p_db.dtype.element_ty), mask=m_rows)


@input_guard
def chunk_kda_bwd_wy_dqkg_fused_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_graph: bool = False,
):
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    if BT % _BC != 0:
        raise ValueError(f'KDA Ascend bwd requires chunk_size % {_BC} == 0, got {BT}')

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if use_graph:
        dq = get_static_buffer("wy_dq", (B, T, HV, K), torch.float, q.device, zero=True)
        dk = get_static_buffer("wy_dk", (B, T, HV, K), torch.float, q.device, zero=True)
        dv2 = get_static_buffer("wy_dv2", tuple(v.shape), v.dtype, v.device, zero=True)
        dg = get_static_buffer("wy_dg", tuple(g.shape), torch.float, g.device, zero=True)
        db = get_static_buffer("wy_db", tuple(beta.shape), torch.float, beta.device, zero=True)
        dA = get_static_buffer("wy_dA", tuple(A.shape), torch.float, A.device, zero=True)
    else:
        dq = g.new_empty(B, T, HV, K, dtype=torch.float)
        dk = g.new_empty(B, T, HV, K, dtype=torch.float)
        dv2 = torch.empty_like(v)
        dg = torch.empty_like(g, dtype=torch.float)
        db = torch.empty_like(beta, dtype=torch.float)
        dA = torch.empty_like(A, dtype=torch.float)
    dA_acc = torch.zeros(B, T, HV, BT, dtype=torch.float, device=A.device)
    db_acc = torch.zeros(B, T, HV, dtype=torch.float, device=beta.device)

    bc_k = _k_part_bc(BT)
    # dw_part contracts over BT, which is wider than k_part's BC.
    BK, BV = _select_wy_tiles(K, V, BT)
    NK = triton.cdiv(K, BK)
    is_varlen = cu_seqlens is not None
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if is_varlen else g.new_zeros(1, dtype=torch.int64)

    v_arg, g_t_contig = _t_contig_arg(v, HV)
    beta_arg = _t_contig_arg(beta, HV)[0]
    A_arg = _t_contig_arg(A, HV)[0]
    dv_arg = _t_contig_arg(dv, HV)[0]

    bh_total = B * HV
    num_core = get_npu_properties()['num_vectorcore']
    task_num = NT * bh_total
    # disable auto-multi-buffer on this V-slab launch
    mask_dma = need_dma_mask(T, BT, K, BK, V, BV, is_varlen) or BT % bc_k != 0
    chunk_kda_bwd_kernel_wy_v_part_npu[(num_core,)](
        v=v_arg,
        beta=beta_arg,
        A=A_arg,
        dv=dv_arg,
        dv2=dv2,
        dA_acc=dA_acc,
        db_acc=db_acc,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        HV=HV,
        V=V,
        BT=BT,
        BV=BV,
        IS_VARLEN=is_varlen,
        USE_GRAPH=use_graph,
        G_T_CONTIG=g_t_contig,
        MASK_DMA=need_dma_mask(T, BT, V, BV, V, BV, is_varlen),
        **ascend_compile_kwargs(),
    )

    k_part_kwargs = dict(
        q=q,
        k=k,
        v_new=v_new,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BC=bc_k,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        USE_GRAPH=use_graph,
        MASK_DMA=mask_dma,
    )
    for k_off in range(NK):
        k_part_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_k_part_npu[(num_core,)](
            **k_part_kwargs,
            **ascend_compile_kwargs(),
            enable_ubuf_saving=True,
        )

    k_arg, k_t_contig = _t_contig_arg(k, H)
    g_arg, g_t_contig = _t_contig_arg(g, HV)
    dw_kwargs = dict(
        k=k_arg,
        g=g_arg,
        beta=beta_arg,
        A=A_arg,
        h=h,
        dv=dv_arg,
        dA_acc=dA_acc,
        db_acc=db_acc,
        dg=dg,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        BH=bh_total,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
        USE_GRAPH=use_graph,
        K_T_CONTIG=k_t_contig,
        G_T_CONTIG=g_t_contig,
        MASK_DMA=mask_dma,
    )
    for k_off in range(NK):
        dw_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_dw_part_npu[(num_core,)](
            **dw_kwargs,
            **ascend_compile_kwargs(),
            enable_ubuf_saving=True,
        )

    _launch_wy_dA_finalize(
        chunk_kda_bwd_kernel_wy_dA_finalize_npu,
        nt=NT,
        bh_total=bh_total,
        T=T,
        BT=BT,
        is_varlen=is_varlen,
        num_core=num_core,
        kernel_kwargs=dict(
            A=A_arg,
            beta=beta_arg,
            dA_acc=dA_acc,
            db_acc=db_acc,
            dA=dA,
            db=db,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            BH=bh_total,
            HV=HV,
            BT=BT,
            IS_VARLEN=is_varlen,
            USE_GRAPH=use_graph,
            G_T_CONTIG=g_t_contig,
        ),
    )

    dv = dv2
    return dq, dk, dv, db, dg, dA
