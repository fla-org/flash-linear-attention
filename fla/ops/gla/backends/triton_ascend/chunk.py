# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GLA chunk kernels for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_BC = 16
_LAUNCH_BLOCK_BUDGET = 4096
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_FALLBACK = 16
_MAX_TILE = 64


def _npu_sync() -> None:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize()


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC, K, 6.0, tiling_row=False, safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK, min_block=16,
        max_block=min(64, max(16, triton.next_power_of_2(K))),
    )


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _launch_3d_nt_nc_bh(kernel, *, nt: int, nc: int, bh: int, kernel_kwargs: dict) -> None:
    budget = _LAUNCH_BLOCK_BUDGET
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    nt_step = nt if nt * nc * bh <= budget else max(1, budget // max(nc * bh, 1))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        nc_budget = max(1, budget // max(nt_len * bh, 1))
        nc_step = min(nc_budget, max_grid_axis_chunks(nc, nt_len * bh, max_grid=ASCEND_MAX_GRID_DIM))
        for nc_off in range(0, nc, nc_step):
            nc_len = min(nc_step, nc - nc_off)
            kernel_kwargs['NC_OFFSET'] = nc_off
            bh_budget = max(1, budget // max(nt_len * nc_len, 1))
            bh_step = min(bh_budget, max_grid_axis_chunks(bh, nt_len * nc_len, max_grid=ASCEND_MAX_GRID_DIM))
            for bh_off in range(0, bh, bh_step):
                bh_len = min(bh_step, bh - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nt_len, nc_len, bh_len)](**kernel_kwargs)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gla_fwd_A_kernel_intra_sub_inter_npu(
    q, k, g, A, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr, NT_OFFSET, NC_OFFSET, BH_OFFSET,
):
    i_t = tl.program_id(0).to(tl.int64) + NT_OFFSET
    i_c = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    o_i = i_t * BT + i_i * BC + tl.arange(0, BC)
    o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
    m_i = o_i < T
    m_j = o_j < T
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_i[:, None] & m_k[None, :]
        m_kj = m_k[:, None] & m_j[None, :]

        p_q = q + (bos * H + i_h) * K + o_i[:, None] * (H * K) + o_k[None, :]
        p_g = g + (bos * H + i_h) * K + o_i[:, None] * (H * K) + o_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_k[:, None] + o_j[None, :] * (H * K)
        p_gk = g + (bos * H + i_h) * K + o_k[:, None] + o_j[None, :] * (H * K)
        p_gn = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=m_qk, other=0.0).to(tl.float32)
        b_g = tl.load(p_g, mask=m_qk, other=0.0).to(tl.float32)
        b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, mask=m_kj, other=0.0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_kj, other=0.0).to(tl.float32)
        b_kg = b_k * exp2(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg, allow_tf32=False)

    o_jA = i_j * BC + tl.arange(0, BC)
    m_A = m_i[:, None] & (o_jA[None, :] < BT)
    p_A = A + (bos * H + i_h) * BT + o_i[:, None] * (H * BT) + o_jA[None, :]
    tl.store(p_A, b_A.to(A.dtype.element_ty), mask=m_A)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_npu(
    q, k, g, A, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, NT_OFFSET, NC_OFFSET, BH_OFFSET,
):
    i_t = tl.program_id(0).to(tl.int64) + NT_OFFSET
    i_i = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) * H * BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    q_ptr = q + (bos * H + i_h) * K
    k_ptr = k + (bos * H + i_h) * K
    g_ptr = g + (bos * H + i_h) * K
    A_ptr = A + (bos * H + i_h) * BT

    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_qk = m_A[:, None] & m_k[None, :]
    b_q = tl.load(q_ptr + o_c[:, None] * (H * K) + o_k[None, :], mask=m_qk, other=0.0).to(tl.float32)
    b_g = tl.load(g_ptr + o_c[:, None] * (H * K) + o_k[None, :], mask=m_qk, other=0.0).to(tl.float32)

    # Diagonal lower-tri via masked static loop (no continue).
    max_j = min(BC, T - i_t * BT - i_i * BC)
    for j in tl.static_range(BC):
        active = j < max_j
        b_k = tl.load(
            k_ptr + (i_t * BT + i_j * BC + j) * H * K + o_k,
            mask=m_k & active, other=0,
        ).to(tl.float32)
        b_gk = tl.load(
            g_ptr + (i_t * BT + i_j * BC + j) * H * K + o_k,
            mask=m_k & active, other=0,
        ).to(tl.float32)
        b_Aj = tl.sum(b_q * b_k[None, :] * exp2(b_g - b_gk[None, :]), 1) * scale
        tl.store(A_ptr + o_A + j, b_Aj, mask=m_A & active)

    tl.debug_barrier()
    # Zero strict upper triangle of the BC×BC diagonal block (keep causal i>=j).
    b_zero = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(
        A_ptr + o_A[:, None] + o_i,
        b_zero,
        mask=m_A[:, None] & (o_i[:, None] < o_i),
    )


@input_guard
def chunk_gla_fwd_intra_gk_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size
    assert K <= 256, "NPU GLA intra currently supports K <= 256"

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(_BC, BT)
    NC = triton.cdiv(BT, BC)
    # Inter tiles K; diag mirrors CUDA K<=256 path: load full [BC, K] at once.
    BK_inter = _get_bk(K)
    BK_diag = max(triton.next_power_of_2(K), 16)

    # Partial writes (lower-tri + diag); zeros avoid dirty upper-tri NaNs.
    A = q.new_zeros(B, T, H, BT, dtype=torch.float)
    base = dict(
        q=q, k=k, g=g, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, H=H, K=K, BT=BT, BC=BC,
        NT_OFFSET=0, NC_OFFSET=0, BH_OFFSET=0,
    )
    _launch_3d_nt_nc_bh(
        chunk_gla_fwd_A_kernel_intra_sub_inter_npu,
        nt=NT, nc=NC * NC, bh=B * H,
        kernel_kwargs={**base, 'BK': BK_inter, 'NC': NC},
    )
    _launch_3d_nt_nc_bh(
        chunk_gla_fwd_A_kernel_intra_sub_intra_npu,
        nt=NT, nc=NC, bh=B * H,
        kernel_kwargs={**base, 'BK': BK_diag},
    )
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize()
    return A

# ---------------------------------------------------------------------------
# Forward: o  (1D core-grid + host-pick BK; pattern from common/chunk_o.py)
# ---------------------------------------------------------------------------


_FWD_O_BV = 128
_FWD_O_MEM_MULT = 6.0


def _get_fwd_o_bk(K: int) -> int:
    """Host-pick K tile for fwd-o; avoids Ascend parallel autotune flake in pytest."""
    return compute_row_tile_block_size(
        64,
        K,
        _FWD_O_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=32,
        max_block=min(128, triton.next_power_of_2(K)),
    )


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'total_chunks', 'task_num', 'num_core'])
def chunk_gla_fwd_kernel_o_npu(
    q, v, g, h, o, A, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    total_chunks, task_num, num_core,
    STATE_V_FIRST: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    core_id = tl.program_id(0)
    h_t_step = HV * total_chunks
    for task_id in tl.range(core_id, task_num, num_core):
        i_v = task_id // h_t_step
        remainder = task_id % h_t_step
        i_hv = remainder // total_chunks
        global_t = remainder % total_chunks
        i_h = i_hv // (HV // H)
        T_cur = T

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + global_t * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + global_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_cur = (eos - bos).to(tl.int32)
            i_tg = global_t.to(tl.int64)
        else:
            NT = tl.cdiv(T, BT)
            i_b = global_t // NT
            i_t = (global_t % NT).to(tl.int32)
            bos = (i_b * T).to(tl.int64)
            i_tg = global_t.to(tl.int64)

        q_ptr = q + (bos * H + i_h) * K
        g_ptr = g + (bos * HV + i_hv) * K
        v_ptr = v + (bos * HV + i_hv) * V
        o_ptr = o + (bos * HV + i_hv) * V
        h_base = h + (i_tg * HV + i_hv).to(tl.int64) * K * V
        a_ptr = A + (bos * HV + i_hv) * BT

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T_cur, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_base, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            b_h = tl.load(p_h, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

        b_o *= scale
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T_cur
        p_a = tl.make_block_ptr(a_ptr, (T_cur, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_v = tl.make_block_ptr(v_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_A = tl.load(p_a, boundary_check=(0, 1))
        m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
        b_A = tl.where(m_s & (m_t[:, None] & m_t[None, :]), b_A, 0.0)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_A.to(b_v.dtype), b_v)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_gla_fwd_o_gk_npu(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    assert K <= 256 and V <= 256, "NPU GLA o currently supports K,V <= 256"

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        total_chunks = B * triton.cdiv(T, BT)
    else:
        total_chunks = len(chunk_indices)

    o = torch.zeros_like(v)
    BV = min(_FWD_O_BV, triton.next_power_of_2(V))
    BK = _get_fwd_o_bk(K)
    NV = triton.cdiv(V, BV)
    num_core = get_npu_properties()['num_aicore']
    task_num = NV * HV * total_chunks
    chunk_gla_fwd_kernel_o_npu[(num_core,)](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        total_chunks=total_chunks,
        task_num=task_num,
        num_core=num_core,
        STATE_V_FIRST=state_v_first,
    )
    return o

# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _bwd_pick_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC, K, 8.0, tiling_row=False, safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK, min_block=16,
        max_block=min(_MAX_TILE, max(16, triton.next_power_of_2(K))),
    )


def _bwd_pick_bv(V: int) -> int:
    return compute_row_tile_block_size(
        64, V, 8.0, tiling_row=False, safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK, min_block=16,
        max_block=min(_MAX_TILE, max(16, triton.next_power_of_2(V))),
    )


def _launch_2d(kernel, *, nt: int, bh: int, kernel_kwargs: dict) -> None:
    budget = _LAUNCH_BLOCK_BUDGET
    nt_step = nt if nt * bh <= budget else max(1, budget // max(bh, 1))
    for nt_off in range(0, nt, nt_step):
        nt_len = min(nt_step, nt - nt_off)
        kernel_kwargs['NT_OFFSET'] = nt_off
        bh_budget = max(1, budget // max(nt_len, 1))
        bh_step = min(bh_budget, max_grid_axis_chunks(bh, nt_len, max_grid=ASCEND_MAX_GRID_DIM))
        for bh_off in range(0, bh, bh_step):
            bh_len = min(bh_step, bh - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](**kernel_kwargs)


def _launch_3d_na_nt_bh(kernel, *, na: int, nt: int, bh: int, kernel_kwargs: dict) -> None:
    """3D launch over (axis0, nt, bh). Does not slice chunk_indices (need global i_tg)."""
    budget = _LAUNCH_BLOCK_BUDGET
    na_step = na if na * nt * bh <= budget else max(1, budget // max(nt * bh, 1))
    for a_off in range(0, na, na_step):
        a_len = min(na_step, na - a_off)
        kernel_kwargs['A_OFFSET'] = a_off
        nt_budget = max(1, budget // max(a_len * bh, 1))
        nt_step = min(nt_budget, max_grid_axis_chunks(nt, a_len * bh, max_grid=ASCEND_MAX_GRID_DIM))
        for nt_off in range(0, nt, nt_step):
            nt_len = min(nt_step, nt - nt_off)
            kernel_kwargs['NT_OFFSET'] = nt_off
            bh_budget = max(1, budget // max(a_len * nt_len, 1))
            bh_step = min(bh_budget, max_grid_axis_chunks(bh, a_len * nt_len, max_grid=ASCEND_MAX_GRID_DIM))
            for bh_off in range(0, bh, bh_step):
                bh_len = min(bh_step, bh - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(a_len, nt_len, bh_len)](**kernel_kwargs)


# ---------------------------------------------------------------------------
# dA
# ---------------------------------------------------------------------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gla_bwd_kernel_dA_npu(
    v, do, dA, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, NT_OFFSET, BH_OFFSET,
):
    i_t = tl.program_id(0).to(tl.int64) + NT_OFFSET
    i_bh = tl.program_id(1).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    if i_t * BT >= T:
        return

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_vt = m_v[:, None] & m_t[None, :]
        b_do = tl.load(
            do + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :],
            mask=m_tv, other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v + (bos * H + i_h) * V + o_v[:, None] + o_t[None, :] * (H * V),
            mask=m_vt, other=0.0,
        ).to(tl.float32)
        b_dA += tl.dot(b_do, b_v, allow_tf32=False)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.)
    p_dA = dA + (bos * H + i_h) * BT + o_t[:, None] * (H * BT) + o_i[None, :]
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_A)


@input_guard
def chunk_gla_bwd_dA_npu(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, V = v.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BV = _bwd_pick_bv(V)
    dA = v.new_zeros(B, T, H, BT, dtype=torch.float)
    _launch_2d(
        chunk_gla_bwd_kernel_dA_npu,
        nt=NT, bh=B * H,
        kernel_kwargs=dict(
            v=v, do=do, dA=dA, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            scale=scale, T=T, H=H, V=V, BT=BT, BV=BV,
            NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    _npu_sync()
    return dA


# ---------------------------------------------------------------------------
# dv
# ---------------------------------------------------------------------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gla_bwd_kernel_dv_npu(
    k, g, A, do, dh, dv, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    A_OFFSET, NT_OFFSET, BH_OFFSET,
):
    i_v = tl.program_id(0) + A_OFFSET
    i_t = tl.program_id(1).to(tl.int64) + NT_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    m_v = o_v < V
    m_A = (o_i[:, None] < BT) & m_t[None, :]
    m_tv = m_t[:, None] & m_v[None, :]
    b_A = tl.load(
        A + (bos * H + i_h) * BT + o_i[:, None] + o_t[None, :] * (H * BT),
        mask=m_A, other=0.0,
    ).to(tl.float32)
    b_do = tl.load(
        do + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :],
        mask=m_tv, other=0.0,
    ).to(tl.float32)
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.)
    b_dv = tl.dot(b_A, b_do, allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        m_kvd = m_k[:, None] & m_v[None, :]
        b_k = tl.load(
            k + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :],
            mask=m_tk, other=0.0,
        ).to(tl.float32)
        b_gk = tl.load(
            g + (bos * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :],
            mask=m_tk, other=0.0,
        ).to(tl.float32)
        b_gn = tl.load(
            g + (bos + min(i_t * BT + BT, T) - 1) * H * K + i_h * K + o_k,
            mask=m_k, other=0,
        ).to(tl.float32)
        if STATE_V_FIRST:
            b_dh = tl.load(
                dh + (i_tg * H + i_h) * K * V + o_k[:, None] + o_v[None, :] * K,
                mask=m_kvd, other=0.0,
            ).to(tl.float32)
        else:
            b_dh = tl.load(
                dh + (i_tg * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :],
                mask=m_kvd, other=0.0,
            ).to(tl.float32)
        b_k = b_k * exp2(b_gn[None, :] - b_gk)
        b_dv += tl.dot(b_k, b_dh, allow_tf32=False)

    tl.store(
        dv + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :],
        b_dv.to(dv.dtype.element_ty), mask=m_tv,
    )


@input_guard
def chunk_gla_bwd_dv_npu(
    k: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK, BV = _bwd_pick_bk(K), _bwd_pick_bv(V)
    dv = torch.zeros_like(do)
    _launch_3d_na_nt_bh(
        chunk_gla_bwd_kernel_dv_npu,
        na=triton.cdiv(V, BV), nt=NT, bh=B * H,
        kernel_kwargs=dict(
            k=k, g=g, A=A, do=do, dh=dh, dv=dv,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
            H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, STATE_V_FIRST=state_v_first,
            A_OFFSET=0, NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    _npu_sync()
    return dv


# ---------------------------------------------------------------------------
# dqk_intra
# ---------------------------------------------------------------------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gla_bwd_kernel_intra_npu(
    q, k, g, dA, dq, dk, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr, A_OFFSET, NT_OFFSET, BH_OFFSET,
):
    # Mirror CUDA/KDA structure: `if i_i > 0` + `range(0, i_i)` (proven on Ascend).
    # Do NOT use static_range(NC)+mask gates for inter-subchunk — that path produced NaNs.
    i_kc = tl.program_id(0) + A_OFFSET
    i_t = tl.program_id(1).to(tl.int64) + NT_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_c = o_c < T
    m_ck = m_c[:, None] & m_k[None, :]
    b_g = tl.load(
        g + (bos * H + i_h) * K + o_c[:, None] * (H * K) + o_k[None, :],
        mask=m_ck, other=0.0,
    ).to(tl.float32)

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + tl.arange(0, BC)
            o_jA = i_j * BC + tl.arange(0, BC)
            m_jk = (o_j[:, None] < T) & m_k[None, :]
            m_da = m_c[:, None] & (o_jA[None, :] < BT)
            b_k = tl.load(
                k + (bos * H + i_h) * K + o_j[:, None] * (H * K) + o_k[None, :],
                mask=m_jk, other=0.0,
            ).to(tl.float32)
            b_gk = tl.load(
                g + (bos * H + i_h) * K + o_j[:, None] * (H * K) + o_k[None, :],
                mask=m_jk, other=0.0,
            ).to(tl.float32)
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            b_dA = tl.load(
                dA + (bos * H + i_h) * BT + o_c[:, None] * (H * BT) + o_jA[None, :],
                mask=m_da, other=0.0,
            ).to(tl.float32)
            b_dq += tl.dot(b_dA, b_kg, allow_tf32=False)
        b_dq *= exp2(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    o_dA = bos * H * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H * BT + i_h * BT + i_i * BC
    p_kj = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gkj = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dAj = tl.load(dA + o_dA + j, mask=m_dA, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dAj[:, None] * b_kj[None, :] * exp2(b_g - b_gkj[None, :]), 0.)
        p_kj += H * K
        p_gkj += H * K

    tl.store(
        dq + (bos * H + i_h) * K + o_c[:, None] * (H * K) + o_k[None, :],
        b_dq.to(dq.dtype.element_ty), mask=m_ck,
    )

    tl.debug_barrier()
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    NC_eff = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC_eff - 1:
        p_gn2 = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H * K + i_h * K + o_k
        b_gn2 = tl.load(p_gn2, mask=m_k, other=0).to(tl.float32)
        for i_j in range(i_i + 1, NC_eff):
            o_j = i_t * BT + i_j * BC + o_i
            o_iA = i_i * BC + tl.arange(0, BC)
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_da = (o_iA[:, None] < BT) & m_j[None, :]
            b_q = tl.load(
                q + (bos * H + i_h) * K + o_j[:, None] * (H * K) + o_k[None, :],
                mask=m_jk, other=0.0,
            ).to(tl.float32)
            b_gq = tl.load(
                g + (bos * H + i_h) * K + o_j[:, None] * (H * K) + o_k[None, :],
                mask=m_jk, other=0.0,
            ).to(tl.float32)
            b_qg = b_q * tl.where(m_j[:, None], exp2(b_gq - b_gn2[None, :]), 0)
            b_dA = tl.load(
                dA + (bos * H + i_h) * BT + o_iA[:, None] + o_j[None, :] * (H * BT),
                mask=m_da, other=0.0,
            ).to(tl.float32)
            b_dk += tl.dot(b_dA, b_qg, allow_tf32=False)
        b_dk *= exp2(b_gn2[None, :] - b_g)

    o_dA2 = bos * H * BT + (i_t * BT + i_i * BC) * H * BT + i_h * BT + i_i * BC + tl.arange(0, BC)
    p_qj = q + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gqj = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dAj = tl.load(dA + o_dA2 + j * H * BT).to(tl.float32)
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dAj[:, None] * b_qj[None, :] * exp2(b_gqj[None, :] - b_g), 0.)
        p_qj += H * K
        p_gqj += H * K

    tl.store(
        dk + (bos * H + i_h) * K + o_c[:, None] * (H * K) + o_k[None, :],
        b_dk.to(dk.dtype.element_ty), mask=m_ck,
    )


@input_guard
def chunk_gla_bwd_dqk_intra_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(_BC, BT)
    BK = min(64, triton.next_power_of_2(K))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq = torch.zeros_like(q, dtype=torch.float)
    dk = torch.zeros_like(k, dtype=torch.float)
    _launch_3d_na_nt_bh(
        chunk_gla_bwd_kernel_intra_npu,
        na=NK * NC, nt=NT, bh=B * H,
        kernel_kwargs=dict(
            q=q, k=k, g=g, dA=dA, dq=dq, dk=dk,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
            H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
            A_OFFSET=0, NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    _npu_sync()
    return dq, dk


# ---------------------------------------------------------------------------
# dqkg (inter)
# ---------------------------------------------------------------------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gla_bwd_kernel_inter_npu(
    q, k, v, g, h, do, dh, dq, dk, dq2, dk2, dg,
    cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, STATE_V_FIRST: tl.constexpr,
    A_OFFSET, NT_OFFSET, BH_OFFSET,
):
    i_k = tl.program_id(0) + A_OFFSET
    i_t = tl.program_id(1).to(tl.int64) + NT_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_tk = m_t[:, None] & m_k[None, :]

    q_base = q + (bos * H + i_h) * K
    k_base = k + (bos * H + i_h) * K
    v_base = v + (bos * H + i_h) * V
    g_base = g + (bos * H + i_h) * K
    h_base = h + (i_tg * H + i_h) * K * V
    do_base = do + (bos * H + i_h) * V
    dh_base = dh + (i_tg * H + i_h) * K * V
    dq_base = dq + (bos * H + i_h) * K
    dk_base = dk + (bos * H + i_h) * K
    dq2_base = dq2 + (bos * H + i_h) * K
    dk2_base = dk2 + (bos * H + i_h) * K
    dg_base = dg + (bos * H + i_h) * K

    b_gk = tl.load(g_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    b_gn = tl.load(g_base + (min(T, i_t * BT + BT) - 1) * H * K + o_k, mask=m_k, other=0).to(tl.float32)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_vk = m_v[:, None] & m_k[None, :]
        b_v = tl.load(v_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        b_do = tl.load(do_base + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        if STATE_V_FIRST:
            b_h = tl.load(h_base + o_v[:, None] * K + o_k[None, :], mask=m_vk, other=0.0).to(tl.float32)
            b_dh = tl.load(dh_base + o_v[:, None] * K + o_k[None, :], mask=m_vk, other=0.0).to(tl.float32)
        else:
            b_h = tl.load(h_base + o_v[:, None] + o_k[None, :] * V, mask=m_vk, other=0.0).to(tl.float32)
            b_dh = tl.load(dh_base + o_v[:, None] + o_k[None, :] * V, mask=m_vk, other=0.0).to(tl.float32)
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, b_dh, allow_tf32=False)

    b_dgk *= exp2(b_gn)
    b_dq *= scale
    b_dq = b_dq * exp2(b_gk)
    b_dk = b_dk * exp2(b_gn[None, :] - b_gk)
    b_q = tl.load(q_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    b_k = tl.load(k_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(dq_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    b_dk += tl.load(dk_base + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    tl.store(dq2_base + o_t[:, None] * (H * K) + o_k[None, :], b_dq.to(dq2.dtype.element_ty), mask=m_tk)
    tl.store(dk2_base + o_t[:, None] * (H * K) + o_k[None, :], b_dk.to(dk2.dtype.element_ty), mask=m_tk)
    tl.store(dg_base + o_t[:, None] * (H * K) + o_k[None, :], b_dg.to(dg.dtype.element_ty), mask=m_tk)


@input_guard
def chunk_gla_bwd_dqkg_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK, BV = _bwd_pick_bk(K), _bwd_pick_bv(V)
    dg = torch.zeros_like(g)
    dq2 = torch.zeros_like(dq)
    dk2 = torch.zeros_like(dk)
    _launch_3d_na_nt_bh(
        chunk_gla_bwd_kernel_inter_npu,
        na=triton.cdiv(K, BK), nt=NT, bh=B * H,
        kernel_kwargs=dict(
            q=q, k=k, v=v, g=g, h=h, do=do, dh=dh, dq=dq, dk=dk,
            dq2=dq2, dk2=dk2, dg=dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=T,
            H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, STATE_V_FIRST=state_v_first,
            A_OFFSET=0, NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    _npu_sync()
    return dq2, dk2, dg
