# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""WY-representation kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from fla.ops.common.backends.triton_ascend.chunk_delta_h import mload, mstore, need_dma_mask
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


# prepare_wy_repr_bwd stage-specific UB models
# finalize_k / a2 keep a conservative 4.5× slab (BK=256 overflows 192KB UB).
# kv at BT=64, BK=BV=256 asks for 2627584 bits and overflows the 192KB UB.
# 4.5× selects BK=BV=128 there, and still allows BK=256 when BT<=32.
_PREPARE_BWD_K_MEM_MULT = 4.5
_PREPARE_BWD_KV_MEM_MULT = 4.5
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_BWD = 128
_MAX_TILE_BWD_KV = 256

# recompute_w_u_fwd: peak UB is max(u-slab, w-slab), not sum — tile BK/BV independently.
# Each slab is ~3.5× BT×{BV or BK} + BT×BT (no gk/qg vs KDA).
_RECOMPUTE_FWD_MEM_MULT = 3.5
_MAX_TILE_FWD = 128
_PREFERRED_TILE = 64


def _g_npu_arg(g: torch.Tensor | None, HV: int) -> tuple[torch.Tensor | None, bool]:
    if g is None or HV == 1:
        return g, False
    return g.transpose(1, 2).contiguous(), True


def _beta_npu_arg(beta: torch.Tensor, HV: int) -> tuple[torch.Tensor, bool]:
    if HV == 1:
        return beta, False
    return beta.transpose(1, 2).contiguous(), True


def _t_npu_buf(
    B: int, T: int, HV: int, *, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, bool]:
    if HV == 1:
        return torch.empty(B, T, HV, dtype=dtype, device=device), False
    return torch.empty(B, HV, T, dtype=dtype, device=device), True


def _bwd_col_tile(BT: int, dim: int, mem_mult: float, max_tile: int) -> int:
    return compute_row_tile_block_size(
        BT, dim, mem_mult,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(max_tile, triton.next_power_of_2(dim)),
    )


def _candidate_fwd_tiles(dim: int) -> list[int]:
    cap = min(_MAX_TILE_FWD, triton.next_power_of_2(dim))
    return [b for b in (_PREFERRED_TILE, _MAX_TILE_FWD, 32, 16, _FALLBACK_TILE) if b <= cap] or [_FALLBACK_TILE]


@triton.jit
def _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN: tl.constexpr):
    if IS_VARLEN:
        return g + bos + i_h * T_seq
    return g + tl.cast(i_b, tl.int64) * HV * T_seq + i_h * T_seq


@triton.jit
def _t_row_ptr(base, T, offset, BLK, CONTIG: tl.constexpr, HV: tl.constexpr):
    if CONTIG:
        o_t = offset + tl.arange(0, BLK)
        return base + o_t
    o_t = offset + tl.arange(0, BLK)
    return base + o_t * HV


def _launch_wy_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(NT, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, NT, max_nt):
        nt_len = min(max_nt, NT - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](**kernel_kwargs)


def _launch_wy_core_grid(kernel, *, task_num: int, kernel_kwargs: dict) -> None:
    num_core = get_npu_properties()["num_aicore"]
    # disable auto-multi-buffer on this core-grid launch
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **ascend_compile_kwargs(), **kernel_kwargs)


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "USE_G": lambda args: args["g"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def recompute_w_u_fwd_kernel_npu(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    BETA_T_CONTIG: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    T_seq = T
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            i_t = i_t_o
            bos = tl.cast(i_b, tl.int64) * T_seq

        k_ptr = k + (bos * H + i_h // (HV // H)) * K
        v_ptr = v + (bos * HV + i_h) * V
        u_ptr = u + (bos * HV + i_h) * V
        w_ptr = w + (bos * HV + i_h) * K
        if BETA_T_CONTIG:
            beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        else:
            beta_ptr = beta + bos * HV + i_h
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        p_b = _t_row_ptr(beta_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, BETA_T_CONTIG, HV)
        b_b = mload(p_b, m_t, MASK_DMA)
        o_i = tl.arange(0, BT)
        m_p_A = m_t[:, None]
        p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_i[None, :]
        if USE_G:
            if G_T_CONTIG:
                g_ptr = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            else:
                g_ptr = g + bos * HV + i_h
            p_g = _t_row_ptr(g_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, G_T_CONTIG, HV)
            b_g = exp2(mload(p_g, m_t, MASK_DMA).to(tl.float32))
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = (o_v >= 0) & (o_v < V)
            m_p_v = m_t[:, None] & m_v[None, :]
            p_v = v_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
            p_u = u_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
            b_v = mload(p_v, m_p_v, MASK_DMA)
            b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
            # Ascend tl.dot may clobber the left operand; reload A each V tile.
            b_A = mload(p_A, m_p_A, MASK_DMA)
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            mstore(p_u, b_u.to(p_u.dtype.element_ty), m_p_v, MASK_DMA)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = (o_k >= 0) & (o_k < K)
            m_p_k = m_t[:, None] & m_k[None, :]
            p_k = k_ptr + o_t[:, None] * (H * K) + o_k[None, :]
            p_w = w_ptr + o_t[:, None] * (HV * K) + o_k[None, :]
            b_k = mload(p_k, m_p_k, MASK_DMA)
            b_kb = b_k * b_b[:, None]
            if USE_G:
                b_kb = b_kb * b_g[:, None]
            # Ascend tl.dot may clobber the left operand; reload A each K tile.
            b_A = mload(p_A, m_p_A, MASK_DMA)
            b_w = tl.dot(b_A, b_kb.to(b_k.dtype), allow_tf32=False)
            mstore(p_w, b_w.to(p_w.dtype.element_ty), m_p_k, MASK_DMA)


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def prepare_wy_repr_bwd_kv_npu(
    k, v, beta, g, A, dw, du, dA, dk, dv, db, dg,
    cu_seqlens, chunk_indices, T, B,
    task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr, BETA_T_CONTIG: tl.constexpr,
    DG_T_CONTIG: tl.constexpr, DB_T_CONTIG: tl.constexpr,
    G_EXP_PRECOMP: tl.constexpr, MASK_DMA: tl.constexpr,
):
    """K/V backward stage on a 1D Cube core-grid.

    Flatten (chunk, head) tasks so large NT·B·HV does not host-split at
    ASCEND_MAX_GRID_DIM. Rebind local pointers every task iteration.
    """
    T_seq = T
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            i_t = i_t_o
            bos = tl.cast(i_b, tl.int64) * T_seq

        if BETA_T_CONTIG:
            beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_b = _t_row_ptr(beta_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
        else:
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_b = beta + (bos * HV + i_h) + o_t * HV
        if DB_T_CONTIG:
            db_ptr = _g_contig_base(db, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_db = _t_row_ptr(db_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
        else:
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_db = db + (bos * HV + i_h) + o_t * HV
        o_i = tl.arange(0, BT)
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        # A is stored with stride-1 on the row axis. Load the contiguous
        # transpose and flip it in UB; a stride-(HV*BT) gather is much slower.
        # The BT axis is a full tile; only T can run past the sequence.
        m_p_AT = m_t[:, None]
        p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_i[None, :]

        b_b = mload(p_b, m_t, MASK_DMA).to(tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_A = tl.trans(mload(p_A, m_p_AT, MASK_DMA)).to(tl.float32)

        if USE_G:
            if G_T_CONTIG:
                g_ptr = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
                p_g = _t_row_ptr(g_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
            else:
                p_g = g + (bos * HV + i_h) + o_t * HV
            b_g = mload(p_g, m_t, MASK_DMA).to(tl.float32)
            b_g_exp = b_g if G_EXP_PRECOMP else exp2(b_g)
            b_bg = b_b * b_g_exp
            b_dg = tl.zeros([BT], dtype=tl.float32)

        k_ptr = k + (bos * H + i_h // (HV // H)) * K
        dk_ptr = dk + (bos * HV + i_h) * K
        dw_ptr = dw + (bos * HV + i_h) * K
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = (o_k >= 0) & (o_k < K)
            m_p_k = m_t[:, None] & m_k[None, :]
            p_k = k_ptr + o_t[:, None] * (H * K) + o_k[None, :]
            p_dk = dk_ptr + o_t[:, None] * (HV * K) + o_k[None, :]
            p_dw = dw_ptr + o_t[:, None] * (HV * K) + o_k[None, :]
            b_k = mload(p_k, m_p_k, MASK_DMA)
            b_dw = mload(p_dw, m_p_k, MASK_DMA)
            # First dot clobbers b_dw; keep a copy for the dk dot.
            b_dw_c = b_dw + 0.0
            b_A_c = b_A.to(b_dw.dtype) + 0.0
            if USE_G:
                b_kbg = (b_k.to(tl.float32) * b_bg[:, None]).to(b_k.dtype)
            else:
                b_kbg = (b_k.to(tl.float32) * b_b[:, None]).to(b_k.dtype)
            # Explicit transpose, then dot. Nested tl.trans() inside tl.dot
            # miscompiles in this kernel on triton-ascend.
            b_kbg_T = tl.trans(b_kbg)
            b_dA = tl.dot(b_dw, b_kbg_T, b_dA, allow_tf32=False)
            b_dkbg = tl.dot(b_A_c, b_dw_c, allow_tf32=False).to(tl.float32)
            b_k_f = b_k.to(tl.float32)
            if USE_G:
                b_dk = b_dkbg * b_bg[:, None]
                b_db += b_g_exp * tl.sum(b_dkbg * b_k_f, 1)
                b_dg += tl.sum(b_dkbg * b_kbg.to(tl.float32), 1)
            else:
                b_dk = b_dkbg * b_b[:, None]
                b_db += tl.sum(b_dkbg * b_k_f, 1)
            mstore(p_dk, b_dk.to(p_dk.dtype.element_ty), m_p_k, MASK_DMA)

        v_ptr = v + (bos * HV + i_h) * V
        dv_ptr = dv + (bos * HV + i_h) * V
        du_ptr = du + (bos * HV + i_h) * V
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = (o_v >= 0) & (o_v < V)
            m_p_v = m_t[:, None] & m_v[None, :]
            p_v = v_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
            p_dv = dv_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
            p_du = du_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
            b_v = mload(p_v, m_p_v, MASK_DMA)
            b_du = mload(p_du, m_p_v, MASK_DMA)
            b_du_c = b_du + 0.0
            b_A_c = b_A.to(b_du.dtype) + 0.0
            b_vb = (b_v.to(tl.float32) * b_b[:, None]).to(b_v.dtype)
            b_vb_T = tl.trans(b_vb)
            b_dA = tl.dot(b_du, b_vb_T, b_dA, allow_tf32=False)
            b_dvb = tl.dot(b_A_c, b_du_c, allow_tf32=False).to(tl.float32)
            b_v_f = b_v.to(tl.float32)
            b_dv = b_dvb * b_b[:, None]
            b_db += tl.sum(b_dvb * b_v_f, 1)
            mstore(p_dv, b_dv.to(p_dv.dtype.element_ty), m_p_v, MASK_DMA)

        p_dA = dA + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_i[None, :]
        mstore(p_dA, tl.trans(b_dA).to(p_dA.dtype.element_ty), m_p_AT, MASK_DMA)
        mstore(p_db, b_db.to(p_db.dtype.element_ty), m_t, MASK_DMA)
        if USE_G:
            if DG_T_CONTIG:
                dg_ptr = _g_contig_base(dg, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
                p_dg = _t_row_ptr(dg_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
            else:
                p_dg = dg + (bos * HV + i_h) + o_t * HV
            mstore(p_dg, b_dg.to(p_dg.dtype.element_ty), m_t, MASK_DMA)


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_mask_dot1_npu(
    A, dA_scr, dA_mid,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, MASK_DMA: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    # Same column-contiguous layout as da_dot2. Load the transpose and compute
    # (dA @ A)^T = A^T @ dA^T so the dot sees a stride-1 K axis.
    o_i = tl.arange(0, BT)
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T
    if MASK_DMA:
        o_col = tl.minimum(o_t, tl.maximum(T - 1, 0))
        scale = tl.where(m_t, 1.0, 0.0)
    else:
        o_col = o_t
    base = (bos * HV + i_h) * BT
    p_A = A + base + o_col[:, None] * (HV * BT) + o_i[None, :]
    p_in = dA_scr + base + o_col[:, None] * (HV * BT) + o_i[None, :]
    p_out = dA_mid + base + o_t[:, None] * (HV * BT) + o_i[None, :]
    b_AT = tl.load(p_A).to(tl.float32)
    b_dAT = tl.load(p_in).to(tl.float32)
    if MASK_DMA:
        b_AT = b_AT * scale[:, None]
        b_dAT = b_dAT * scale[:, None]
    m_T = (o_i[None, :] > o_i[:, None]) & (m_t[:, None] & m_t[None, :])
    b_dAT = tl.where(m_T, b_dAT, 0)
    b_outT = tl.dot(b_AT, b_dAT, allow_tf32=False)
    if MASK_DMA:
        tl.store(p_out, b_outT.to(p_out.dtype.element_ty), mask=m_t[:, None])
    else:
        tl.store(p_out, b_outT.to(p_out.dtype.element_ty))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_dot2_npu(
    A, dA_mid, dA_out,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, MASK_DMA: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
        eos = bos + T

    # Memory layout is column-contiguous: logical[i, j] lives at i + j*(HV*BT).
    # Loading that tile with j as the last pointer axis is a stride-(HV*BT)
    # gather and makes this dot ~7x slower than the block-pointer version.
    # Load j as the contiguous axis (so the tile is the transpose) and compute
    # (A @ dA)^T = dA^T @ A^T, which stores back into the same layout.
    o_i = tl.arange(0, BT)
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T
    if MASK_DMA:
        o_col = tl.minimum(o_t, tl.maximum(T - 1, 0))
        scale = tl.where(m_t, 1.0, 0.0)
    else:
        o_col = o_t
    base = (bos * HV + i_h) * BT
    p_A = A + base + o_col[:, None] * (HV * BT) + o_i[None, :]
    p_in = dA_mid + base + o_col[:, None] * (HV * BT) + o_i[None, :]
    p_out = dA_out + base + o_t[:, None] * (HV * BT) + o_i[None, :]
    b_AT = tl.load(p_A).to(tl.float32)
    b_dAT = tl.load(p_in).to(tl.float32)
    if MASK_DMA:
        b_AT = b_AT * scale[:, None]
        b_dAT = b_dAT * scale[:, None]
    b_outT = tl.dot(b_dAT, b_AT, allow_tf32=False)
    m_T = (o_i[None, :] > o_i[:, None]) & (m_t[:, None] & m_t[None, :])
    b_outT = tl.where(m_T, -b_outT, 0)
    if MASK_DMA:
        tl.store(p_out, b_outT.to(dA_out.dtype.element_ty), mask=m_t[:, None])
    else:
        tl.store(p_out, b_outT.to(dA_out.dtype.element_ty))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_gate_npu(
    g, dA_out,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, G_T_CONTIG: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    if G_T_CONTIG:
        g_ptr = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        p_g = _t_row_ptr(g_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
    else:
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        p_g = g + (bos * HV + i_h) + o_t * HV
    # Row block covers the whole row shape; only the time axis can be OOB.
    o_i = tl.arange(0, BT)
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_col = m_t[None, :]
    p_dA = dA_out + (bos * HV + i_h) * BT + o_i[:, None] + o_t[None, :] * (HV * BT)
    b_g = mload(p_g, m_t, MASK_DMA).to(tl.float32)
    b_dA = mload(p_dA, m_col, MASK_DMA).to(tl.float32)
    b_prod = b_dA * exp2(b_g[:, None] - b_g[None, :])
    b_dA = tl.where(b_prod == b_prod, b_prod, 0.0)
    mstore(p_dA, b_dA.to(p_dA.dtype.element_ty), m_col, MASK_DMA)


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def prepare_wy_repr_bwd_finalize_k_npu(
    k, beta, dA_out, dk, db,
    cu_seqlens, chunk_indices, T, B,
    task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, BETA_T_CONTIG: tl.constexpr, DB_T_CONTIG: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    T_seq = T
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            i_t = i_t_o
            bos = tl.cast(i_b, tl.int64) * T_seq

        if BETA_T_CONTIG:
            beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_b = _t_row_ptr(beta_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
        else:
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_b = beta + (bos * HV + i_h) + o_t * HV
        if DB_T_CONTIG:
            db_ptr = _g_contig_base(db, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_db = _t_row_ptr(db_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
        else:
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t = (o_t >= 0) & (o_t < T)
            p_db = db + (bos * HV + i_h) + o_t * HV
        o_i = tl.arange(0, BT)
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        m_p_dA = m_t[None, :]
        p_dA = dA_out + (bos * HV + i_h) * BT + o_i[:, None] + o_t[None, :] * (HV * BT)

        b_b = mload(p_b, m_t, MASK_DMA).to(tl.float32)
        b_db = mload(p_db, m_t, MASK_DMA).to(tl.float32)
        b_dA = mload(p_dA, m_p_dA, MASK_DMA).to(tl.float32)
        b_dA_c = b_dA + 0.0

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = (o_k >= 0) & (o_k < K)
            m_p_k = m_t[:, None] & m_k[None, :]
            p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
            p_dk = dk + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
            b_k = mload(p_k, m_p_k, MASK_DMA).to(tl.float32)
            b_kb = b_k * b_b[:, None]
            # Ascend tl.dot clobbers lhs; keep a pristine copy for the rhs dot.
            b_dA_lhs = b_dA + 0.0
            b_dkb = tl.dot(b_dA_lhs, b_k, allow_tf32=False)
            b_db += tl.sum(b_dkb * b_k, 1)
            b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb), b_dA_c, allow_tf32=False))
            b_dk += mload(p_dk, m_p_k, MASK_DMA).to(tl.float32)
            mstore(p_dk, b_dk.to(p_dk.dtype.element_ty), m_p_k, MASK_DMA)

        mstore(p_db, b_db.to(p_db.dtype.element_ty), m_t, MASK_DMA)


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finalize_a2_dg_npu(
    k, beta, dA_out, dg,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, BETA_T_CONTIG: tl.constexpr, DG_T_CONTIG: tl.constexpr,
    MASK_DMA: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    """Fuse A2 = (k k^T) * beta with dg += row(dA*A2) - col(dA*A2). Keep A2 in UB."""
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T

    if BETA_T_CONTIG:
        beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        p_b = _t_row_ptr(beta_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
    else:
        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        m_t = (o_t >= 0) & (o_t < T)
        p_b = beta + (bos * HV + i_h) + o_t * HV
    o_i = tl.arange(0, BT)
    o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
    m_t = (o_t >= 0) & (o_t < T)
    m_p_dA = m_t[None, :]
    p_dA = dA_out + (bos * HV + i_h) * BT + o_i[:, None] + o_t[None, :] * (HV * BT)
    if DG_T_CONTIG:
        dg_ptr = _g_contig_base(dg, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        p_dg = _t_row_ptr(dg_ptr, T, tl.cast(i_t, tl.int64) * BT, BT, True, HV)
    else:
        p_dg = dg + (bos * HV + i_h) + o_t * HV

    b_b = mload(p_b, m_t, MASK_DMA).to(tl.float32)
    b_A2 = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = (o_k >= 0) & (o_k < K)
        m_p_k = m_t[:, None] & m_k[None, :]
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = mload(p_k, m_p_k, MASK_DMA).to(tl.float32)
        b_k_c = b_k + 0.0
        b_A2 = tl.dot(b_k, tl.trans(b_k_c), b_A2, allow_tf32=False)
    b_A2 *= b_b[:, None]
    b_dA = mload(p_dA, m_p_dA, MASK_DMA).to(tl.float32)
    b_prod = b_dA * b_A2
    b_dg = mload(p_dg, m_t, MASK_DMA).to(tl.float32)
    b_dg += tl.sum(b_prod, axis=1) - tl.sum(b_prod, axis=0)
    mstore(p_dg, b_dg.to(p_dg.dtype.element_ty), m_t, MASK_DMA)


@input_guard
def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Minimize V/K slab iterations under independent UB budgets for u- and w-slabs.
    max_bk = _bwd_col_tile(BT, K, _RECOMPUTE_FWD_MEM_MULT, _MAX_TILE_FWD)
    max_bv = _bwd_col_tile(BT, V, _RECOMPUTE_FWD_MEM_MULT, _MAX_TILE_FWD)
    BK = max(8, min(max_bk, triton.next_power_of_2(K)))
    BV = max(8, min(max_bv, triton.next_power_of_2(V)))
    # Dividing tiles keep the kernel on static DMA. A partial last slab masks every load.
    k_tiles = [b for b in _candidate_fwd_tiles(K) if b <= max_bk and K % b == 0]
    v_tiles = [b for b in _candidate_fwd_tiles(V) if b <= max_bv and V % b == 0]
    if not k_tiles:
        k_tiles = [b for b in _candidate_fwd_tiles(K) if b <= max_bk]
    if not v_tiles:
        v_tiles = [b for b in _candidate_fwd_tiles(V) if b <= max_bv]
    best_cost = None
    for bk in k_tiles:
        for bv in v_tiles:
            cost = triton.cdiv(V, bv) + triton.cdiv(K, bk)
            if best_cost is None or cost < best_cost or (cost == best_cost and bk + bv > BK + BV):
                best_cost, BK, BV = cost, bk, bv

    u = torch.empty_like(v)
    w = k.new_empty(B, T, HV, K)
    beta, beta_t_contig = _beta_npu_arg(beta, HV)
    g, g_t_contig = _g_npu_arg(g, HV)

    _launch_wy_core_grid(
        recompute_w_u_fwd_kernel_npu,
        task_num=NT * B * HV,
        kernel_kwargs=dict(
            k=k,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            G_T_CONTIG=g_t_contig,
            BETA_T_CONTIG=beta_t_contig,
            MASK_DMA=need_dma_mask(T, BT, K, BK, V, BV, cu_seqlens is not None),
        ),
    )
    return w, u


def prepare_wy_repr_bwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK, BV = _bwd_col_tile(BT, K, _PREPARE_BWD_KV_MEM_MULT, _MAX_TILE_BWD_KV), _bwd_col_tile(
        BT, V, _PREPARE_BWD_KV_MEM_MULT, _MAX_TILE_BWD_KV)
    BK_FIN = _bwd_col_tile(BT, K, _PREPARE_BWD_K_MEM_MULT, _MAX_TILE_BWD)
    use_g = g is not None
    is_varlen = cu_seqlens is not None

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    db, db_t_contig = _t_npu_buf(B, T, HV, dtype=beta.dtype, device=k.device)
    beta_arg, beta_t_contig = _beta_npu_arg(beta, HV)
    dg, dg_t_contig = None, False
    g_gate, g_t_contig = None, False
    g_k_arg = k
    g_exp_precomp = False
    if use_g:
        dg, dg_t_contig = _t_npu_buf(B, T, HV, dtype=g.dtype, device=k.device)
        g_gate, g_t_contig = _g_npu_arg(g, HV)
        g_k_arg = g_gate
        if not is_varlen:
            g_k_arg = torch.exp2(g_gate.float()).to(g_gate.dtype)
            g_exp_precomp = True
    dg_arg = dg if use_g else beta
    dA_scr = torch.empty_like(A, dtype=torch.float32)
    dA_mid = torch.empty_like(A, dtype=torch.float32)
    dA_out = torch.empty_like(A, dtype=torch.float32)

    mask_dma = need_dma_mask(T, BT, K, BK, V, BV, is_varlen)
    mask_fin = need_dma_mask(T, BT, K, BK_FIN, K, BK_FIN, is_varlen)
    mask_time = is_varlen or T % BT != 0
    base = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BT,
        IS_VARLEN=is_varlen,
    )
    core_base = dict(B=B, **base)
    task_num = NT * B * HV
    _launch_wy_core_grid(
        prepare_wy_repr_bwd_kv_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            k=k, v=v, beta=beta_arg, g=g_k_arg, A=A, dw=dw, du=du, dA=dA_scr,
            dk=dk, dv=dv, db=db, dg=dg_arg,
            H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, USE_G=use_g,
            G_T_CONTIG=g_t_contig, BETA_T_CONTIG=beta_t_contig,
            DG_T_CONTIG=dg_t_contig, DB_T_CONTIG=db_t_contig,
            G_EXP_PRECOMP=g_exp_precomp,
            MASK_DMA=mask_dma,
            **core_base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_mask_dot1_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A, dA_scr=dA_scr, dA_mid=dA_mid,
            HV=HV, MASK_DMA=mask_time,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_dot2_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A, dA_mid=dA_mid, dA_out=dA_out,
            HV=HV, MASK_DMA=mask_time,
            **base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_da_gate_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                g=g_gate, dA_out=dA_out,
                HV=HV, G_T_CONTIG=g_t_contig,
                MASK_DMA=mask_time,
                **base,
            ),
        )
    _launch_wy_core_grid(
        prepare_wy_repr_bwd_finalize_k_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            k=k, beta=beta_arg, dA_out=dA_out, dk=dk, db=db,
            H=H, HV=HV, K=K, BK=BK_FIN, BETA_T_CONTIG=beta_t_contig, DB_T_CONTIG=db_t_contig,
            MASK_DMA=mask_fin,
            **core_base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_a2_dg_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                k=k, beta=beta_arg, dA_out=dA_out, dg=dg_arg,
                H=H, HV=HV, K=K, BK=BK_FIN,
                BETA_T_CONTIG=beta_t_contig, DG_T_CONTIG=dg_t_contig,
                MASK_DMA=mask_fin,
                **base,
            ),
        )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    if db_t_contig:
        db = db.transpose(1, 2).contiguous()
    if use_g and dg_t_contig:
        dg = dg.transpose(1, 2).contiguous()
    return dk, dv, db, dg
