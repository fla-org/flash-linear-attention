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

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_NUM_WARPS = 2
_NUM_WARPS_FWD = 4
# recompute_w_u_fwd: b_A[BT,BT], b_vb[BT,BV], b_kb[BT,BK]
_RECOMPUTE_FWD_MEM_MULT = 6.0
# prepare_wy_repr_bwd: multiple [BT,BT] fp32 tiles + [BT,BK/BV] tiles
_PREPARE_BWD_MEM_MULT = 18.0
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_FWD = 64
_MAX_TILE_BWD = 32


def _get_fwd_tiles(BT: int, K: int, V: int) -> tuple[int, int]:
    BK = compute_row_tile_block_size(
        BT, K, _RECOMPUTE_FWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_FWD, triton.next_power_of_2(K)),
    )
    BV = compute_row_tile_block_size(
        BT, V, _RECOMPUTE_FWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_FWD, triton.next_power_of_2(V)),
    )
    return BK, BV


def _get_bwd_tiles(BT: int, K: int, V: int) -> tuple[int, int]:
    BK = compute_row_tile_block_size(
        BT, K, _PREPARE_BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_BWD, triton.next_power_of_2(K)),
    )
    BV = compute_row_tile_block_size(
        BT, V, _PREPARE_BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_BWD, triton.next_power_of_2(V)),
    )
    return BK, BV


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


@triton.jit(do_not_specialize=['T'])
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
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = exp2(tl.load(p_g, boundary_check=(0,)))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_w = tl.make_block_ptr(w + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_k_npu(
    k, beta, g, A, dw, dk, dA_scr, db, dg,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_dA = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_exp = exp2(b_g)
        b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_dk = tl.make_block_ptr(dk + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_G:
            b_kbg = b_k * (b_b * b_g_exp)[:, None]
        else:
            b_kbg = b_k * b_b[:, None]
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A.to(b_dw.dtype), b_dw)
        if USE_G:
            b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
            b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
            b_dg += tl.sum(b_dkbg * b_kbg, 1)
        else:
            b_dk = b_dkbg * b_b[:, None]
            b_db += tl.sum(b_dkbg * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
    if USE_G:
        p_dg = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_v_npu(
    v, beta, A, du, dv, dA_scr, db,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_dA = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.load(p_db, boundary_check=(0,)).to(tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_mask_npu(
    dA_scr,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_dA = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_dot1_npu(
    A, dA_scr, dA_mid,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_in = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_out = tl.make_block_ptr(dA_mid + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.load(p_in, boundary_check=(0, 1)).to(tl.float32)
    b_out = tl.dot(b_dA, b_A.to(tl.float32))
    tl.store(p_out, b_out.to(p_out.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_dot2_npu(
    A, dA_mid, dA_out,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_in = tl.make_block_ptr(dA_mid + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_out = tl.make_block_ptr(dA_out + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.load(p_in, boundary_check=(0, 1)).to(tl.float32)
    b_dA = tl.dot(b_A.to(tl.float32), b_dA)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, -b_dA, 0)
    tl.store(p_out, b_dA.to(p_out.dtype.element_ty), boundary_check=(0, 1))


_DG_BLK = 16


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_gate_npu(
    g, dA_out,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    n_sub = BT // BC

    for r in range(n_sub):
        i_tr = i_t * BT + r * BC
        p_gr = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_tr,), (BC,), (0,))
        b_gr = tl.load(p_gr, boundary_check=(0,)).to(tl.float32)
        for c in range(n_sub):
            i_tc = i_t * BT + c * BC
            p_dA = tl.make_block_ptr(
                dA_out + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
            p_gc = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_tc,), (BC,), (0,))
            b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
            b_diff = b_gr[:, None] - b_gc[None, :]
            b_gate = exp2(b_diff)
            b_prod = b_dA * b_gate
            b_dA = tl.where(b_prod == b_prod, b_prod, 0.0)
            tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finalize_k_npu(
    k, beta, dA_out, dk, db,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_dA = tl.make_block_ptr(dA_out + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.load(p_db, boundary_check=(0,)).to(tl.float32)
    b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_dk = tl.make_block_ptr(dk + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * b_b[:, None]
        b_dkb = tl.dot(b_dA, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb), b_dA))
        b_dk += tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


_DG_ROW_BR = 16


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finalize_a2_npu(
    k, beta, a2_scr,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_a2 = tl.make_block_ptr(a2_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_b = tl.load(p_b, boundary_check=(0,))
    b_A2 = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A2 += tl.dot(b_k, tl.trans(b_k))
    b_A2 *= b_b[:, None]
    tl.store(p_a2, b_A2.to(p_a2.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finalize_dg_npu(
    dA_out, a2_scr, dg, col_acc_scr,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    n_sub = BT // BC
    col_off = (i_tg * HV + i_h) * BT
    p_col0 = tl.make_block_ptr(col_acc_scr + col_off, (BT,), (1,), (0,), (BT,), (0,))
    tl.store(p_col0, tl.zeros([BT], dtype=tl.float32), boundary_check=(0,))

    for r in range(n_sub):
        i_tr = i_t * BT + r * BC
        p_dg_r = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_tr,), (BC,), (0,))
        b_dg_r = tl.load(p_dg_r, boundary_check=(0,)).to(tl.float32)
        for c in range(n_sub):
            p_dA = tl.make_block_ptr(
                dA_out + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            p_a2 = tl.make_block_ptr(
                a2_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
            b_a2 = tl.load(p_a2, boundary_check=(0, 1)).to(tl.float32)
            prod = b_dA * b_a2
            b_dg_r += tl.sum(prod, axis=1)
            p_col = tl.make_block_ptr(
                col_acc_scr + col_off, (BT,), (1,), (c * BC,), (BC,), (0,),
            )
            b_col = tl.load(p_col, boundary_check=(0,)).to(tl.float32)
            b_col += tl.sum(prod, axis=0)
            tl.store(p_col, b_col.to(p_col.dtype.element_ty), boundary_check=(0,))
        tl.store(p_dg_r, b_dg_r.to(p_dg_r.dtype.element_ty), boundary_check=(0,))

    p_dg = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_col = tl.make_block_ptr(col_acc_scr + col_off, (BT,), (1,), (0,), (BT,), (0,))
    b_dg = tl.load(p_dg, boundary_check=(0,)).to(tl.float32)
    b_col = tl.load(p_col, boundary_check=(0,)).to(tl.float32)
    b_dg -= b_col
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


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
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK, BV = _get_fwd_tiles(BT, K, V)
    use_g = g is not None
    is_varlen = cu_seqlens is not None
    g_arg = g if use_g else beta

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    _launch_wy_kernel(
        recompute_w_u_fwd_kernel_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            k=k,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            g=g_arg,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_G=use_g,
            IS_VARLEN=is_varlen,
            num_warps=_NUM_WARPS_FWD,
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
    BK, BV = _get_bwd_tiles(BT, K, V)
    use_g = g is not None
    is_varlen = cu_seqlens is not None

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if use_g else None
    db = torch.empty_like(beta)
    g_arg = g if use_g else beta
    dg_arg = dg if use_g else beta
    dA_scr = torch.zeros_like(A, dtype=torch.float32)
    dA_mid = torch.zeros_like(A, dtype=torch.float32)
    dA_out = torch.zeros_like(A, dtype=torch.float32)
    a2_scr = torch.zeros_like(A, dtype=torch.float32)
    col_acc_scr = torch.zeros(B, triton.cdiv(T, BT) if cu_seqlens is None else len(
        chunk_indices), HV, BT, dtype=torch.float32, device=k.device)

    base = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BT,
        IS_VARLEN=is_varlen,
        num_warps=_NUM_WARPS,
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_k_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            k=k, beta=beta, g=g_arg, A=A, dw=dw,
            dk=dk, dA_scr=dA_scr, db=db, dg=dg_arg,
            H=H, HV=HV, K=K, BK=BK, USE_G=use_g,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_v_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            v=v, beta=beta, A=A, du=du, dv=dv, dA_scr=dA_scr, db=db,
            HV=HV, V=V, BV=BV,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_mask_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            dA_scr=dA_scr,
            HV=HV,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_dot1_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A, dA_scr=dA_scr, dA_mid=dA_mid,
            HV=HV,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_dot2_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A, dA_mid=dA_mid, dA_out=dA_out,
            HV=HV,
            **base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_da_gate_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                g=g_arg, dA_out=dA_out,
                HV=HV, BC=_DG_BLK,
                **base,
            ),
        )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_finalize_k_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            k=k, beta=beta, dA_out=dA_out, dk=dk, db=db,
            H=H, HV=HV, K=K, BK=BK,
            **base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_a2_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                k=k, beta=beta, a2_scr=a2_scr,
                H=H, HV=HV, K=K, BK=BK,
                **base,
            ),
        )
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_dg_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                dA_out=dA_out, a2_scr=a2_scr, dg=dg_arg, col_acc_scr=col_acc_scr,
                HV=HV, BC=_DG_BLK,
                **base,
            ),
        )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg
