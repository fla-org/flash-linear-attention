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

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

_DAV_NUM_WARPS = 2
_DAV_MEM_MULT = 8.0
_DAV_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE = 64


def _get_dAv_bv(BT: int, V: int) -> int:
    return compute_row_tile_block_size(
        BT, V, _DAV_MEM_MULT,
        tiling_row=False,
        safety_margin=_DAV_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE, triton.next_power_of_2(V)),
    )


def _launch_dAv_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, nt, max_nt):
        nt_len = min(max_nt, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=_DAV_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv_npu(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT

    p_A = tl.make_block_ptr(A + (bos * HV + i_hv) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, HV * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v, allow_tf32=False)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    p_dA = tl.make_block_ptr(dA, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


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
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, HV, V = *k.shape, do.shape[2], do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BV = _get_dAv_bv(BT, V)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, HV, BT, dtype=torch.float)
    dv = torch.zeros_like(do)

    _launch_dAv_2d_kernel(
        chunk_kda_bwd_kernel_dAv_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            q=q,
            k=k,
            v=v,
            A=A,
            do=do,
            dv=dv,
            dA=dA,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            IS_VARLEN=cu_seqlens is not None,
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    return dA, dv


_BC = 16
_NUM_WARPS = 2
_BWD_MEM_MULT = 10.0
_SAFETY_MARGIN = 0.80
_FALLBACK_TILE = 16
_MAX_TILE = 128


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(K)),
    )


def _get_bv(V: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        V,
        _BWD_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=16,
        max_block=min(_MAX_TILE, triton.next_power_of_2(V)),
    )


def _launch_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict, num_warps: int = _NUM_WARPS) -> None:
    max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nt_off in range(0, nt, max_nt):
        nt_len = min(max_nt, nt - nt_off)
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](num_warps=num_warps, **kernel_kwargs)


def _launch_3d_kernel(
    kernel,
    *,
    nk: int,
    nt: int,
    bh_total: int,
    kernel_kwargs: dict,
    num_warps: int = _NUM_WARPS,
) -> None:
    max_nk = max_grid_axis_chunks(nk, nt * bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    chunk_indices = kernel_kwargs.get('chunk_indices')
    cu_seqlens = kernel_kwargs.get('cu_seqlens')
    for nk_off in range(0, nk, max_nk):
        nk_len = min(max_nk, nk - nk_off)
        kernel_kwargs['K_OFFSET'] = nk_off
        max_nt = max_grid_axis_chunks(nt, nk_len * bh_total, max_grid=ASCEND_MAX_GRID_DIM)
        for nt_off in range(0, nt, max_nt):
            nt_len = min(max_nt, nt - nt_off)
            if cu_seqlens is not None and chunk_indices is not None:
                kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
                kernel_kwargs['NT_OFFSET'] = 0
            else:
                kernel_kwargs['NT_OFFSET'] = nt_off
            max_bh = max_grid_axis_chunks(bh_total, nk_len * nt_len, max_grid=ASCEND_MAX_GRID_DIM)
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(nk_len, nt_len, bh_len)](num_warps=num_warps, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
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
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    v += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dA_acc += (bos * HV + i_hv) * BT
    db_acc += bos * HV + i_hv

    n_sub = BT // BC

    for s_r in range(n_sub):
        i_tc_r = i_t * BT + s_r * BC

        p_beta_r = tl.make_block_ptr(beta, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        b_beta_r = tl.load(p_beta_r, boundary_check=(0,))

        b_db_r = tl.zeros([BC], dtype=tl.float32)

        for s_c in range(n_sub):
            i_tc_c = i_t * BT + s_c * BC
            b_dA_rc = tl.zeros([BC, BC], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_dv_r = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
                p_v_c = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
                b_dv_r = tl.load(p_dv_r, boundary_check=(0, 1))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))
                b_dA_rc += tl.dot(b_dv_r, tl.trans(b_v_c), allow_tf32=False)

            p_dA = tl.make_block_ptr(dA_acc, (T, BT), (HV * BT, 1), (i_tc_r, s_c * BC), (BC, BC), (1, 0))
            tl.store(p_dA, b_dA_rc.to(p_dA.dtype.element_ty), boundary_check=(0, 1))

        for i_v in range(tl.cdiv(V, BV)):
            b_dvb_r = tl.zeros([BC, BV], dtype=tl.float32)
            for s_a in range(n_sub):
                i_tc_a = i_t * BT + s_a * BC
                p_A_ra = tl.make_block_ptr(A, (BT, T), (1, HV * BT), (s_r * BC, i_tc_a), (BC, BC), (0, 1))
                p_dv_a = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc_a, i_v * BV), (BC, BV), (1, 0))
                b_A_ra = tl.load(p_A_ra, boundary_check=(0, 1))
                b_dv_a = tl.load(p_dv_a, boundary_check=(0, 1))
                b_dvb_r += tl.dot(b_A_ra, b_dv_a, allow_tf32=False)

            p_v_r = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
            b_v_r = tl.load(p_v_r, boundary_check=(0, 1))
            b_db_r += tl.sum(b_dvb_r * b_v_r, 1)

            b_dv2_r = b_dvb_r * b_beta_r[:, None]
            p_dv2 = tl.make_block_ptr(dv2, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_dv2, b_dv2_r.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        p_db = tl.make_block_ptr(db_acc, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        tl.store(p_db, b_db_r.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_k_part_npu(
    q,
    k,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dg,
    dA_acc,
    db_acc,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
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
    K_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K * V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K * V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dA_acc += (bos * HV + i_hv) * BT
    db_acc += bos * HV + i_hv

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV * K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

    o_i = tl.arange(0, BC)
    n_sub = BT // BC
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)

    b_dgk *= exp2(b_gn)

    b_kdk_sum = tl.zeros([BK], dtype=tl.float32)
    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

        b_dk = tl.zeros([BC, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
            if STATE_V_FIRST:
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype), allow_tf32=False)

        b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
        b_kdk_sum += tl.sum(b_k * b_dk, axis=0)

    b_dgk_total = b_dgk + b_kdk_sum

    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T
        m_last_s = (i_tc_s + o_i) == min(T, i_t * BT + BT) - 1

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

        b_dq = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk = tl.zeros([BC, BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))

            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype), allow_tf32=False)

        b_gk_exp = exp2(b_g)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kdk = b_k * b_dk
        b_dg = b_q * b_dq - b_kdk + m_last_s[:, None] * b_dgk_total

        p_dq = tl.make_block_ptr(dq, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
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
    T,
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
    K_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_tg = (i_b * tl.cdiv(T, BT) + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    dv += (bos * HV + i_hv) * V
    dA_acc += (bos * HV + i_hv) * BT
    db_acc += bos * HV + i_hv
    dg += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K

    n_sub = BT // BC

    for s_r in range(n_sub):
        i_tc_r = i_t * BT + s_r * BC

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_beta_r = tl.make_block_ptr(beta, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_beta_r = tl.load(p_beta_r, boundary_check=(0,))
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta_r[:, None]

        b_dw_r = tl.zeros([BC, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw_r += tl.dot(b_dv.to(b_h.dtype), b_h.to(b_h.dtype), allow_tf32=False)
        b_dw_r = -b_dw_r.to(b_k.dtype)

        b_dkgb_r = tl.zeros([BC, BK], dtype=tl.float32)
        for s_c in range(n_sub):
            i_tc_c = i_t * BT + s_c * BC
            b_dw_c = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_dv = tl.load(p_dv, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dw_c += tl.dot(b_dv.to(b_h.dtype), b_h.to(b_h.dtype), allow_tf32=False)
            b_dw_c = -b_dw_c.to(b_k.dtype)

            p_A_rc = tl.make_block_ptr(A, (BT, T), (1, HV * BT), (s_r * BC, i_tc_c), (BC, BC), (0, 1))
            b_A_rc = tl.load(p_A_rc, boundary_check=(0, 1))
            b_dkgb_r += tl.dot(b_A_rc, b_dw_c, allow_tf32=False)

            p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
            p_g_c = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
            b_kg_c = tl.load(p_k_c, boundary_check=(0, 1)) * exp2(
                tl.load(p_g_c, boundary_check=(0, 1)).to(tl.float32),
            )

            p_dA_acc = tl.make_block_ptr(dA_acc, (T, BT), (HV * BT, 1), (i_tc_r, s_c * BC), (BC, BC), (1, 0))
            b_dA_rc = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
            b_dA_rc += tl.dot(b_dw_r, tl.trans(b_kg_c.to(b_k.dtype)), allow_tf32=False)
            tl.store(p_dA_acc, b_dA_rc.to(p_dA_acc.dtype.element_ty), boundary_check=(0, 1))

        p_db_acc = tl.make_block_ptr(db_acc, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        b_db_r = tl.load(p_db_acc, boundary_check=(0,)).to(tl.float32)
        b_db_r += tl.sum(b_dkgb_r * b_kg, 1)
        tl.store(p_db_acc, b_db_r.to(p_db_acc.dtype.element_ty), boundary_check=(0,))

        p_dk = tl.make_block_ptr(dk, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb_r * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        p_dg = tl.make_block_ptr(dg, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb_r * b_beta_r[:, None]
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dA_finalize_npu(
    beta,
    A,
    dA_acc,
    db_acc,
    dA,
    db,
    cu_seqlens,
    chunk_indices,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    dA_acc += (bos * HV + i_hv) * BT
    db_acc += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT
    db += bos * HV + i_hv

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    for s_r in range(n_sub):
        i_tc_r = i_t * BT + s_r * BC
        m_r = (i_tc_r + o_i) < T

        for s_c in range(n_sub):
            i_tc_c = i_t * BT + s_c * BC
            m_c = (i_tc_c + o_i) < T

            b_dA_rc = tl.zeros([BC, BC], dtype=tl.float32)
            for s_a in range(n_sub):
                i_tc_a = i_t * BT + s_a * BC
                m_a = (i_tc_a + o_i) < T
                b_T_ac = tl.zeros([BC, BC], dtype=tl.float32)
                for s_b in range(n_sub):
                    i_tc_b = i_t * BT + s_b * BC
                    m_b = (i_tc_b + o_i) < T

                    p_dA_ab = tl.make_block_ptr(dA_acc, (T, BT), (HV * BT, 1), (i_tc_a, s_b * BC), (BC, BC), (1, 0))
                    p_beta_b = tl.make_block_ptr(beta, (T,), (HV,), (i_tc_b,), (BC,), (0,))
                    p_A_bc = tl.make_block_ptr(A, (BT, T), (1, HV * BT), (s_b * BC, i_tc_c), (BC, BC), (0, 1))

                    b_dA_ab = tl.load(p_dA_ab, boundary_check=(0, 1)).to(tl.float32)
                    b_beta_b = tl.load(p_beta_b, boundary_check=(0,))
                    b_A_bc = tl.load(p_A_bc, boundary_check=(0, 1))

                    o_t_a = i_tc_a + o_i
                    o_t_b = i_tc_b + o_i
                    m_A_ab = (o_t_a[:, None] > o_t_b[None, :]) & (m_a[:, None] & m_b)
                    b_M_ab = tl.where(m_A_ab, b_dA_ab * b_beta_b[None, :], 0)
                    b_T_ac += tl.dot(b_M_ab.to(b_A_bc.dtype), b_A_bc, allow_tf32=False)

                p_A_ra = tl.make_block_ptr(A, (BT, T), (1, HV * BT), (s_r * BC, i_tc_a), (BC, BC), (0, 1))
                b_A_ra = tl.load(p_A_ra, boundary_check=(0, 1))
                b_dA_rc += tl.dot(b_A_ra, b_T_ac.to(b_A_ra.dtype), allow_tf32=False)

            o_t_r = i_tc_r + o_i
            o_t_c = i_tc_c + o_i
            m_A_rc = (o_t_r[:, None] > o_t_c[None, :]) & (m_r[:, None] & m_c)
            b_dA_rc = tl.where(m_A_rc, -b_dA_rc, 0)

            p_dA = tl.make_block_ptr(dA, (T, BT), (HV * BT, 1), (i_tc_r, s_c * BC), (BC, BC), (1, 0))
            tl.store(p_dA, b_dA_rc.to(p_dA.dtype.element_ty), boundary_check=(0, 1))

        p_db_acc = tl.make_block_ptr(db_acc, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        p_db = tl.make_block_ptr(db, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        tl.store(p_db, tl.load(p_db_acc, boundary_check=(0,)).to(p_db.dtype.element_ty), boundary_check=(0,))


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
):
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    if BT % _BC != 0:
        raise ValueError(f'KDA Ascend bwd requires chunk_size % {_BC} == 0, got {BT}')

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    dA_acc = torch.zeros(B, T, HV, BT, dtype=torch.float, device=A.device)
    db_acc = torch.zeros(B, T, HV, dtype=torch.float, device=beta.device)

    BK = _get_bk(K)
    BV = _get_bv(V)
    NK = triton.cdiv(K, BK)
    is_varlen = cu_seqlens is not None

    common = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        HV=HV,
        BT=BT,
        BC=_BC,
        IS_VARLEN=is_varlen,
        NT_OFFSET=0,
        BH_OFFSET=0,
    )

    _launch_2d_kernel(
        chunk_kda_bwd_kernel_wy_v_part_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            v=v,
            beta=beta,
            A=A,
            dv=dv,
            dv2=dv2,
            dA_acc=dA_acc,
            db_acc=db_acc,
            V=V,
            BV=BV,
            **common,
        ),
    )

    _launch_3d_kernel(
        chunk_kda_bwd_kernel_wy_k_part_npu,
        nk=NK,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            q=q,
            k=k,
            v_new=v_new,
            g=g,
            beta=beta,
            A=A,
            h=h,
            do=do,
            dh=dh,
            dq=dq,
            dk=dk,
            dg=dg,
            dA_acc=dA_acc,
            db_acc=db_acc,
            scale=scale,
            H=H,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            STATE_V_FIRST=state_v_first,
            K_OFFSET=0,
            **common,
        ),
    )

    dw_kwargs = dict(
        k=k,
        g=g,
        beta=beta,
        A=A,
        h=h,
        dv=dv,
        dA_acc=dA_acc,
        db_acc=db_acc,
        dg=dg,
        dk=dk,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        **common,
    )
    for k_off in range(NK):
        dw_kwargs['K_OFFSET'] = k_off
        _launch_3d_kernel(
            chunk_kda_bwd_kernel_wy_dw_part_npu,
            nk=1,
            nt=NT,
            bh_total=B * HV,
            kernel_kwargs=dw_kwargs,
        )

    _launch_2d_kernel(
        chunk_kda_bwd_kernel_wy_dA_finalize_npu,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            beta=beta,
            A=A,
            dA_acc=dA_acc,
            db_acc=db_acc,
            dA=dA,
            db=db,
            **common,
        ),
    )

    dv = dv2
    return dq, dk, dv, db, dg, dA
