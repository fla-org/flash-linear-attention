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

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
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
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT

    p_A = tl.make_block_ptr(A + (bos * HV + i_hv) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, HV * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do_c = b_do + 0.0
        b_dA += tl.dot(b_do, b_v, allow_tf32=False)
        b_A_i = tl.load(p_A, boundary_check=(0, 1))
        b_A_i = tl.where(m_A, b_A_i, 0).to(b_do.dtype)
        b_dv = tl.dot(b_A_i, b_do_c, allow_tf32=False)
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
    B, T, HV, V = k.shape[0], k.shape[1], do.shape[2], do.shape[-1]
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
            NT_OFFSET=0,
            BH_OFFSET=0,
        ),
    )
    return dA, dv


_BC = 16
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
):
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
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

        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v_ptr, (T, V), (v_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dA += tl.dot(b_dv, tl.trans(b_v), allow_tf32=False)
            # Ascend tl.dot clobbers lhs; copy A before every V-slab use.
            b_A_c = b_A + 0.0
            b_dvb = tl.dot(b_A_c, b_dv, allow_tf32=False)
            b_db += tl.sum(b_dvb * b_v, 1)
            p_dv2 = tl.make_block_ptr(dv2_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_dv2, (b_dvb * b_beta[:, None]).to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_db = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


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
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = (i_b * tl.cdiv(T, BT) + i_t).to(tl.int64)
            bos, eos = tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T

        q_ptr = q + (bos * H + i_h) * K
        k_ptr = k + (bos * H + i_h) * K
        v_new_ptr = v_new + (bos * HV + i_hv) * V
        g_ptr = g + (bos * HV + i_hv) * K
        h_ptr = h + (i_tg * HV + i_hv).to(tl.int64) * K * V
        do_ptr = do + (bos * HV + i_hv) * V
        dh_ptr = dh + (i_tg * HV + i_hv).to(tl.int64) * K * V
        dq_ptr = dq + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        dg_ptr = dg + (bos * HV + i_hv) * K

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_gn = g_ptr + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        o_i = tl.arange(0, BC)
        n_sub = BT // BC
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dgk += tl.sum(b_h * b_dh, axis=0)

        b_dgk *= exp2(b_gn)

        b_kdk_sum = tl.zeros([BK], dtype=tl.float32)
        for s in range(n_sub):
            i_tc_s = i_t * BT + s * BC
            m_s = (i_tc_s + o_i) < T

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

            b_dk = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_v_new = tl.make_block_ptr(v_new_ptr, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))
                b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype), allow_tf32=False)

            b_dk = b_dk * tl.where(m_s[:, None], exp2(b_gn[None, :] - b_g), 0)
            b_kdk_sum += tl.sum(b_k * b_dk, axis=0)
            p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        b_dgk_total = b_dgk + b_kdk_sum

        for s in range(n_sub):
            i_tc_s = i_t * BT + s * BC
            m_last_s = (i_tc_s + o_i) == min(T, i_t * BT + BT) - 1

            p_k = tl.make_block_ptr(k_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_q = tl.make_block_ptr(q_ptr, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)

            b_dq = tl.zeros([BC, BK], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_do = tl.make_block_ptr(do_ptr, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)

            b_dq = b_dq * exp2(b_g) * scale
            b_dg = b_q * b_dq - b_k * b_dk + m_last_s[:, None] * b_dgk_total

            p_dq = tl.make_block_ptr(dq_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


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
    K_OFFSET: tl.constexpr,
):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = (i_b * tl.cdiv(T, BT) + i_t).to(tl.int64)
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

        h_ptr = h + (i_tg * HV + i_hv).to(tl.int64) * K * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv
        dg_ptr = dg + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K

        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv, b_h.to(b_dv.dtype), allow_tf32=False)

        p_k = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta[:, None]
        # Match CUDA: downcast dw/kg to A.dtype before dA / dkgb GEMMs.
        b_dw = -b_dw.to(b_A.dtype)
        b_kg_a = b_kg.to(b_A.dtype)
        b_dkgb = tl.dot(b_A, b_dw, allow_tf32=False)

        p_dA_acc = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
        b_dw_c = b_dw + 0.0
        b_dA += tl.dot(b_dw_c, tl.trans(b_kg_a), allow_tf32=False)
        tl.store(p_dA_acc, b_dA.to(p_dA_acc.dtype.element_ty), boundary_check=(0, 1))

        p_db_acc = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_db = tl.load(p_db_acc, boundary_check=(0,)).to(tl.float32)
        b_db += tl.sum(b_dkgb * b_kg, 1)
        tl.store(p_db_acc, b_db.to(p_db_acc.dtype.element_ty), boundary_check=(0,))

        p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb * b_beta[:, None]
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


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
):
    """dA = mask(-A @ ((mask * dA_acc * beta) @ A)); copy db_acc into db.

    Fuses the old mask kernel into mid+finalize so dA_acc stays in UB.
    TAIL_MODE 0 = aligned bulk (no boundary_check). TAIL_MODE 1 = tail/varlen.
    First tl.dot clobbers masked dA (dead). Second uses b_A as lhs (dead after store).
    """
    core_id = tl.program_id(0)
    T_seq = T

    for task_id in tl.range(core_id, task_num, num_core):
        i_t = NT_OFFSET + task_id // BH
        i_bh = task_id % BH
        i_b, i_hv = i_bh // HV, i_bh % HV

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
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

        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_dA_acc = tl.make_block_ptr(dA_acc_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_db_acc = tl.make_block_ptr(db_acc_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))

        o_t = i_t * BT + tl.arange(0, BT)
        if TAIL_MODE == 0:
            b_A = tl.load(p_A)
            b_dA = tl.load(p_dA_acc).to(tl.float32)
            b_beta = tl.load(p_beta)
            m_A = o_t[:, None] > o_t[None, :]
        else:
            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
            b_beta = tl.load(p_beta, boundary_check=(0,))
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])

        b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
        # mid: (mask * dA_acc * beta) @ A. lhs clobbers b_dA; A is rhs then lhs.
        b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
        b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
        b_fin = tl.where(m_A, -b_fin, 0)

        if TAIL_MODE == 0:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
            tl.store(p_db, tl.load(p_db_acc).to(p_db.dtype.element_ty))
        else:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
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
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if is_varlen else g.new_zeros(1, dtype=torch.int64)

    v_arg, g_t_contig = _t_contig_arg(v, HV)
    beta_arg = _t_contig_arg(beta, HV)[0]
    A_arg = _t_contig_arg(A, HV)[0]
    dv_arg = _t_contig_arg(dv, HV)[0]

    bh_total = B * HV
    num_core = get_npu_properties()['num_vectorcore']
    task_num = NT * bh_total
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
        G_T_CONTIG=g_t_contig,
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
        BC=32 if BT >= 32 else _BC,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=is_varlen,
    )
    for k_off in range(NK):
        k_part_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_k_part_npu[(num_core,)](**k_part_kwargs)

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
        K_T_CONTIG=k_t_contig,
        G_T_CONTIG=g_t_contig,
    )
    for k_off in range(NK):
        dw_kwargs['K_OFFSET'] = k_off
        chunk_kda_bwd_kernel_wy_dw_part_npu[(num_core,)](**dw_kwargs)

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
            G_T_CONTIG=g_t_contig,
        ),
    )

    dv = dv2
    return dq, dk, dv, db, dg, dA
