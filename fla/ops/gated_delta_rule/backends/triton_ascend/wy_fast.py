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

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


# prepare_wy_repr_bwd stage-specific UB models
_PREPARE_BWD_K_MEM_MULT = 4.5
_PREPARE_BWD_V_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_BWD = 128


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


def _get_bwd_k_tile(BT: int, K: int) -> int:
    return compute_row_tile_block_size(
        BT, K, _PREPARE_BWD_K_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_BWD, triton.next_power_of_2(K)),
    )


def _get_bwd_v_tile(BT: int, V: int) -> int:
    return compute_row_tile_block_size(
        BT, V, _PREPARE_BWD_V_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_TILE,
        min_block=8,
        max_block=min(_MAX_TILE_BWD, triton.next_power_of_2(V)),
    )


def _get_bwd_tiles(BT: int, K: int, V: int) -> tuple[int, int]:
    return _get_bwd_k_tile(BT, K), _get_bwd_v_tile(BT, V)


@triton.jit
def _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN: tl.constexpr):
    if IS_VARLEN:
        return g + bos + i_h * T_seq
    return g + i_b * HV * T_seq + i_h * T_seq


@triton.jit
def _t_block_ptr(base, T, offset, BLK, CONTIG: tl.constexpr, HV: tl.constexpr):
    if CONTIG:
        return tl.make_block_ptr(base, (T,), (1,), (offset,), (BLK,), (0,))
    return tl.make_block_ptr(base, (T,), (HV,), (offset,), (BLK,), (0,))


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
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **kernel_kwargs)


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
):
    T_max = T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
            bos_bh = bos
        else:
            i_t = i_t_o
            bos, eos = i_b * T, i_b * T + T
            bos_bh = i_b * HV * T_max

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * HV + i_h) * BT + offs_t_2d * (HV * BT) + offs_bt * 1
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_beta = beta + bos_bh + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)

            ptr_v = v + (bos * HV + i_h) * V + offs_t_2d * (HV * V) + offs_v * 1
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)

            ptr_u = u + (bos * HV + i_h) * V + offs_t_2d * (HV * V) + offs_v * 1
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        if USE_G:
            ptr_g = g + bos_bh + i_h * T_max + global_offs_t
            b_g = exp2(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            ptr_k = (
                k
                + (bos * H + i_h // (HV // H)) * K
                + offs_t_2d * (H * K)
                + offs_k * 1
            )
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = b_k * b_beta[:, None]
            if USE_G:
                b_kb = b_kb * b_g[:, None]
            b_w = tl.dot(b_A, b_kb, allow_tf32=False)

            ptr_w = w + (bos * HV + i_h) * K + offs_t_2d * (HV * K) + offs_k * 1
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T", "B", "task_num", "num_core"])
def prepare_wy_repr_bwd_kv_npu(
    k, v, beta, g, A, dw, du, dk, dv, dA_scr, db, dg,
    cu_seqlens, chunk_indices, T, B,
    task_num, num_core,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
    G_T_CONTIG: tl.constexpr, BETA_T_CONTIG: tl.constexpr,
    DG_T_CONTIG: tl.constexpr, DB_T_CONTIG: tl.constexpr,
    G_EXP_PRECOMP: tl.constexpr,
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
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            i_t = i_t_o
            bos, eos = i_b * T_seq, i_b * T_seq + T_seq

        if BETA_T_CONTIG:
            beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            p_b = _t_block_ptr(beta_ptr, T, i_t * BT, BT, True, HV)
        else:
            p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        if DB_T_CONTIG:
            db_ptr = _g_contig_base(db, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            p_db = _t_block_ptr(db_ptr, T, i_t * BT, BT, True, HV)
        else:
            p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(
            A + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1),
        )
        p_dA = tl.make_block_ptr(
            dA_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1),
        )

        b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_db = tl.zeros([BT], dtype=tl.float32)
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)

        if USE_G:
            if G_T_CONTIG:
                g_ptr = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
                p_g = _t_block_ptr(g_ptr, T, i_t * BT, BT, True, HV)
            else:
                p_g = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_g_exp = b_g if G_EXP_PRECOMP else exp2(b_g)
            b_dg = tl.zeros([BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_dk = tl.make_block_ptr(
                dk + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_dw = tl.make_block_ptr(
                dw + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if USE_G:
                b_kbg = b_k * (b_b * b_g_exp)[:, None]
            else:
                b_kbg = b_k * b_b[:, None]
            b_dw = tl.load(p_dw, boundary_check=(0, 1)).to(tl.float32)
            b_dA += tl.dot(b_dw, tl.trans(b_kbg), allow_tf32=False)
            b_dkbg = tl.dot(b_A, b_dw, allow_tf32=False)
            if USE_G:
                b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
                b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
                b_dg += tl.sum(b_dkbg * b_kbg, 1)
            else:
                b_dk = b_dkbg * b_b[:, None]
                b_db += tl.sum(b_dkbg * b_k, 1)
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(
                v + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            p_dv = tl.make_block_ptr(
                dv + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            p_du = tl.make_block_ptr(
                du + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            b_du = tl.load(p_du, boundary_check=(0, 1)).to(tl.float32)
            b_vb = b_v * b_b[:, None]
            b_dA += tl.dot(b_du, tl.trans(b_vb), allow_tf32=False)
            b_dvb = tl.dot(b_A, b_du, allow_tf32=False)
            b_dv = b_dvb * b_b[:, None]
            b_db += tl.sum(b_dvb * b_v, 1)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
        if USE_G:
            if DG_T_CONTIG:
                dg_ptr = _g_contig_base(dg, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
                p_dg = _t_block_ptr(dg_ptr, T, i_t * BT, BT, True, HV)
            else:
                p_dg = tl.make_block_ptr(dg + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_da_mask_dot1_npu(
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
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(
        A + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1),
    )
    p_in = tl.make_block_ptr(
        dA_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1),
    )
    p_out = tl.make_block_ptr(
        dA_mid + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_dA = tl.load(p_in, boundary_check=(0, 1)).to(tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_out = tl.dot(b_dA, b_A, allow_tf32=False)
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
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_dA = tl.load(p_in, boundary_check=(0, 1)).to(tl.float32)
    b_dA = tl.dot(b_A, b_dA, allow_tf32=False)
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
    IS_VARLEN: tl.constexpr, G_T_CONTIG: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    n_sub = BT // BC
    if G_T_CONTIG:
        g_ptr = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
    else:
        g_ptr = g + (bos * HV + i_h)

    for r in range(n_sub):
        i_tr = i_t * BT + r * BC
        p_gr = _t_block_ptr(g_ptr, T, i_tr, BC, G_T_CONTIG, HV)
        b_gr = tl.load(p_gr, boundary_check=(0,)).to(tl.float32)
        for c in range(n_sub):
            i_tc = i_t * BT + c * BC
            p_dA = tl.make_block_ptr(
                dA_out + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
            p_gc = _t_block_ptr(g_ptr, T, i_tc, BC, G_T_CONTIG, HV)
            b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
            b_gate = exp2(b_gr[:, None] - b_gc[None, :])
            b_prod = b_dA * b_gate
            b_dA = tl.where(b_prod == b_prod, b_prod, 0.0)
            tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


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
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            i_t = i_t_o
            bos, eos = i_b * T_seq, i_b * T_seq + T_seq

        if BETA_T_CONTIG:
            beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            p_b = _t_block_ptr(beta_ptr, T, i_t * BT, BT, True, HV)
        else:
            p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        if DB_T_CONTIG:
            db_ptr = _g_contig_base(db, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            p_db = _t_block_ptr(db_ptr, T, i_t * BT, BT, True, HV)
        else:
            p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        p_dA = tl.make_block_ptr(
            dA_out + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1),
        )

        b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_db = tl.load(p_db, boundary_check=(0,)).to(tl.float32)
        b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            p_dk = tl.make_block_ptr(
                dk + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_kb = b_k * b_b[:, None]
            b_dkb = tl.dot(b_dA, b_k, allow_tf32=False)
            b_db += tl.sum(b_dkb * b_k, 1)
            b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb), b_dA, allow_tf32=False))
            b_dk += tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finalize_a2_npu(
    k, beta, a2_scr,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr, BETA_T_CONTIG: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if BETA_T_CONTIG:
        beta_ptr = _g_contig_base(beta, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        p_b = _t_block_ptr(beta_ptr, T, i_t * BT, BT, True, HV)
    else:
        p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
    p_a2 = tl.make_block_ptr(a2_scr + (bos * HV + i_h) * BT, (BT, T), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    b_A2 = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_A2 += tl.dot(b_k, tl.trans(b_k), allow_tf32=False)
    b_A2 *= b_b[:, None]
    tl.store(p_a2, b_A2.to(p_a2.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_finalize_dg_npu(
    dA_out, a2_scr, dg, col_acc_scr,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    IS_VARLEN: tl.constexpr, DG_T_CONTIG: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T
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

    if DG_T_CONTIG:
        dg_ptr = _g_contig_base(dg, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
    else:
        dg_ptr = dg + (bos * HV + i_h)

    for r in range(n_sub):
        i_tr = i_t * BT + r * BC
        p_dg_r = _t_block_ptr(dg_ptr, T, i_tr, BC, DG_T_CONTIG, HV)
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

    p_dg = _t_block_ptr(dg_ptr, T, i_t * BT, BT, DG_T_CONTIG, HV)
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
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = 64
    BV = 64

    u = torch.empty_like(v)
    w = k.new_empty(B, T, HV, K)
    beta = beta.transpose(1, 2).contiguous()
    if g is not None:
        g = g.transpose(1, 2).contiguous()

    num_core = get_npu_properties()["num_aicore"]
    task_num = NT * B * HV
    recompute_w_u_fwd_kernel_npu[(num_core,)](
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
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
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
    dg, dg_t_contig = None, False
    if use_g:
        dg, dg_t_contig = _t_npu_buf(B, T, HV, dtype=g.dtype, device=k.device)
    db, db_t_contig = _t_npu_buf(B, T, HV, dtype=beta.dtype, device=k.device)
    beta_arg, beta_t_contig = _beta_npu_arg(beta, HV)
    g_gate, g_t_contig = None, False
    g_exp_precomp = False
    if use_g:
        g_gate, g_t_contig = _g_npu_arg(g, HV)
        g_k_arg = g_gate
        if not is_varlen:
            g_k_arg = torch.exp2(g_gate.float()).to(g_gate.dtype)
            g_exp_precomp = True
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
    )
    core_base = dict(B=B, **base)
    task_num = NT * B * HV
    _launch_wy_core_grid(
        prepare_wy_repr_bwd_kv_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            k=k, v=v, beta=beta_arg, g=g_k_arg if use_g else k, A=A, dw=dw, du=du,
            dk=dk, dv=dv, dA_scr=dA_scr, db=db, dg=dg_arg,
            H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, USE_G=use_g,
            G_T_CONTIG=g_t_contig, BETA_T_CONTIG=beta_t_contig,
            DG_T_CONTIG=dg_t_contig, DB_T_CONTIG=db_t_contig,
            G_EXP_PRECOMP=g_exp_precomp,
            **core_base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_mask_dot1_npu,
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
                g=g_gate, dA_out=dA_out,
                HV=HV, BC=_DG_BLK, G_T_CONTIG=g_t_contig,
                **base,
            ),
        )
    _launch_wy_core_grid(
        prepare_wy_repr_bwd_finalize_k_npu,
        task_num=task_num,
        kernel_kwargs=dict(
            k=k, beta=beta_arg, dA_out=dA_out, dk=dk, db=db,
            H=H, HV=HV, K=K, BK=BK, BETA_T_CONTIG=beta_t_contig, DB_T_CONTIG=db_t_contig,
            **core_base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_a2_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                k=k, beta=beta_arg, a2_scr=a2_scr,
                H=H, HV=HV, K=K, BK=BK, BETA_T_CONTIG=beta_t_contig,
                **base,
            ),
        )
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_dg_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                dA_out=dA_out, a2_scr=a2_scr, dg=dg_arg, col_acc_scr=col_acc_scr,
                HV=HV, BC=_DG_BLK, DG_T_CONTIG=dg_t_contig,
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
