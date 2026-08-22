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

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    compute_row_tile_block_size,
    get_npu_properties,
    launch_grid_chunked,
)

_BC = 16
_SAFETY_MARGIN = 0.80
_FALLBACK = 16
_MAX_TILE = 64


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC, K, 6.0, tiling_row=False, safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK, min_block=16,
        max_block=min(64, max(16, triton.next_power_of_2(K))),
    )


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

    # intra diagonal: fixed tl.static_range(BC) with a per-j active mask
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
    # zero entries above the causal diagonal in the BC×BC block
    b_zero = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(
        A_ptr + o_A[:, None] + o_i,
        b_zero,
        mask=m_A[:, None] & (o_i[:, None] < o_i),
    )


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NK_OFFSET', 'NTNC_OFFSET', 'BH_OFFSET'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_split_npu(
    q, k, g, A, cu_seqlens, chunk_indices, scale, T,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr, NK_OFFSET, NTNC_OFFSET, BH_OFFSET,
):
    i_k = tl.program_id(0) + NK_OFFSET
    i_tc = tl.program_id(1) + NTNC_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = (i_tc // NC).to(tl.int64), i_tc % NC
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) * H * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    q_ptr = q + (bos * H + i_h) * K
    k_ptr = k + (bos * H + i_h) * K
    g_ptr = g + (bos * H + i_h) * K
    A_ptr = A + ((i_k * all + bos) * H + i_h) * BC

    o_c = i_t * BT + i_i * BC + tl.arange(0, BC)
    m_qk = m_A[:, None] & m_k[None, :]
    b_q = tl.load(q_ptr + o_c[:, None] * (H * K) + o_k[None, :], mask=m_qk, other=0.0).to(tl.float32)
    b_g = tl.load(g_ptr + o_c[:, None] * (H * K) + o_k[None, :], mask=m_qk, other=0.0).to(tl.float32)

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
    b_zero = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(
        A_ptr + o_A[:, None] + o_i,
        b_zero,
        mask=m_A[:, None] & (o_i[:, None] < o_i),
    )


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge_npu(
    A, A2, cu_seqlens, chunk_indices, T,
    B: tl.constexpr, H: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, NK: tl.constexpr,
    IS_VARLEN: tl.constexpr, NT_OFFSET, NC_OFFSET, BH_OFFSET,
):
    i_t = tl.program_id(0).to(tl.int64) + NT_OFFSET
    i_c = tl.program_id(1) + NC_OFFSET
    i_bh = tl.program_id(2).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_c * BC >= T:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    o_c = i_t * BT + i_c * BC + tl.arange(0, BC)
    o_i = tl.arange(0, BC)
    m_c = o_c < T
    m_A = m_c[:, None] & (o_i[None, :] < BC)
    m_A2 = m_c[:, None] & ((i_c * BC + o_i)[None, :] < BT)
    for i_k in range(0, NK):
        p_A = A + (i_k * all + bos) * H * BC + i_h * BC + o_c[:, None] * (H * BC) + o_i[None, :]
        b_A += tl.load(p_A, mask=m_A, other=0.0).to(tl.float32)
    p_A2 = A2 + (bos * H + i_h) * BT + o_c[:, None] * (H * BT) + (i_c * BC + o_i)[None, :]
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), mask=m_A2)


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

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(_BC, BT)
    NC = triton.cdiv(BT, BC)
    BK_inter = _get_bk(K)

    A = q.new_zeros(B, T, H, BT, dtype=torch.float)
    base = dict(
        q=q, k=k, g=g, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        scale=scale, T=T, H=H, K=K, BT=BT, BC=BC,
        NT_OFFSET=0, NC_OFFSET=0, BH_OFFSET=0,
    )
    launch_grid_chunked(
        chunk_gla_fwd_A_kernel_intra_sub_inter_npu,
        (NT, NC * NC, B * H),
        offset_keys=('NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'),
        kernel_kwargs={**base, 'BK': BK_inter, 'NC': NC},
    )
    if K <= 256:
        BK_diag = max(triton.next_power_of_2(K), 16)
        launch_grid_chunked(
            chunk_gla_fwd_A_kernel_intra_sub_intra_npu,
            (NT, NC, B * H),
            offset_keys=('NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'),
            kernel_kwargs={**base, 'BK': BK_diag},
        )
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_zeros(NK, B, T, H, BC, dtype=torch.float)
        split_base = dict(
            q=q, k=k, g=g, A=A_intra, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            scale=scale, T=T, B=B, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
            NK_OFFSET=0, NTNC_OFFSET=0, BH_OFFSET=0,
        )
        launch_grid_chunked(
            chunk_gla_fwd_A_kernel_intra_sub_intra_split_npu,
            (NK, NT * NC, B * H),
            offset_keys=('NK_OFFSET', 'NTNC_OFFSET', 'BH_OFFSET'),
            quanta=(1, NC, 1),
            kernel_kwargs=split_base,
        )
        merge_base = dict(
            A=A_intra, A2=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, B=B, H=H, BT=BT, BC=BC, NK=NK,
            NT_OFFSET=0, NC_OFFSET=0, BH_OFFSET=0,
        )
        launch_grid_chunked(
            chunk_gla_fwd_A_kernel_intra_sub_intra_merge_npu,
            (NT, NC, B * H),
            offset_keys=('NT_OFFSET', 'NC_OFFSET', 'BH_OFFSET'),
            kernel_kwargs=merge_base,
        )
    return A


_FWD_O_BV = 128
_FWD_O_MEM_MULT = 6.0


def _get_fwd_o_bk(K: int) -> int:
    """Host-pick K tile for fwd-o from UB budget."""
    return compute_row_tile_block_size(
        64,
        K,
        _FWD_O_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK,
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
    total_chunks_i64 = total_chunks.to(tl.int64)
    h_t_step = total_chunks_i64 * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_v = (task_id // h_t_step).to(tl.int32)
        remainder = task_id % h_t_step
        i_hv = (remainder // total_chunks_i64).to(tl.int32)
        global_t = (remainder % total_chunks_i64).to(tl.int32)
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
            bos = tl.cast(i_b, tl.int64) * T
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
    launch_grid_chunked(
        chunk_gla_bwd_kernel_dA_npu,
        (NT, B * H),
        offset_keys=('NT_OFFSET', 'BH_OFFSET'),
        kernel_kwargs=dict(
            v=v, do=do, dA=dA, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            scale=scale, T=T, H=H, V=V, BT=BT, BV=BV,
            NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    return dA


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
    launch_grid_chunked(
        chunk_gla_bwd_kernel_dv_npu,
        (triton.cdiv(V, BV), NT, B * H),
        offset_keys=('A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'),
        kernel_kwargs=dict(
            k=k, g=g, A=A, do=do, dh=dh, dv=dv,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
            H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, STATE_V_FIRST=state_v_first,
            A_OFFSET=0, NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    return dv


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T', 'A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'])
def chunk_gla_bwd_kernel_intra_npu(
    q, k, g, dA, dq, dk, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr, A_OFFSET, NT_OFFSET, BH_OFFSET,
):
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
    max_j = min(BC, T - i_t * BT - i_i * BC)
    for j in tl.static_range(BC):
        active = j < max_j
        b_dAj = tl.load(dA + o_dA + j, mask=m_dA & active, other=0).to(tl.float32)
        b_kj = tl.load(
            k + (bos + i_t * BT + i_i * BC + j) * H * K + i_h * K + o_k,
            mask=m_k & active, other=0,
        ).to(tl.float32)
        b_gkj = tl.load(
            g + (bos + i_t * BT + i_i * BC + j) * H * K + i_h * K + o_k,
            mask=m_k & active, other=0,
        ).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i & active, b_dAj[:, None] * b_kj[None, :] * exp2(b_g - b_gkj[None, :]), 0.)

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
    for j in tl.static_range(BC):
        active = j < max_j
        b_dAj = tl.load(dA + o_dA2 + j * H * BT, mask=active, other=0).to(tl.float32)
        b_qj = tl.load(
            q + (bos + i_t * BT + i_i * BC + j) * H * K + i_h * K + o_k,
            mask=m_k & active, other=0,
        ).to(tl.float32)
        b_gqj = tl.load(
            g + (bos + i_t * BT + i_i * BC + j) * H * K + i_h * K + o_k,
            mask=m_k & active, other=0,
        ).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i & active, b_dAj[:, None] * b_qj[None, :] * exp2(b_gqj[None, :] - b_g), 0.)

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
    BK = _bwd_pick_bk(K)
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq = torch.zeros_like(q, dtype=torch.float)
    dk = torch.zeros_like(k, dtype=torch.float)
    launch_grid_chunked(
        chunk_gla_bwd_kernel_intra_npu,
        (NK * NC, NT, B * H),
        offset_keys=('A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'),
        kernel_kwargs=dict(
            q=q, k=k, g=g, dA=dA, dq=dq, dk=dk,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
            H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
            A_OFFSET=0, NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    return dq, dk


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
    launch_grid_chunked(
        chunk_gla_bwd_kernel_inter_npu,
        (triton.cdiv(K, BK), NT, B * H),
        offset_keys=('A_OFFSET', 'NT_OFFSET', 'BH_OFFSET'),
        kernel_kwargs=dict(
            q=q, k=k, v=v, g=g, h=h, do=do, dh=dh, dq=dq, dk=dk,
            dq2=dq2, dk2=dk2, dg=dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=T,
            H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, STATE_V_FIRST=state_v_first,
            A_OFFSET=0, NT_OFFSET=0, BH_OFFSET=0,
        ),
    )
    return dq2, dk2, dg
