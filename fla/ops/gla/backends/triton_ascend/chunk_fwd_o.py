# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_gla_fwd_o_gk adapted for triton-ascend on Ascend NPU."""

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

_NUM_WARPS = 4
_BC = 16
_O_MEM_MULT = 6.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_FALLBACK_BV = 16
_MAX_BK = 64
_MAX_BV = 64


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _O_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_BK, triton.next_power_of_2(K)),
    )


def _get_bv(V: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        V,
        _O_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BV,
        min_block=16,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


def _launch_fwd_o_kernel(kernel, *, nv: int, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nv = max_grid_axis_chunks(nv, nt * bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for v_off in range(0, nv, max_nv):
        v_len = min(max_nv, nv - v_off)
        kernel_kwargs['V_OFFSET'] = v_off
        max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
        for nt_off in range(0, nt, max_nt):
            nt_len = min(max_nt, nt - nt_off)
            chunk_indices = kernel_kwargs.get('chunk_indices')
            cu_seqlens = kernel_kwargs.get('cu_seqlens')
            if cu_seqlens is not None and chunk_indices is not None:
                kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
                kernel_kwargs['NT_OFFSET'] = 0
            else:
                kernel_kwargs['NT_OFFSET'] = nt_off
            max_bh = max_grid_axis_chunks(bh_total, nt_len, max_grid=ASCEND_MAX_GRID_DIM)
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(v_len, nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o_npu(
    q,
    v,
    g,
    h,
    o,
    A,
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
    V_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_v = tl.program_id(0) + V_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    o_i = tl.arange(0, BC)
    o_j = tl.arange(0, BT)
    n_sub = BT // BC

    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T
        b_o = tl.zeros([BC, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            b_h = tl.load(p_h, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype), allow_tf32=False)
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype), allow_tf32=False)

        b_o *= scale
        p_A = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_t * BT + s * BC, 0), (BC, BT), (1, 0))
        p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_col = (i_t * BT + o_j) < T
        m_causal = (s * BC + o_i)[:, None] >= o_j[None, :]
        m_A = m_causal & m_s[:, None] & m_col[None, :]
        b_A = tl.where(m_A, b_A, 0.).to(b_v.dtype)
        b_o += tl.dot(b_A, b_v, allow_tf32=False)
        p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
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
) -> torch.Tensor:
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.zeros_like(v)
    BK = _get_bk(K)
    BV = _get_bv(V)
    nv = triton.cdiv(V, BV)

    _launch_fwd_o_kernel(
        chunk_gla_fwd_kernel_o_npu,
        nv=nv,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs={
            'q': q,
            'v': v,
            'g': g,
            'h': h,
            'o': o,
            'A': A,
            'cu_seqlens': cu_seqlens,
            'chunk_indices': chunk_indices,
            'scale': scale,
            'T': T,
            'H': H,
            'HV': HV,
            'K': K,
            'V': V,
            'BT': BT,
            'BC': _BC,
            'BK': BK,
            'BV': BV,
            'STATE_V_FIRST': state_v_first,
            'IS_VARLEN': cu_seqlens is not None,
            'V_OFFSET': 0,
            'NT_OFFSET': 0,
            'BH_OFFSET': 0,
        },
    )
    return o
