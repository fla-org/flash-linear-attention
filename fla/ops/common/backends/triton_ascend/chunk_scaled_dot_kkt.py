# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_scaled_dot_kkt_fwd adapted for triton-ascend on Ascend NPU."""

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
# One [BC,BC] fp32 tile + two [BC,BK] operand tiles.
_KKT_MEM_MULT = 4.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 8
_MAX_BK = 64


def _get_bk(K: int) -> int:
    return compute_row_tile_block_size(
        _BC,
        K,
        _KKT_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=8,
        max_block=min(_MAX_BK, triton.next_power_of_2(K)),
    )


def _launch_kkt_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
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
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_npu(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
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

    if i_t * BT >= T:
        return

    k_base = k + (bos * H + i_h // (HV // H)) * K
    A_base = A + (bos * HV + i_h) * BT
    beta_base = beta + bos * HV + i_h
    g_base = g + bos * HV + i_h

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T
        p_bs = tl.make_block_ptr(beta_base, (T,), (HV,), (i_tc_s,), (BC,), (0,))
        b_bs = tl.load(p_bs, boundary_check=(0,))
        if USE_G:
            p_gs = tl.make_block_ptr(g_base, (T,), (HV,), (i_tc_s,), (BC,), (0,))
            b_gs = tl.load(p_gs, boundary_check=(0,))

        for c in range(s + 1):
            i_tc_c = i_t * BT + c * BC
            m_c = (i_tc_c + o_i) < T
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_ks = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
                p_kc = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
                b_ks = tl.load(p_ks, boundary_check=(0, 1))
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                b_A += tl.dot(b_ks, tl.trans(b_kc), allow_tf32=False)

            if USE_G:
                p_gc = tl.make_block_ptr(g_base, (T,), (HV,), (i_tc_c,), (BC,), (0,))
                b_gc = tl.load(p_gc, boundary_check=(0,))
                b_gdiff = b_gs[:, None] - b_gc[None, :]
                b_A *= exp2(b_gdiff)
            b_A *= b_bs[:, None]

            if s == c:
                m_blk = (o_i[:, None] > o_i[None, :]) & (m_s[:, None] & m_s)
            else:
                m_blk = m_s[:, None] & m_c
            b_A = tl.where(m_blk, b_A, 0)

            p_A = tl.make_block_ptr(A_base, (T, BT), (HV * BT, 1), (i_tc_s, c * BC), (BC, BC), (1, 0))
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_scaled_dot_kkt_fwd_npu(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)
    BK = _get_bk(K)
    use_g = g is not None
    g_arg = g if use_g else beta
    _launch_kkt_kernel(
        chunk_scaled_dot_kkt_fwd_kernel_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs={
            'k': k,
            'g': g_arg,
            'beta': beta,
            'A': A,
            'cu_seqlens': cu_seqlens,
            'chunk_indices': chunk_indices,
            'T': T,
            'H': H,
            'HV': HV,
            'K': K,
            'BT': BT,
            'BC': _BC,
            'BK': BK,
            'USE_G': use_g,
            'IS_VARLEN': cu_seqlens is not None,
            'NT_OFFSET': 0,
            'BH_OFFSET': 0,
        },
    )
    return A
