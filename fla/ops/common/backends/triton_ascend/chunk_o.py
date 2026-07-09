# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_fwd_o, chunk_bwd_dv_local, and chunk_bwd_dqkwg adapted for triton-ascend on Ascend NPU."""

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
def chunk_fwd_kernel_o_inter_npu(
    q,
    h,
    g,
    g_gamma,
    o,
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
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    V_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_v = tl.program_id(0) + V_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

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

    q += (bos * H + i_h // (HV // H)) * K
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        b_o = tl.zeros([BC, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_q, tl.trans(b_h), allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h, allow_tf32=False)

        if USE_G:
            p_gs = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc_s,), (BC,), (0,))
            b_gs = tl.load(p_gs, boundary_check=(0,))
            b_o = b_o * exp2(b_gs)[:, None]
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_gs = b_gamma * (s * BC + o_i + 1).to(tl.float32)
            b_o = b_o * exp2(b_gs)[:, None]

        b_o = b_o * scale
        p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_fused_hv1_npu(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    V_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    """Full-BT fused inter+intra path for HV==1."""
    i_v = tl.program_id(0) + V_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

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

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o += tl.dot(b_q, tl.trans(b_h), allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_A += tl.dot(b_q, b_k, allow_tf32=False)

    if USE_G:
        g += bos * HV + i_h
        p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_intra_hv1_npu(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ACCUMULATE_OUTPUT: tl.constexpr,
    V_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    """Full-BT intra path for HV==1."""
    i_v = tl.program_id(0) + V_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_q, b_k, allow_tf32=False)

    if USE_G:
        g += bos * HV + i_h
        p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False) * scale
    if ACCUMULATE_OUTPUT:
        b_o_prev = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        b_o = b_o + b_o_prev
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_intra_npu(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
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
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ACCUMULATE_OUTPUT: tl.constexpr,
    V_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_v = tl.program_id(0) + V_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    for s in range(n_sub):
        i_tc_s = i_t * BT + s * BC
        m_s = (i_tc_s + o_i) < T
        b_o = tl.zeros([BC, BV], dtype=tl.float32)

        if USE_G:
            p_gs = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc_s,), (BC,), (0,))
            b_gs = tl.load(p_gs, boundary_check=(0,))
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_gs = b_gamma * (s * BC + o_i + 1).to(tl.float32)

        for c in range(s + 1):
            i_tc_c = i_t * BT + c * BC
            m_c = (i_tc_c + o_i) < T
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_s, i_k * BK), (BC, BK), (1, 0))
                p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_A += tl.dot(b_q, tl.trans(b_k), allow_tf32=False)

            if USE_G:
                p_gc = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc_c,), (BC,), (0,))
                b_gc = tl.load(p_gc, boundary_check=(0,))
                b_A = b_A * exp2(b_gs[:, None] - b_gc[None, :])
            if USE_G_GAMMA:
                b_gc = b_gamma * (c * BC + o_i + 1).to(tl.float32)
                b_A = b_A * exp2(b_gs[:, None] - b_gc[None, :])

            if s == c:
                m_blk = (o_i[:, None] >= o_i[None, :]) & (m_s[:, None] & m_s)
            else:
                m_blk = m_s[:, None] & m_c
            b_A = tl.where(m_blk, b_A, 0)

            p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)

        p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_tc_s, i_v * BV), (BC, BV), (1, 0))
        if ACCUMULATE_OUTPUT:
            b_o_prev = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
            b_o = (b_o * scale) + b_o_prev
        else:
            b_o = b_o * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_fwd_o_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    BK = _get_bk(K)
    BV = _get_bv(V)
    use_g = g is not None
    use_g_gamma = g_gamma is not None
    g_arg = g if use_g else q
    base_kwargs = {
        'g': g_arg,
        'g_gamma': g_gamma,
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
        'USE_G': use_g,
        'USE_G_GAMMA': use_g_gamma,
        'IS_VARLEN': cu_seqlens is not None,
        'V_OFFSET': 0,
        'NT_OFFSET': 0,
        'BH_OFFSET': 0,
    }
    nv = triton.cdiv(V, BV)
    # Identity gate (exp2(0)=1) when g is disabled.
    if not use_g and not use_g_gamma:
        g_arg = torch.zeros(B, T, HV, dtype=torch.float32, device=q.device)
        use_g = True
    base_kwargs['g'] = g_arg
    base_kwargs['USE_G'] = use_g
    if HV == 1:
        _launch_fwd_o_kernel(
            chunk_fwd_kernel_o_inter_npu,
            nv=nv,
            nt=NT,
            bh_total=B * HV,
            kernel_kwargs={
                **base_kwargs,
                'q': q,
                'h': h,
                'o': o,
                'STATE_V_FIRST': state_v_first,
            },
        )
        intra_hv1_kwargs = {k: v for k, v in base_kwargs.items() if k != 'BC'}
        _launch_fwd_o_kernel(
            chunk_fwd_kernel_o_intra_hv1_npu,
            nv=nv,
            nt=NT,
            bh_total=B * HV,
            kernel_kwargs={
                **intra_hv1_kwargs,
                'q': q,
                'k': k,
                'v': v,
                'o': o,
                'ACCUMULATE_OUTPUT': True,
            },
        )
        return o
    _launch_fwd_o_kernel(
        chunk_fwd_kernel_o_inter_npu,
        nv=nv,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs={
            **base_kwargs,
            'q': q,
            'h': h,
            'o': o,
            'STATE_V_FIRST': state_v_first,
        },
    )
    _launch_fwd_o_kernel(
        chunk_fwd_kernel_o_intra_npu,
        nv=nv,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs={
            **base_kwargs,
            'q': q,
            'k': k,
            'v': v,
            'o': o,
            'ACCUMULATE_OUTPUT': True,
        },
    )
    return o


def _launch_bwd_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
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
            kernel[(nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


def _launch_bwd_3d_kernel(
    kernel,
    *,
    nk: int,
    nt: int,
    bh_total: int,
    kernel_kwargs: dict,
) -> None:
    max_nt = max_grid_axis_chunks(nt, bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for k_idx in range(nk):
        kernel_kwargs['K_OFFSET'] = k_idx
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
                kernel[(1, nt_len, bh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv_local_hv1_npu(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    """Full-BT bwd_dv_local path for HV==1."""
    i_t = tl.program_id(0) + NT_OFFSET
    i_bh = tl.program_id(1) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    if USE_G:
        g += bos * HV + i_h
        p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q, allow_tf32=False) * scale
    if USE_G or USE_G_GAMMA:
        b_A *= exp2(b_g[None, :] - b_g[:, None])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv_local_npu(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
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
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
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

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    for i_v in range(tl.cdiv(V, BV)):
        for r in range(n_sub):
            i_tc_r = i_t * BT + r * BC
            m_r = (i_tc_r + o_i) < T
            b_dv = tl.zeros([BC, BV], dtype=tl.float32)

            if USE_G:
                p_gr = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc_r,), (BC,), (0,))
                b_g_r = tl.load(p_gr, boundary_check=(0,))
            if USE_G_GAMMA:
                b_gamma = tl.load(g_gamma + i_h)
                b_g_r = b_gamma * (r * BC + o_i + 1).to(tl.float32)

            for c in range(r, n_sub):
                i_tc_c = i_t * BT + c * BC
                m_c = (i_tc_c + o_i) < T
                b_A = tl.zeros([BC, BC], dtype=tl.float32)
                for i_k in range(tl.cdiv(K, BK)):
                    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
                    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_A += tl.dot(b_k, tl.trans(b_q), allow_tf32=False) * scale

                if USE_G:
                    p_gc = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc_c,), (BC,), (0,))
                    b_g_c = tl.load(p_gc, boundary_check=(0,))
                    b_A = b_A * exp2(b_g_c[None, :] - b_g_r[:, None])
                if USE_G_GAMMA:
                    b_g_c = b_gamma * (c * BC + o_i + 1).to(tl.float32)
                    b_A = b_A * exp2(b_g_c[None, :] - b_g_r[:, None])

                if r == c:
                    m_blk = (o_i[:, None] <= o_i[None, :]) & (m_r[:, None] & m_r)
                else:
                    m_blk = m_r[:, None] & m_c
                b_A = tl.where(m_blk, b_A, 0)

                p_doc = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
                b_doc = tl.load(p_doc, boundary_check=(0, 1))
                b_dv += tl.dot(b_A.to(b_doc.dtype), b_doc, allow_tf32=False)

            p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg_npu(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dq_f32,
    dk_f32,
    dw,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    i_k = tl.program_id(0) + K_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

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

    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V
    dh += (i_tg * HV + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K
    dq_f32 += (bos * HV + i_h) * K
    dk_f32 += (bos * HV + i_h) * K

    if USE_DW:
        dw += (bos * HV + i_h) * K
        dv += (bos * HV + i_h) * V

    o_i = tl.arange(0, BC)
    n_sub = BT // BC
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    if USE_G:
        g += bos * HV + i_h
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV).to(tl.float32)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g_last = b_gamma * min(BT, T - i_t * BT)

    for i_v in range(tl.cdiv(V, BV)):
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_h.dtype), b_h.to(b_h.dtype), allow_tf32=False)

    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    for c in range(n_sub):
        i_tc_c = i_t * BT + c * BC
        m_c = (i_tc_c + o_i) < T
        b_dk_c = tl.zeros([BC, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
            if STATE_V_FIRST:
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dk_c += tl.dot(b_v.to(tl.float32), b_dh.to(tl.float32), allow_tf32=False)

        p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        b_k_c = tl.load(p_k_c, boundary_check=(0, 1))

        if USE_G:
            p_gc = tl.make_block_ptr(g, (T,), (HV,), (i_tc_c,), (BC,), (0,))
            b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
            b_dk_c = b_dk_c * tl.where(m_c, exp2(-b_gc + b_g_last), 0)[:, None]
        elif USE_G_GAMMA:
            b_gc = b_gamma * (c * BC + o_i + 1).to(tl.float32)
            b_dk_c = b_dk_c * tl.where(m_c, exp2(-b_gc + b_g_last), 0)[:, None]

        for r in range(c, n_sub):
            i_tc_r = i_t * BT + r * BC
            m_r = (i_tc_r + o_i) < T
            b_ds = tl.zeros([BC, BC], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_do_r = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
                p_v_c = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
                b_do_r = tl.load(p_do_r, boundary_check=(0, 1))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))
                b_ds += tl.dot(b_do_r, tl.trans(b_v_c), allow_tf32=False)

            if USE_G:
                p_gr = tl.make_block_ptr(g, (T,), (HV,), (i_tc_r,), (BC,), (0,))
                b_gr = tl.load(p_gr, boundary_check=(0,)).to(tl.float32)
                b_ds = b_ds * exp2(b_gr[:, None] - b_gc[None, :]) * scale
            elif USE_G_GAMMA:
                b_gr = b_gamma * (r * BC + o_i + 1).to(tl.float32)
                b_ds = b_ds * exp2(b_gr[:, None] - b_gc[None, :]) * scale
            else:
                b_ds = b_ds * scale

            if r == c:
                m_blk = (o_i[:, None] >= o_i[None, :]) & (m_r[:, None] & m_c)
            else:
                m_blk = m_r[:, None] & m_c
            b_ds = tl.where(m_blk, b_ds, 0).to(b_k_c.dtype)

            p_q_r = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
            b_q_r = tl.load(p_q_r, boundary_check=(0, 1))
            b_dk_c += tl.dot(tl.trans(b_ds), b_q_r, allow_tf32=False)

        p_dk_c = tl.make_block_ptr(dk, (T, K), (HV * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        p_dk_f32_c = tl.make_block_ptr(dk_f32, (T, K), (HV * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        tl.store(p_dk_c, b_dk_c.to(p_dk_c.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk_f32_c, b_dk_c, boundary_check=(0, 1))

    for r in range(n_sub):
        i_tc_r = i_t * BT + r * BC
        m_r = (i_tc_r + o_i) < T
        b_dq_r = tl.zeros([BC, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_do_r = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_do_r = tl.load(p_do_r, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dq_r += tl.dot(b_do_r, b_h.to(b_do_r.dtype), allow_tf32=False)

        p_q_r = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        b_q_r = tl.load(p_q_r, boundary_check=(0, 1))

        if USE_G:
            p_gr = tl.make_block_ptr(g, (T,), (HV,), (i_tc_r,), (BC,), (0,))
            b_gr = tl.load(p_gr, boundary_check=(0,)).to(tl.float32)
            b_dq_r = b_dq_r * exp2(b_gr)[:, None] * scale
        elif USE_G_GAMMA:
            b_gr = b_gamma * (r * BC + o_i + 1).to(tl.float32)
            b_dq_r = b_dq_r * exp2(b_gr)[:, None] * scale

        for c in range(r + 1):
            i_tc_c = i_t * BT + c * BC
            m_c = (i_tc_c + o_i) < T
            b_ds = tl.zeros([BC, BC], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_do_r2 = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
                p_v_c = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
                b_do_r2 = tl.load(p_do_r2, boundary_check=(0, 1))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))
                b_ds += tl.dot(b_do_r2, tl.trans(b_v_c), allow_tf32=False)

            if USE_G:
                p_gc = tl.make_block_ptr(g, (T,), (HV,), (i_tc_c,), (BC,), (0,))
                b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
                b_ds = b_ds * exp2(b_gr[:, None] - b_gc[None, :]) * scale
            elif USE_G_GAMMA:
                b_gc = b_gamma * (c * BC + o_i + 1).to(tl.float32)
                b_ds = b_ds * exp2(b_gr[:, None] - b_gc[None, :]) * scale
            else:
                b_ds = b_ds * scale

            if r == c:
                m_blk = (o_i[:, None] >= o_i[None, :]) & (m_r[:, None] & m_c)
            else:
                m_blk = m_r[:, None] & m_c
            p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
            b_k_c = tl.load(p_k_c, boundary_check=(0, 1))
            b_ds = tl.where(m_blk, b_ds, 0).to(b_k_c.dtype)
            b_dq_r += tl.dot(b_ds, b_k_c, allow_tf32=False)

        if not USE_G and not USE_G_GAMMA:
            b_dq_r *= scale

        p_dq_r = tl.make_block_ptr(dq, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_dq_f32_r = tl.make_block_ptr(dq_f32, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        tl.store(p_dq_r, b_dq_r.to(p_dq_r.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dq_f32_r, b_dq_r, boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dg_npu(
    q,
    k,
    v,
    g,
    h,
    dh,
    dq_f32,
    dk_f32,
    dg,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
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
    """dg kernel: b_dg_last + sum(dq*q) - sum(dk*k) from fp32 scratch."""
    i_k = tl.program_id(0) + K_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV

    all = B * T
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

    v += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V
    dh += (i_tg * HV + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq_f32 += (bos * HV + i_h) * K
    dk_f32 += (bos * HV + i_h) * K
    dg += i_k * all * HV
    dg += bos * HV + i_h
    g += bos * HV + i_h

    o_i = tl.arange(0, BC)
    n_sub = BT // BC
    last_idx = min(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + last_idx * HV).to(tl.float32)

    b_dg_last = 0.0
    for i_v in range(tl.cdiv(V, BV)):
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dg_last += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32))

    b_dg_last *= exp2(b_g_last)

    for c in range(n_sub):
        i_tc_c = i_t * BT + c * BC
        m_c = (i_tc_c + o_i) < T
        b_dk_pre = tl.zeros([BC, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
            if STATE_V_FIRST:
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dk_pre += tl.dot(b_v.to(tl.float32), b_dh.to(tl.float32), allow_tf32=False)

        p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        b_k_c = tl.load(p_k_c, boundary_check=(0, 1))
        p_gc = tl.make_block_ptr(g, (T,), (HV,), (i_tc_c,), (BC,), (0,))
        b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
        b_dk_pre = b_dk_pre * tl.where(m_c, exp2(-b_gc + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk_pre * b_k_c.to(tl.float32))

    for r in range(n_sub):
        i_tc_r = i_t * BT + r * BC
        p_dq_r = tl.make_block_ptr(dq_f32, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_dk_r = tl.make_block_ptr(dk_f32, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_q_r = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_k_r = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        b_dq_r = tl.load(p_dq_r, boundary_check=(0, 1))
        b_dk_r = tl.load(p_dk_r, boundary_check=(0, 1))
        b_q_r = tl.load(p_q_r, boundary_check=(0, 1)).to(tl.float32)
        b_k_r = tl.load(p_k_r, boundary_check=(0, 1)).to(tl.float32)
        b_dg_r = tl.sum(b_dq_r * b_q_r, axis=1) - tl.sum(b_dk_r * b_k_r, axis=1)
        o_row = i_tc_r + o_i
        b_dg_r = tl.where(o_row < last_idx, b_dg_r, b_dg_r + b_dg_last)
        p_dg_r = tl.make_block_ptr(dg, (T,), (HV,), (i_tc_r,), (BC,), (0,))
        tl.store(p_dg_r, b_dg_r.to(p_dg_r.dtype.element_ty), boundary_check=(0,))


@input_guard
def chunk_bwd_dv_local_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    BK = _get_bk(K)
    BV = _get_bv(V)
    use_g = g is not None
    use_g_gamma = g_gamma is not None
    g_arg = g if use_g else q
    if not use_g and not use_g_gamma:
        g_arg = torch.zeros(B, T, HV, dtype=torch.float32, device=q.device)
        use_g = True

    dv = torch.empty_like(do)
    bwd_kernel = chunk_bwd_kernel_dv_local_hv1_npu if HV == 1 else chunk_bwd_kernel_dv_local_npu
    kernel_kwargs = {
        'q': q,
        'k': k,
        'g': g_arg,
        'g_gamma': g_gamma,
        'do': do,
        'dv': dv,
        'cu_seqlens': cu_seqlens,
        'chunk_indices': chunk_indices,
        'scale': scale,
        'T': T,
        'H': H,
        'HV': HV,
        'K': K,
        'V': V,
        'BT': BT,
        'BK': BK,
        'BV': BV,
        'USE_G': use_g,
        'USE_G_GAMMA': use_g_gamma,
        'IS_VARLEN': cu_seqlens is not None,
        'NT_OFFSET': 0,
        'BH_OFFSET': 0,
    }
    if HV != 1:
        kernel_kwargs['BC'] = _BC
    _launch_bwd_2d_kernel(
        bwd_kernel,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs=kernel_kwargs,
    )
    return dv


@input_guard
def chunk_bwd_dqkwg_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K ** -0.5

    BK = _get_bk(K)
    BV = _get_bv(V)
    NK = triton.cdiv(K, BK)
    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)
    dq_f32 = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
    dk_f32 = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    _launch_bwd_3d_kernel(
        chunk_bwd_kernel_dqkwg_npu,
        nk=NK,
        nt=NT,
        bh_total=B * HV,
        kernel_kwargs={
            'q': q,
            'k': k,
            'v': v,
            'g': g,
            'g_gamma': g_gamma,
            'h': h,
            'do': do,
            'dh': dh,
            'dw': dw,
            'dq': dq,
            'dk': dk,
            'dq_f32': dq_f32,
            'dk_f32': dk_f32,
            'dv': dv,
            'cu_seqlens': cu_seqlens,
            'chunk_indices': chunk_indices,
            'scale': scale,
            'B': B,
            'T': T,
            'H': H,
            'HV': HV,
            'K': K,
            'V': V,
            'BT': BT,
            'BC': _BC,
            'BK': BK,
            'BV': BV,
            'USE_G': g is not None,
            'USE_G_GAMMA': g_gamma is not None,
            'USE_DW': w is not None,
            'STATE_V_FIRST': state_v_first,
            'IS_VARLEN': cu_seqlens is not None,
            'K_OFFSET': 0,
            'NT_OFFSET': 0,
            'BH_OFFSET': 0,
        },
    )

    if dg is not None:
        _launch_bwd_3d_kernel(
            chunk_bwd_kernel_dg_npu,
            nk=NK,
            nt=NT,
            bh_total=B * HV,
            kernel_kwargs={
                'q': q,
                'k': k,
                'v': v,
                'g': g,
                'h': h,
                'dh': dh,
                'dq_f32': dq_f32,
                'dk_f32': dk_f32,
                'dg': dg,
                'cu_seqlens': cu_seqlens,
                'chunk_indices': chunk_indices,
                'B': B,
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
                'K_OFFSET': 0,
                'NT_OFFSET': 0,
                'BH_OFFSET': 0,
            },
        )

    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg
