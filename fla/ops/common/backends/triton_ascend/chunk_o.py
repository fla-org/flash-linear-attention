# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend kernels for chunk outputs and their gradients."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import ascend_compile_kwargs, autotune_cache_kwargs, input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_LAUNCH_BLOCK_BUDGET,
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    get_npu_properties,
    launch_grid_chunked,
    max_grid_axis_chunks,
)

_BC_CANDIDATES = (32, 16)
_O_MEM_MULT = 6.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BK = 16
_FALLBACK_BV = 16
_MAX_BK = 64
_MAX_BV = 64
_FULL_BT_BK_CANDIDATES = (64, 32, 16)
_FULL_BT_BV_CANDIDATES = (64, 32, 16)
_DV_FULL_BK_CANDIDATES = (128, 64, 32, 16)
_DV_FULL_BV_CANDIDATES = (128, 64, 32, 16)
# peak UB estimate for bwd_dv_local n_sub==2 path (192 KiB on typical Ascend cores).
_UB_BYTES = 196608


def _get_bk(K: int, BC: int) -> int:
    return compute_row_tile_block_size(
        row_dim=BC,
        fixed_dim=K,
        memory_multiplier=_O_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BK,
        min_block=16,
        max_block=min(_MAX_BK, triton.next_power_of_2(K)),
    )


def _get_bv(V: int, BC: int) -> int:
    return compute_row_tile_block_size(
        row_dim=BC,
        fixed_dim=V,
        memory_multiplier=_O_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BV,
        min_block=16,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


def _get_bc(BT: int, K: int, V: int) -> int:
    """Pick the largest causal sub-block that keeps bwd_dv_local under UB."""
    ub_budget = int(_UB_BYTES * _SAFETY_MARGIN)
    for BC in _BC_CANDIDATES:
        if BT % BC != 0:
            continue
        BK = _get_bk(K=K, BC=BC)
        BV = _get_bv(V=V, BC=BC)
        n_sub = BT // BC
        peak_bytes = n_sub * BC * BC * 4 + 2 * BC * BV * 2 + 2 * BC * BK * 2 + 2 * BC * 4
        if peak_bytes <= ub_budget:
            return BC
    return 16


def _dv_full_peak_bytes(BT: int, BK: int, BV: int) -> int:
    """Peak UB across key/value passes, including copies for tl.dot's destructive inputs."""
    kloop = BT * BT * 4 + BT * BK * 2 * 2 + BT * max(BT, BK) * 4
    vloop = BT * BT * 4 * 2 + BT * BV * 2 + BT * BV * 4
    return max(kloop, vloop)


def _get_dv_full_tiles(BT: int, K: int, V: int) -> tuple[int, int] | None:
    """Return (BK, BV) for full-BT dv_local, or None if it cannot fit UB."""
    ub_budget = int(_UB_BYTES * _SAFETY_MARGIN)
    k_cap = min(128, triton.next_power_of_2(K))
    v_cap = min(128, triton.next_power_of_2(V))
    for BK in _DV_FULL_BK_CANDIDATES:
        if k_cap < BK:
            continue
        for BV in _DV_FULL_BV_CANDIDATES:
            if v_cap < BV:
                continue
            if _dv_full_peak_bytes(BT=BT, BK=BK, BV=BV) <= ub_budget:
                return BK, BV
    return None


def _dqkwg_full_peak_bytes(BT: int, BK: int, BV: int, use_dw: bool) -> int:
    """Peak UB across score, key/value, and epilogue passes, excluding the separate h/dh reduction."""
    ds_pass = BT * BT * 4 + BT * BV * 2 * 2 + BT * BV * 2 + BT * max(BT, BV) * 4
    k_vloop = (
        BT * BT * 2
        + BT * BK * 4 * (3 if use_dw else 2)
        + BT * BV * 2 * (3 if use_dw else 2)
        + BV * BK * 2 * 2
        + BT * max(BK, BV) * 4
    )
    epilogue = BT * BT * 2 * 2 + BT * BK * 4 * 2 + BT * BK * 2 * 2 + BT * 4
    return max(ds_pass, k_vloop, epilogue)


def _get_dqkwg_full_tiles(BT: int, K: int, V: int, use_dw: bool) -> tuple[int, int] | None:
    """Return (BK, BV) for the full-BT dqkwg path, or None if it cannot fit UB."""
    ub_budget = int(_UB_BYTES * _SAFETY_MARGIN)
    k_cap = min(_MAX_BK, triton.next_power_of_2(K))
    v_cap = min(_MAX_BV, triton.next_power_of_2(V))
    for BK in _FULL_BT_BK_CANDIDATES:
        if k_cap < BK:
            continue
        for BV in _FULL_BT_BV_CANDIDATES:
            if v_cap < BV:
                continue
            if _dqkwg_full_peak_bytes(BT=BT, BK=BK, BV=BV, use_dw=use_dw) <= ub_budget:
                return BK, BV
    return None


_HDH_BK_CANDIDATES = (256, 128, 64, 32, 16)
_HDH_BV_CANDIDATES = (256, 128, 64, 32, 16)


def _get_hdh_tiles(K: int, V: int) -> tuple[int, int]:
    """Tiles for the Vector-only Frobenius <h, dh> kernel."""
    ub_budget = int(_UB_BYTES * _SAFETY_MARGIN)
    k_cap = min(256, triton.next_power_of_2(K))
    v_cap = min(256, triton.next_power_of_2(V))
    for BK in _HDH_BK_CANDIDATES:
        if k_cap < BK:
            continue
        for BV in _HDH_BV_CANDIDATES:
            if v_cap < BV:
                continue
            peak = 2 * BK * BV * 2 + BK * BV * 4 + BK * 4
            if peak <= ub_budget:
                return BK, BV
    return 16, 16


def _g_npu_arg(g: torch.Tensor | None, HV: int) -> tuple[torch.Tensor | None, bool]:
    """Transpose g to [B, HV, T] when HV>1 for contiguous token-axis loads."""
    if g is None or HV == 1:
        return g, False
    return g.transpose(1, 2).contiguous(), True


@triton.jit
def _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN: tl.constexpr):
    if IS_VARLEN:
        return g + bos + i_h * T_seq
    return g + tl.cast(i_b, tl.int64) * HV * T_seq + i_h * T_seq


@triton.jit
def _g_block_ptr(g_base, T, offset, BC, G_T_CONTIG: tl.constexpr, HV: tl.constexpr):
    if G_T_CONTIG:
        return tl.make_block_ptr(g_base, (T,), (1,), (offset,), (BC,), (0,))
    return tl.make_block_ptr(g_base, (T,), (HV,), (offset,), (BC,), (0,))


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128}),
        triton.Config({'BK': 64}),
        triton.Config({'BK': 32}),
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
)
@triton.jit(do_not_specialize=["T", "total_chunks", "task_num", "num_core", "H", "HV", "K", "V", "N"])
def chunk_o_fwd_kernel(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H,
    HV,
    K,
    V,
    N,
    total_chunks,
    task_num,
    num_core,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    core_id = tl.program_id(0)
    h_t_step = HV * total_chunks
    for task_id in tl.range(core_id, task_num, num_core):
        i_v = task_id // h_t_step
        remainder = task_id % h_t_step
        i_h = remainder // total_chunks
        global_t = remainder % total_chunks
        T_cur = T

        if IS_VARLEN:
            # find the sequence whose chunk offsets contain global_t.
            i_n = 0
            for n in tl.range(0, N, 1):
                i_n = tl.where(tl.load(chunk_offsets + n + 1) <= global_t, n + 1, i_n)
            i_t = global_t - tl.load(chunk_offsets + i_n).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_cur = (eos - bos).to(tl.int32)
            i_tg = global_t
        else:
            NT = tl.cdiv(T, BT)
            i_n = global_t // NT
            i_t = global_t % NT
            bos = tl.cast(i_n, tl.int64) * T
            i_tg = global_t

        # local pointers avoid accumulating offsets across tasks.
        q_ptr = q + (bos * H + i_h // (HV // H)) * K
        k_ptr = k + (bos * H + i_h // (HV // H)) * K
        v_ptr = v + (bos * HV + i_h) * V
        o_ptr = o + (bos * HV + i_h) * V
        h_base = h + (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_ptr, (K, T_cur), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_base, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1))

            # ascend tl.dot clobbers lhs; copy before the first dot on b_q.
            b_q_c = b_q + 0.0
            # [BT, BK] @ [BK, BV] -> [BT, BV]
            if STATE_V_FIRST:
                b_o = tl.dot(b_q, tl.trans(b_h), b_o)
            else:
                b_o = tl.dot(b_q, b_h, b_o)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A = tl.dot(b_q_c, b_k, b_A)

        if USE_G:
            # g is transposed to [B, HV, T] for contiguous token loads.
            if IS_VARLEN:
                g_ptr = g + bos + i_h * T
            else:
                g_ptr = g + tl.cast(i_n, tl.int64) * HV * T + i_h * T
            p_g = tl.make_block_ptr(g_ptr, (T_cur,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))

            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)
            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * exp2(b_g[:, None] - b_g[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T_cur
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(v_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
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
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
        NT = triton.cdiv(T, BT)
        total_chunks = N * NT
    else:
        N, chunk_offsets = len(cu_seqlens) - 1, prepare_chunk_offsets(cu_seqlens=cu_seqlens, chunk_size=BT)
        # chunk_offsets[-1] stores the cumulative total chunks across all batches
        total_chunks = chunk_offsets[-1].item()

    BV = 128
    NV = triton.cdiv(V, BV)
    num_core = get_npu_properties()["num_aicore"]
    task_num = NV * HV * total_chunks

    if g is not None:
        g = g.transpose(1, 2).contiguous()
    chunk_o_fwd_kernel[(num_core,)](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N=N,
        total_chunks=total_chunks,
        task_num=task_num,
        num_core=num_core,
        BT=BT,
        BV=BV,
        STATE_V_FIRST=state_v_first,
    )
    return o


def _launch_bwd_2d_kernel(kernel, *, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(axis_size=nt, other_grid_product=bh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for nt_off in range(0, nt, max_nt):
        nt_len = min(max_nt, nt - nt_off)
        chunk_indices = kernel_kwargs.get('chunk_indices')
        cu_seqlens = kernel_kwargs.get('cu_seqlens')
        if cu_seqlens is not None and chunk_indices is not None:
            kernel_kwargs['chunk_indices'] = chunk_indices[nt_off:nt_off + nt_len]
            kernel_kwargs['NT_OFFSET'] = 0
        else:
            kernel_kwargs['NT_OFFSET'] = nt_off
        max_bh = max_grid_axis_chunks(axis_size=bh_total, other_grid_product=nt_len, max_grid=ASCEND_MAX_GRID_DIM)
        for bh_off in range(0, bh_total, max_bh):
            bh_len = min(max_bh, bh_total - bh_off)
            kernel_kwargs['BH_OFFSET'] = bh_off
            kernel[(nt_len, bh_len)](**kernel_kwargs)


def _launch_bwd_3d_kernel(kernel, *, nk: int, nt: int, bh_total: int, kernel_kwargs: dict) -> None:
    max_nt = max_grid_axis_chunks(axis_size=nt, other_grid_product=bh_total, max_grid=ASCEND_MAX_GRID_DIM)
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
            max_bh = max_grid_axis_chunks(axis_size=bh_total, other_grid_product=nt_len, max_grid=ASCEND_MAX_GRID_DIM)
            for bh_off in range(0, bh_total, max_bh):
                bh_len = min(max_bh, bh_total - bh_off)
                kernel_kwargs['BH_OFFSET'] = bh_off
                kernel[(1, nt_len, bh_len)](**kernel_kwargs)


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core'])
def chunk_dv_local_bwd_kernel_full(
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
    task_num,
    num_core,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute full-chunk value gradients on a persistent Cube grid.

    A fixed grid avoids host splits at ASCEND_MAX_GRID_DIM.
    Local pointers must be reset for each task.
    """
    core_id = tl.program_id(0)
    bh = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // bh
        i_bh = task_id % bh
        i_b, i_h = i_bh // HV, i_bh % HV
        T_seq = T
        T_cur = T

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_cur = (eos - bos).to(tl.int32)
        else:
            bos = tl.cast(i_b, tl.int64) * T_seq

        q_ptr = q + (bos * H + i_h // (HV // H)) * K
        k_ptr = k + (bos * H + i_h // (HV // H)) * K
        do_ptr = do + (bos * HV + i_h) * V
        dv_ptr = dv + (bos * HV + i_h) * V

        if USE_G:
            if G_T_CONTIG:
                g_base = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            else:
                g_base = g + bos * HV + i_h
            p_g = _g_block_ptr(g_base, T_cur, i_t * BT, BT, G_T_CONTIG, HV)
            b_g = tl.load(p_g, boundary_check=(0,))
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_q = tl.make_block_ptr(q_ptr, (K, T_cur), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_A += tl.dot(b_k, b_q, allow_tf32=False) * scale
        if USE_G or USE_G_GAMMA:
            b_A *= exp2(b_g[None, :] - b_g[:, None])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T_cur
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)
        b_A_pristine = b_A + 0.0

        for i_v in range(tl.cdiv(V, BV)):
            p_do = tl.make_block_ptr(do_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_A_i = b_A_pristine + 0.0
            b_dv = tl.dot(b_A_i.to(b_do.dtype), b_do, allow_tf32=False)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_dv_local_bwd_kernel(
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
    G_T_CONTIG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
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

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    if G_T_CONTIG:
        g_base = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
    else:
        g += bos * HV + i_h
        g_base = g

    o_i = tl.arange(0, BC)
    n_sub = BT // BC
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)

    for i_v in range(tl.cdiv(V, BV)):
        if n_sub == 2:
            i_tc0 = i_t * BT
            i_tc1 = i_t * BT + BC
            m0 = (i_tc0 + o_i) < T
            m1 = (i_tc1 + o_i) < T
            b_dv0 = tl.zeros([BC, BV], dtype=tl.float32)
            b_dv1 = tl.zeros([BC, BV], dtype=tl.float32)

            if USE_G:
                p_g0 = _g_block_ptr(g_base, T, i_tc0, BC, G_T_CONTIG, HV)
                p_g1 = _g_block_ptr(g_base, T, i_tc1, BC, G_T_CONTIG, HV)
                b_g0 = tl.load(p_g0, boundary_check=(0,))
                b_g1 = tl.load(p_g1, boundary_check=(0,))
            if USE_G_GAMMA:
                b_g0 = b_gamma * (o_i + 1).to(tl.float32)
                b_g1 = b_gamma * (BC + o_i + 1).to(tl.float32)

            b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
            b_A01 = tl.zeros([BC, BC], dtype=tl.float32)
            b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
                p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
                b_k0 = tl.load(p_k0, boundary_check=(0, 1))
                b_k1 = tl.load(p_k1, boundary_check=(0, 1))
                p_q0 = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_tc0), (BK, BC), (0, 1))
                p_q1 = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_tc1), (BK, BC), (0, 1))
                b_q0 = tl.load(p_q0, boundary_check=(0, 1))
                b_q1 = tl.load(p_q1, boundary_check=(0, 1))
                b_k0_c = b_k0 + 0.0
                b_A00 += tl.dot(b_k0, b_q0, allow_tf32=False) * scale
                b_A01 += tl.dot(b_k0_c, b_q1, allow_tf32=False) * scale
                b_A11 += tl.dot(b_k1, b_q1, allow_tf32=False) * scale

            if USE_G or USE_G_GAMMA:
                b_A00 = b_A00 * exp2(b_g0[None, :] - b_g0[:, None])
                b_A01 = b_A01 * exp2(b_g1[None, :] - b_g0[:, None])
                b_A11 = b_A11 * exp2(b_g1[None, :] - b_g1[:, None])
            m00 = (o_i[:, None] <= o_i[None, :]) & (m0[:, None] & m0)
            m01 = m0[:, None] & m1
            m11 = (o_i[:, None] <= o_i[None, :]) & (m1[:, None] & m1)
            b_A00 = tl.where(m00, b_A00, 0)
            b_A01 = tl.where(m01, b_A01, 0)
            b_A11 = tl.where(m11, b_A11, 0)

            p_do0 = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0))
            p_do1 = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0))
            b_do0 = tl.load(p_do0, boundary_check=(0, 1))
            b_do1 = tl.load(p_do1, boundary_check=(0, 1))
            b_dv0 = tl.dot(b_A00.to(b_do0.dtype), b_do0, b_dv0, allow_tf32=False)
            b_dv0 = tl.dot(b_A01.to(b_do1.dtype), b_do1, b_dv0, allow_tf32=False)
            b_dv1 = tl.dot(b_A11.to(b_do1.dtype), b_do1, b_dv1, allow_tf32=False)

            p_dv0 = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0))
            p_dv1 = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_dv0, b_dv0.to(p_dv0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dv1, b_dv1.to(p_dv1.dtype.element_ty), boundary_check=(0, 1))
        else:
            for r in range(n_sub):
                i_tc_r = i_t * BT + r * BC
                m_r = (i_tc_r + o_i) < T
                b_dv = tl.zeros([BC, BV], dtype=tl.float32)

                if USE_G:
                    p_gr = _g_block_ptr(g_base, T, i_tc_r, BC, G_T_CONTIG, HV)
                    b_g_r = tl.load(p_gr, boundary_check=(0,))
                if USE_G_GAMMA:
                    b_g_r = b_gamma * (r * BC + o_i + 1).to(tl.float32)

                for c in range(r, n_sub):
                    i_tc_c = i_t * BT + c * BC
                    m_c = (i_tc_c + o_i) < T
                    b_A = tl.zeros([BC, BC], dtype=tl.float32)
                    for i_k in range(tl.cdiv(K, BK)):
                        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
                        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_tc_c), (BK, BC), (0, 1))
                        b_k = tl.load(p_k, boundary_check=(0, 1))
                        b_q = tl.load(p_q, boundary_check=(0, 1))
                        b_A += tl.dot(b_k, b_q, allow_tf32=False) * scale

                    if USE_G:
                        p_gc = _g_block_ptr(g_base, T, i_tc_c, BC, G_T_CONTIG, HV)
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
                    b_dv = tl.dot(b_A.to(b_doc.dtype), b_doc, b_dv, allow_tf32=False)

                p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
                tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_dqkwg_bwd_kernel(
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
    G_T_CONTIG: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    K_OFFSET: tl.constexpr,
    NT_OFFSET: tl.constexpr,
    BH_OFFSET: tl.constexpr,
):
    """BC-tiled dq/dk/dw with fused ds: each (r,c) block computes do@v.T once for both grads."""
    i_k = tl.program_id(0) + K_OFFSET
    i_t = tl.program_id(1) + NT_OFFSET
    i_bh = tl.program_id(2) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
    T_seq = T

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = tl.cast(i_b, tl.int64) * NT + i_t
        bos = tl.cast(i_b, tl.int64) * T

    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
    dh += (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K
    dq_f32 += (bos * HV + i_h) * K
    dk_f32 += (bos * HV + i_h) * K

    if USE_DW:
        dw += (bos * HV + i_h) * K
        dv += (bos * HV + i_h) * V

    if USE_G:
        if G_T_CONTIG:
            g_base = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
        else:
            g += bos * HV + i_h
            g_base = g

    o_i = tl.arange(0, BC)
    n_sub = BT // BC

    if USE_G:
        last_idx = min(i_t * BT + BT, T) - 1
        if G_T_CONTIG:
            b_g_last = tl.load(g_base + last_idx).to(tl.float32)
        else:
            b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g_last = b_gamma * min(BT, T - i_t * BT)

    # dw = -dv @ h  (independent of ds)
    if USE_DW:
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw = tl.dot(b_dv.to(b_h.dtype), b_h.to(b_h.dtype), b_dw, allow_tf32=False)
        p_dw = tl.make_block_ptr(dw, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    # zero dk scratch because the fused intra path accumulates ds.T@q into it.
    for c0 in range(n_sub):
        i_tc = i_t * BT + c0 * BC
        p_zk = tl.make_block_ptr(dk_f32, (T, K), (HV * K, 1), (i_tc, i_k * BK), (BC, BK), (1, 0))
        tl.store(p_zk, tl.zeros([BC, BK], dtype=tl.float32), boundary_check=(0, 1))

    # share ds between dq and dk for each (r, c) block.
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
            b_dq_r = tl.dot(b_do_r, b_h.to(b_do_r.dtype), b_dq_r, allow_tf32=False)

        if USE_G:
            p_gr = _g_block_ptr(g_base, T, i_tc_r, BC, G_T_CONTIG, HV)
            b_gr = tl.load(p_gr, boundary_check=(0,)).to(tl.float32)
            b_dq_r = b_dq_r * exp2(b_gr)[:, None] * scale
        elif USE_G_GAMMA:
            b_gr = b_gamma * (r * BC + o_i + 1).to(tl.float32)
            b_dq_r = b_dq_r * exp2(b_gr)[:, None] * scale

        p_q_r = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        b_q_r = tl.load(p_q_r, boundary_check=(0, 1))

        for c in range(r + 1):
            i_tc_c = i_t * BT + c * BC
            m_c = (i_tc_c + o_i) < T
            b_ds = tl.zeros([BC, BC], dtype=tl.float32)
            for i_v in range(tl.cdiv(V, BV)):
                p_do_r2 = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_tc_r, i_v * BV), (BC, BV), (1, 0))
                p_v_c = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_tc_c, i_v * BV), (BC, BV), (1, 0))
                b_do_r2 = tl.load(p_do_r2, boundary_check=(0, 1))
                b_v_c = tl.load(p_v_c, boundary_check=(0, 1))
                b_ds = tl.dot(b_do_r2, tl.trans(b_v_c), b_ds, allow_tf32=False)

            if USE_G:
                p_gc = _g_block_ptr(g_base, T, i_tc_c, BC, G_T_CONTIG, HV)
                b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
                b_ds = b_ds * exp2(b_gr[:, None] - b_gc[None, :]) * scale
            elif USE_G_GAMMA:
                b_gc = b_gamma * (c * BC + o_i + 1).to(tl.float32)
                b_ds = b_ds * exp2(b_gr[:, None] - b_gc[None, :]) * scale

            if r == c:
                m_blk = (o_i[:, None] >= o_i[None, :]) & (m_r[:, None] & m_c)
            else:
                m_blk = m_r[:, None] & m_c
            p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
            b_k_c = tl.load(p_k_c, boundary_check=(0, 1))
            b_ds = tl.where(m_blk, b_ds, 0).to(b_k_c.dtype)
            b_ds_c = b_ds + 0.0
            b_dq_r = tl.dot(b_ds, b_k_c, b_dq_r, allow_tf32=False)

            p_dk_acc = tl.make_block_ptr(dk_f32, (T, K), (HV * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
            b_dk_acc = tl.load(p_dk_acc, boundary_check=(0, 1))
            b_ds_dk = tl.dot(tl.trans(b_ds_c), b_q_r, allow_tf32=False)
            if not USE_G and not USE_G_GAMMA:
                b_ds_dk = b_ds_dk * scale
            b_dk_acc += b_ds_dk
            tl.store(p_dk_acc, b_dk_acc, boundary_check=(0, 1))

        if not USE_G and not USE_G_GAMMA:
            b_dq_r *= scale

        p_dq_r = tl.make_block_ptr(dq, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        p_dq_f32_r = tl.make_block_ptr(dq_f32, (T, K), (HV * K, 1), (i_tc_r, i_k * BK), (BC, BK), (1, 0))
        tl.store(p_dq_r, b_dq_r.to(p_dq_r.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dq_f32_r, b_dq_r, boundary_check=(0, 1))

    # combine gated inter (v@dh) and fused intra contributions to dk.
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
            b_dk_c = tl.dot(b_v.to(tl.float32), b_dh.to(tl.float32), b_dk_c, allow_tf32=False)

        if USE_G:
            p_gc = _g_block_ptr(g_base, T, i_tc_c, BC, G_T_CONTIG, HV)
            b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
            b_dk_c = b_dk_c * tl.where(m_c, exp2(-b_gc + b_g_last), 0)[:, None]
        elif USE_G_GAMMA:
            b_gc = b_gamma * (c * BC + o_i + 1).to(tl.float32)
            b_dk_c = b_dk_c * tl.where(m_c, exp2(-b_gc + b_g_last), 0)[:, None]

        p_dk_acc = tl.make_block_ptr(dk_f32, (T, K), (HV * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        b_dk_c += tl.load(p_dk_acc, boundary_check=(0, 1))
        p_dk_c = tl.make_block_ptr(dk, (T, K), (HV * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        tl.store(p_dk_c, b_dk_c.to(p_dk_c.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk_acc, b_dk_c, boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core'])
def chunk_dqkwg_bwd_kernel_full(
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
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    task_num,
    num_core,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute full-chunk gradients on a persistent Cube grid.

    A fixed grid avoids multidimensional mapping and host splits at ASCEND_MAX_GRID_DIM.
    Local pointers must be reset for each task.
    The final-token gate contribution from h and dh runs in `chunk_dg_bwd_kernel_hdh`.
    """
    core_id = tl.program_id(0)
    bh = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // bh
        i_bh = task_id % bh
        i_b, i_h = i_bh // HV, i_bh % HV
        T_seq = T
        T_cur = T

        if IS_VARLEN:
            i_tg = i_t
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_cur = (eos - bos).to(tl.int32)
        else:
            NT = tl.cdiv(T_seq, BT)
            i_tg = tl.cast(i_b, tl.int64) * NT + i_t
            bos = tl.cast(i_b, tl.int64) * T_seq

        v_ptr = v + (bos * HV + i_h) * V
        do_ptr = do + (bos * HV + i_h) * V
        h_ptr = h + (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
        dh_ptr = dh + (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
        q_ptr = q + (bos * H + i_h // (HV // H)) * K
        k_ptr = k + (bos * H + i_h // (HV // H)) * K
        dq_ptr = dq + (bos * HV + i_h) * K
        dk_ptr = dk + (bos * HV + i_h) * K
        if USE_DW:
            dw_ptr = dw + (bos * HV + i_h) * K
            dv_ptr = dv + (bos * HV + i_h) * V

        if USE_G:
            dg_head = dg + bos * HV + i_h
            last_idx = min(i_t * BT + BT, T_cur) - 1
            if G_T_CONTIG:
                g_base = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
                b_g_last = tl.load(g_base + last_idx).to(tl.float32)
            else:
                g_base = g + bos * HV + i_h
                b_g_last = tl.load(g_base + last_idx * HV).to(tl.float32)
            p_g = _g_block_ptr(g_base, T_cur, i_t * BT, BT, G_T_CONTIG, HV)
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1).to(tl.float32)
            b_g_last = b_gamma * min(BT, T_cur - i_t * BT)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T_cur
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

        b_ds = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(v_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_ds = tl.dot(b_do, tl.trans(b_v), b_ds, allow_tf32=False)

        if USE_G or USE_G_GAMMA:
            b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        else:
            b_ds = b_ds * m_A.to(tl.float32) * scale
        b_ds_keep = b_ds.to(q.dtype.element_ty)

        for i_k in range(tl.cdiv(K, BK)):
            b_dq = tl.zeros([BT, BK], dtype=tl.float32)
            b_dk = tl.zeros([BT, BK], dtype=tl.float32)
            if USE_DW:
                b_dw = tl.zeros([BT, BK], dtype=tl.float32)

            for i_v in range(tl.cdiv(V, BV)):
                p_v = tl.make_block_ptr(v_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_do = tl.make_block_ptr(do_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))
                b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq, allow_tf32=False)
                b_dk = tl.dot(b_v, b_dh.to(b_v.dtype), b_dk, allow_tf32=False)
                if USE_DW:
                    p_dv = tl.make_block_ptr(dv_ptr, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_dv = tl.load(p_dv, boundary_check=(0, 1))
                    b_dw = tl.dot(b_dv.to(b_h.dtype), b_h.to(b_h.dtype), b_dw, allow_tf32=False)

            if USE_DW:
                p_dw = tl.make_block_ptr(dw_ptr, (T_cur, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

            p_q = tl.make_block_ptr(q_ptr, (T_cur, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_ptr, (T_cur, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            b_ds_lhs = b_ds_keep + 0.0
            b_ds_rhs = b_ds_keep + 0.0
            if USE_G or USE_G_GAMMA:
                b_dq = b_dq * exp2(b_g)[:, None] * scale
                b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
                if USE_G:
                    b_dg_last = tl.sum(b_dk * b_k.to(tl.float32))
                b_dq = tl.dot(b_ds_lhs, b_k, b_dq, allow_tf32=False)
                b_dk = tl.dot(tl.trans(b_ds_rhs), b_q, b_dk, allow_tf32=False)
            else:
                b_dq = b_dq * scale + tl.dot(b_ds_lhs, b_k, allow_tf32=False)
                b_dk = tl.dot(tl.trans(b_ds_rhs), b_q, b_dk, allow_tf32=False)

            p_dq = tl.make_block_ptr(dq_ptr, (T_cur, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_ptr, (T_cur, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
            if USE_G:
                b_dg = tl.sum(b_dq * b_q.to(tl.float32), axis=1) - tl.sum(b_dk * b_k.to(tl.float32), axis=1)
                b_dg = tl.where(o_t < last_idx, b_dg, b_dg + b_dg_last)
                dg_k = dg_head + tl.cast(i_k, tl.int64) * tl.cast(B, tl.int64) * tl.cast(T_seq, tl.int64) * HV
                p_dg = tl.make_block_ptr(dg_k, (T_cur,), (HV,), (i_t * BT,), (BT,), (0,))
                tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T', 'task_num', 'num_core'])
def chunk_dg_bwd_kernel_hdh(
    h,
    dh,
    g,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    task_num,
    num_core,
    B: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    G_T_CONTIG: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute the final-token gate contribution from h and dh on vector cores.

    A separate vector kernel avoids the scalarized fp32 reduction in mixed Cube kernels.
    """
    core_id = tl.program_id(0)
    bh = B * HV
    for task_id in tl.range(core_id, task_num, num_core):
        i_t = task_id // bh
        i_bh = task_id % bh
        i_b, i_h = i_bh // HV, i_bh % HV
        T_seq = T
        T_cur = T

        if IS_VARLEN:
            i_tg = i_t
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_cur = (eos - bos).to(tl.int32)
        else:
            NT = tl.cdiv(T_seq, BT)
            i_tg = tl.cast(i_b, tl.int64) * NT + i_t
            bos = tl.cast(i_b, tl.int64) * T_seq

        h_ptr = h + (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
        dh_ptr = dh + (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
        last_idx = min(i_t * BT + BT, T_cur) - 1
        if G_T_CONTIG:
            g_base = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
            b_g_last = tl.load(g_base + last_idx).to(tl.float32)
        else:
            g_base = g + bos * HV + i_h
            b_g_last = tl.load(g_base + last_idx * HV).to(tl.float32)

        acc = 0.0
        for i_k in range(tl.cdiv(K, BK)):
            for i_v in range(tl.cdiv(V, BV)):
                if STATE_V_FIRST:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                else:
                    p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                    p_dh = tl.make_block_ptr(dh_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
                b_dh = tl.load(p_dh, boundary_check=(0, 1)).to(tl.float32)
                acc += tl.sum(tl.sum(b_h * b_dh, axis=1))

        dg_ptr = dg + bos * HV + i_h
        p_last = dg_ptr + tl.cast(last_idx, tl.int64) * HV
        tl.store(p_last, tl.load(p_last) + acc * exp2(b_g_last))


@triton.jit(do_not_specialize=['T'])
def chunk_dg_bwd_kernel(
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
    G_T_CONTIG: tl.constexpr,
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
    T_seq = T

    n_tokens = B * T
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = tl.cast(i_b, tl.int64) * NT + i_t
        bos = tl.cast(i_b, tl.int64) * T

    v += (bos * HV + i_h) * V
    h += (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
    dh += (tl.cast(i_tg, tl.int64) * HV + i_h) * K * V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq_f32 += (bos * HV + i_h) * K
    dk_f32 += (bos * HV + i_h) * K
    dg += i_k * n_tokens * HV
    dg += bos * HV + i_h
    if G_T_CONTIG:
        g_base = _g_contig_base(g, bos, i_b, i_h, T_seq, HV, IS_VARLEN)
    else:
        g += bos * HV + i_h
        g_base = g

    o_i = tl.arange(0, BC)
    n_sub = BT // BC
    last_idx = min(i_t * BT + BT, T) - 1
    if G_T_CONTIG:
        b_g_last = tl.load(g_base + last_idx).to(tl.float32)
    else:
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
            b_dk_pre = tl.dot(b_v.to(tl.float32), b_dh.to(tl.float32), b_dk_pre, allow_tf32=False)

        p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc_c, i_k * BK), (BC, BK), (1, 0))
        b_k_c = tl.load(p_k_c, boundary_check=(0, 1))
        p_gc = _g_block_ptr(g_base, T, i_tc_c, BC, G_T_CONTIG, HV)
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
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    use_g = g is not None
    use_g_gamma = g_gamma is not None
    full_tiles = _get_dv_full_tiles(BT=BT, K=K, V=V)
    use_full = full_tiles is not None
    if use_full:
        BK, BV = full_tiles
        bwd_kernel = chunk_dv_local_bwd_kernel_full
    else:
        BC = _get_bc(BT=BT, K=K, V=V)
        BK = _get_bk(K=K, BC=BC)
        BV = _get_bv(V=V, BC=BC)
        bwd_kernel = chunk_dv_local_bwd_kernel
    if not use_g and not use_g_gamma and not use_full:
        g_arg = torch.zeros(B, T, HV, dtype=torch.float32, device=q.device)
        use_g = True
        g_t_contig = False
    elif use_g:
        g_arg, g_t_contig = _g_npu_arg(g=g, HV=HV)
    else:
        g_arg = q
        g_t_contig = False

    dv = torch.empty_like(do)
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
        'G_T_CONTIG': g_t_contig,
        'IS_VARLEN': cu_seqlens is not None,
    }
    if use_full:
        kernel_kwargs['B'] = B
        num_core = get_npu_properties()["num_aicore"]
        kernel_kwargs['task_num'] = NT * B * HV
        kernel_kwargs['num_core'] = num_core
        bwd_kernel[(num_core,)](**kernel_kwargs)
    else:
        kernel_kwargs['BC'] = BC
        kernel_kwargs['NT_OFFSET'] = 0
        kernel_kwargs['BH_OFFSET'] = 0
        _launch_bwd_2d_kernel(kernel=bwd_kernel, nt=NT, bh_total=B * HV, kernel_kwargs=kernel_kwargs)
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
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K ** -0.5
    scale = float(scale)

    use_dw = w is not None
    # ungated full-BT fails on a Cube-to-Cube tl.trans copy; gated paths flush ds through vector exp2.
    full_tiles = _get_dqkwg_full_tiles(BT=BT, K=K, V=V, use_dw=use_dw)
    use_full = full_tiles is not None and (g is not None or g_gamma is not None)
    if use_full:
        BK, BV = full_tiles
        dqkwg_kernel = chunk_dqkwg_bwd_kernel_full
    else:
        BC = _get_bc(BT=BT, K=K, V=V)
        BK = _get_bk(K=K, BC=BC)
        BV = _get_bv(V=V, BC=BC)
        dqkwg_kernel = chunk_dqkwg_bwd_kernel
        dq_f32 = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
        dk_f32 = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
    NK = triton.cdiv(K, BK)
    if g is not None:
        g_arg, g_t_contig = _g_npu_arg(g=g, HV=HV)
    else:
        g_arg = q
        g_t_contig = False
    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if use_dw else None

    dqkwg_kwargs = {
        'q': q,
        'k': k,
        'v': v,
        'g': g_arg,
        'g_gamma': g_gamma,
        'h': h,
        'do': do,
        'dh': dh,
        'dq': dq,
        'dk': dk,
        'dw': dw,
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
        'USE_G': g is not None,
        'USE_G_GAMMA': g_gamma is not None,
        'USE_DW': use_dw,
        'G_T_CONTIG': g_t_contig,
        'STATE_V_FIRST': state_v_first,
        'IS_VARLEN': cu_seqlens is not None,
    }
    if use_full:
        dqkwg_kwargs['dg'] = dg if dg is not None else dq
        dqkwg_kwargs['B'] = B
        num_core = get_npu_properties()["num_aicore"]
        dqkwg_kwargs['task_num'] = NT * B * HV
        dqkwg_kwargs['num_core'] = num_core
        dqkwg_kernel[(num_core,)](**dqkwg_kwargs)
    else:
        dqkwg_kwargs['dq_f32'] = dq_f32
        dqkwg_kwargs['dk_f32'] = dk_f32
        dqkwg_kwargs['BC'] = BC
        dqkwg_kwargs['K_OFFSET'] = 0
        dqkwg_kwargs['NT_OFFSET'] = 0
        dqkwg_kwargs['BH_OFFSET'] = 0
        _launch_bwd_3d_kernel(kernel=dqkwg_kernel, nk=NK, nt=NT, bh_total=B * HV, kernel_kwargs=dqkwg_kwargs)

    if dg is not None and not use_full:
        _launch_bwd_3d_kernel(
            kernel=chunk_dg_bwd_kernel,
            nk=NK,
            nt=NT,
            bh_total=B * HV,
            kernel_kwargs={
                'q': q,
                'k': k,
                'v': v,
                'g': g_arg,
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
                'BC': BC,
                'BK': BK,
                'BV': BV,
                'G_T_CONTIG': g_t_contig,
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
        if use_full:
            hdh_bk, hdh_bv = _get_hdh_tiles(K=K, V=V)
            num_vec = get_npu_properties()["num_vectorcore"]
            chunk_dg_bwd_kernel_hdh[(num_vec,)](
                h=h,
                dh=dh,
                g=g_arg,
                dg=dg,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                task_num=NT * B * HV,
                num_core=num_vec,
                B=B,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK=hdh_bk,
                BV=hdh_bv,
                G_T_CONTIG=g_t_contig,
                STATE_V_FIRST=state_v_first,
                IS_VARLEN=cu_seqlens is not None,
                **ascend_compile_kwargs(),
            )
    return dq, dk, dw, dg


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (2, 4, 8)
        for num_stages in (2, 3, 4)
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'PID_OFFSET', 'BH_OFFSET'])
def chunk_bwd_kernel_dv_npu(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    dh,
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
    STATE_V_FIRST: tl.constexpr,
    PID_OFFSET,
    BH_OFFSET,
):
    pid = tl.program_id(0).to(tl.int64) + PID_OFFSET
    NV = tl.cdiv(V, BV)
    i_v, i_t = pid % NV, pid // NV
    i_bh = tl.program_id(1).to(tl.int64) + BH_OFFSET
    i_b, i_h = i_bh // HV, i_bh % HV
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

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
        p_q = q + o_k[:, None] + o_t[None, :] * (H*K)
        b_q = tl.load(p_q, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_A = tl.dot(b_k, b_q, b_A)
        if STATE_V_FIRST:
            p_dh = dh + o_v[:, None] * K + o_k[None, :]
            b_dh = tl.trans(tl.load(p_dh, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0))
        else:
            p_dh = dh + o_k[:, None] * V + o_v[None, :]
            b_dh = tl.load(p_dh, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_dv = tl.dot(b_k, b_dh.to(b_k.dtype), b_dv)

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G or USE_G_GAMMA:
        b_A = tl.where(m_A, b_A * exp2(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
        b_dv *= tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
    p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
    p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    b_dv = tl.dot(b_A.to(b_do.dtype), b_do, b_dv)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


def chunk_bwd_dv_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    if q.dtype in (torch.float16, torch.bfloat16):
        # Triton miscompiles masked K-tail iterations into OOB shared-memory access for 16-bit odd K/V (IMA)
        assert K % 2 == 0 and V % 2 == 0, \
            f"chunk_bwd_dv requires even K and V for {q.dtype}, got K={K}, V={V}"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BK = min(max(triton.next_power_of_2(K), 16), 32)
    BV = min(max(triton.next_power_of_2(V), 16), 32)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dv = torch.empty_like(do)
    grid = (NV * NT, B * HV)
    dv_kwargs = dict(
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        dh=dh,
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
        STATE_V_FIRST=state_v_first,
        PID_OFFSET=0,
        BH_OFFSET=0,
    )
    if grid[0] * grid[1] > ASCEND_LAUNCH_BLOCK_BUDGET:
        launch_grid_chunked(
            chunk_bwd_kernel_dv_npu,
            grid,
            offset_keys=('PID_OFFSET', 'BH_OFFSET'),
            kernel_kwargs=dv_kwargs,
        )
    else:
        chunk_bwd_kernel_dv_npu[grid](**dv_kwargs)
    return dv
