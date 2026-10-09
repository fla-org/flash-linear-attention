# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gated delta rule state updates for Ascend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    compute_row_tile_block_size,
    get_npu_properties,
    get_ub_manager,
)


@triton.jit
def mload(ptr, mask, MASK: tl.constexpr):
    """Mask only when ``MASK`` is true. Aligned launches DCE the predicate and keep a static DMA."""
    if MASK:
        return tl.load(ptr, mask=mask, other=0)
    return tl.load(ptr)


@triton.jit
def mstore(ptr, val, mask, MASK: tl.constexpr):
    if MASK:
        tl.store(ptr, val, mask=mask)
    else:
        tl.store(ptr, val)


def need_dma_mask(T: int, BT: int, K: int, BK: int, V: int, BV: int, varlen: bool) -> bool:
    if varlen or T % BT != 0:
        return True
    return bool(K % BK != 0 or V % BV != 0)


def mask_dma_heuristic(args) -> bool:
    """Constexpr gate so aligned launches DCE masked DMA. ``T`` is runtime, so this stays on the host."""
    return need_dma_mask(
        args["T"], args["BT"], args["K"], args["BK"], args["V"], args["BV"],
        args["cu_seqlens"] is not None,
    )


# b_h[BK,BV] fp32 × n_slabs stay live; w/k are sequential one-slab tiles.
# ``enable_ubuf_saving`` on the fwd launch packs that live set under 192KiB
# (without it, BK=256/BV=128 reports 256KiB and fails to compile).
# ``unit_flag`` overlaps Cube/Fixpipe waits (``--enable-hivm-unit-flag-sync``).
_FWD_H_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BV = 16
_MAX_BV = 64
_FWD_H_BT = 64
# live-range packing fits BK=256/BV=128 despite a peak estimate of 1.18 times physical UB capacity.
_FWD_UB_SOFT = 1.20
# Masked blockdim64 (unaligned T / varlen) is fatter than oneslab. D128 BK=128/BV=128
# analytical peak is 132352 B, but the compiler asks for 196864 B (256 B over 192 KiB).
_FWD_UB_MASKED = 0.67
_DHU_BT = 64
# ubuf_saving packs the D256 / BV=128 live set (about 1.17× 192 KiB) the way fwd_h does.
_DHU_UB_SOFT = 1.20
# Masked DMA (varlen or T not divisible by BT) keeps pad buffers the packer does not
# reclaim. D256 / 4 slabs / BV=128 asks for 263168 B against 192 KiB, so stay inside raw UB.
_DHU_UB_MASKED = 1.0
# Tighter fraction when in-kernel exp2(g) inflates live UB (~1.7× analytical).
_DHU_UB_GATE_INLINE = 0.60


def _get_bv(K: int, V: int) -> int:
    return compute_row_tile_block_size(
        row_dim=min(K, 64),
        fixed_dim=V,
        memory_multiplier=_FWD_H_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BV,
        min_block=16,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


def _fwd_h_peak_bytes(BK: int, BV: int, n_slabs: int, BT: int = _FWD_H_BT) -> int:
    """Estimate forward UB usage with all state slabs live and one w/k slab at a time."""
    return n_slabs * BK * BV * 4 + BK * BT * 4 + BT * BV * 4 + BK * 4 + 12 * BT


def _fwd_h_tile_cost(K: int, V: int, BK: int, BV: int) -> int:
    """Scalar proxy: nv × (base ptrs + ~4 per K-slab), same family as bwd dhu."""
    return triton.cdiv(V, BV) * (3 + 4 * triton.cdiv(K, BK))


def _select_fwd_h_tiles(
    K: int,
    V: int,
    state_v_first: bool,
    *,
    masked: bool = False,
) -> tuple[int, int]:
    """Pick (BK, BV) minimizing ptr/scalar cost under a UB cap.

    ``STATE_V_FIRST`` keeps BK=64 because ``tl.trans(b_h)`` copies inflate live UB.
    At most four K-slabs (kernel unroll); BK starts at 64.
    ``masked`` is the blockdim64 path (unaligned T or varlen), whose compiler UB
    exceeds the analytical peak; oneslab keeps the soft over-admit.
    """
    if state_v_first:
        return 64, _get_bv(K=K, V=V)

    soft_cap = int(get_ub_manager().ub_capacity_bytes * (_FWD_UB_MASKED if masked else _FWD_UB_SOFT))
    max_bk = min(256, triton.next_power_of_2(max(K, 64)))
    desired_v = triton.next_power_of_2(V)
    best: tuple[int, int, int] | None = None  # cost, BK, BV

    bk = 64
    while bk <= max_bk:
        n_slabs = triton.cdiv(K, bk)
        if n_slabs <= 4:
            bv = 16
            while bv <= min(desired_v, 256):
                if _fwd_h_peak_bytes(BK=bk, BV=bv, n_slabs=n_slabs) <= soft_cap:
                    cost = _fwd_h_tile_cost(K=K, V=V, BK=bk, BV=bv)
                    if (
                        best is None
                        or cost < best[0]
                        or (cost == best[0] and (bk > best[1] or (bk == best[1] and bv > best[2])))
                    ):
                        best = (cost, bk, bv)
                bv *= 2
        bk *= 2

    if best is None:
        return 64, _get_bv(K=K, V=V)
    return best[1], best[2]


def _dhu_peak_bytes(
    BK: int,
    BV: int,
    n_slabs: int,
    BT: int = _DHU_BT,
    partial_kv: bool = False,
    row_select: bool = False,
) -> int:
    # dh slabs stay live. Non-v-first keeps dv, do, and the dot acc as [BT, BV] fp32.
    # The extra do/dv copies are v-first only (lhs clobber), so they are not charged here.
    # +512 B is the address tile. D256 / BV=128 / 4 slabs is ~230 KiB and relies on ubuf_saving.
    # Ragged K/V keeps one more [BK, BV] fp32 tile. D100 / 2 slabs / BV=128 asks for
    # 197120 B, 512 B over 192 KiB, and ubuf_saving does not reclaim it.
    # Masked launches select invalid token rows of b_dv to 0. That select does not alias
    # b_dv: D128 / 2 slabs / BV=128 asks for 213504 B against 192 KiB.
    peak = n_slabs * BK * BV * 4 + 3 * BT * BV * 4 + 512
    if partial_kv:
        peak += BK * BV * 4
    if row_select:
        peak += BT * BV * 4
    return peak


def _dhu_tile_cost(K: int, V: int, BK: int, BV: int) -> int:
    """Reverse-loop scalar proxy: nv × (base ptrs + ~4 per K-slab)."""
    return triton.cdiv(V, BV) * (3 + 4 * triton.cdiv(K, BK))


def _select_bwd_dhu_tiles(
    K: int,
    V: int,
    state_v_first: bool,
    *,
    gate_inline: bool = False,
    masked: bool = False,
) -> tuple[int, int]:
    """Pick (BK, BV) minimizing launch×ptr cost under a soft UB cap.

    ``gate_inline`` covers USE_G without host-precomputed exp2 (unaligned T / varlen).
    ``masked`` is blockdim64 DMA (varlen or unaligned T); those pad buffers are not packed.
    Ragged K/V tiles are scored against raw UB: the extra tile is not packed.
    """
    if state_v_first:
        return 64, _get_bv(K=K, V=V)

    ub = get_ub_manager().ub_capacity_bytes
    inline_cap = int(ub * _DHU_UB_GATE_INLINE)
    soft_cap = int(ub * _DHU_UB_SOFT)
    masked_cap = int(ub * _DHU_UB_MASKED)
    # bwd kernel is blockdim64: K is covered by up to four BK=64 slabs only.
    max_bk = 64
    desired_v = triton.next_power_of_2(V)
    # partial K/V slabs plus tail-token masking can exceed UB with BV > 64 (e.g. D=100).
    if K % 64 or V % 64:
        desired_v = min(desired_v, 64)
    best: tuple[int, int, int] | None = None  # cost, BK, BV

    bk = 64
    while bk <= max_bk:
        n_slabs = triton.cdiv(K, bk)
        bv = 16
        while bv <= min(desired_v, 256):
            partial = (K % bk != 0) or (V % bv != 0)
            if gate_inline:
                cap = inline_cap
            elif partial:
                cap = ub
            elif masked:
                cap = masked_cap
            else:
                cap = soft_cap
            # The 2D row select is compiled whenever DMA is masked: varlen, a ragged
            # chunk, or a partial K/V tile.
            if _dhu_peak_bytes(
                bk, bv, n_slabs, partial_kv=partial, row_select=masked or partial,
            ) <= cap:
                cost = _dhu_tile_cost(K, V, bk, bv)
                if best is None or cost < best[0] or (cost == best[0] and bk > best[1]):
                    best = (cost, bk, bv)
            bv *= 2
        bk *= 2

    if best is None:
        return 64, _get_bv(K=K, V=V)
    return best[1], best[2]


def _launch_core_grid(kernel, *, task_num: int, kernel_kwargs: dict, **compile_opts) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **compile_opts, **kernel_kwargs)


# enable_ubuf_saving packs BK=256/BV=128 into 192 KiB; unit_flag overlaps Cube/Fixpipe waits.
_FWD_H_COMPILE = dict(enable_ubuf_saving=True, unit_flag=True)
_DHU_COMPILE = dict(enable_ubuf_saving=True, unit_flag=True)

_FWD_H_HEURISTICS = {
    "USE_G": lambda args: args["g"] is not None or args["g_ratio"] is not None,
    "USE_G_PRECOMP": lambda args: args["g_ratio"] is not None,
    "USE_GK": lambda args: args["gk"] is not None or args["gk_last_exp"] is not None,
    "USE_GK_PRECOMP": lambda args: args["gk_last_exp"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "MASK_DMA": mask_dma_heuristic,
}


@triton.heuristics(_FWD_H_HEURISTICS)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_gated_delta_rule_h_fwd_kernel(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    g_ratio,
    g_last_exp,
    gk_last_exp,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_PRECOMP: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GK_PRECOMP: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    core_id = tl.program_id(0)
    NV: tl.constexpr = tl.cdiv(V, BV)
    DH_CS: tl.constexpr = HV * K * V
    stride_v: tl.constexpr = HV * V
    stride_k: tl.constexpr = H * K
    stride_w: tl.constexpr = HV * K
    T_max = T
    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_h = i_nh // HV, i_nh % HV
        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            T = (eos - bos).to(tl.int32)
            NT = tl.cdiv(T, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int64)
        else:
            bos = tl.cast(i_n, tl.int64) * T
            NT = tl.cdiv(T, BT)
            boh = tl.cast(i_n, tl.int64) * NT

        v_start = i_v * BV
        # rebind GM bases per task to avoid Ascend MTE faults from accumulated pointer offsets.
        w_base = w + (bos * HV + i_h) * K
        k_base = k + (bos * H + i_h // (HV // H)) * K
        v_base = v + (bos * HV + i_h) * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_h) * V
        h_nh = h + (boh * HV + i_h) * K * V

        if USE_G:
            if USE_G_PRECOMP:
                g_ratio_ptr = g_ratio + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_last_exp_ptr = g_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT
        if USE_GK:
            if USE_GK_PRECOMP:
                gk_last_exp_nh = gk_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT * K

        # state tiles are [BK, BV], or [BV, BK] with STATE_V_FIRST.
        if STATE_V_FIRST:
            b_h1 = tl.zeros([BV, BK], dtype=tl.float32)
            if K > BK:
                b_h2 = tl.zeros([BV, BK], dtype=tl.float32)
            if K > BK * 2:
                b_h3 = tl.zeros([BV, BK], dtype=tl.float32)
            if K > BK * 3:
                b_h4 = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
            if K > BK:
                b_h2 = tl.zeros([BK, BV], dtype=tl.float32)
            if K > BK * 2:
                b_h3 = tl.zeros([BK, BV], dtype=tl.float32)
            if K > BK * 3:
                b_h4 = tl.zeros([BK, BV], dtype=tl.float32)

        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            if STATE_V_FIRST:
                # h0 layout [V, K]: block (V_seg=BV, K_seg=BK)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_h0_1 = m_v[:, None] & m_i[None, :]
                p_h0_1 = h0_ptr + o_v[:, None] * K + o_i[None, :]
                b_h1 += mload(p_h0_1, m_p_h0_1, MASK_DMA).to(tl.float32)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_2 = m_v[:, None] & m_k[None, :]
                    p_h0_2 = h0_ptr + o_v[:, None] * K + o_k[None, :]
                    b_h2 += mload(p_h0_2, m_p_h0_2, MASK_DMA).to(tl.float32)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_3 = m_v[:, None] & m_k[None, :]
                    p_h0_3 = h0_ptr + o_v[:, None] * K + o_k[None, :]
                    b_h3 += mload(p_h0_3, m_p_h0_3, MASK_DMA).to(tl.float32)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_4 = m_v[:, None] & m_k[None, :]
                    p_h0_4 = h0_ptr + o_v[:, None] * K + o_k[None, :]
                    b_h4 += mload(p_h0_4, m_p_h0_4, MASK_DMA).to(tl.float32)
            else:
                # h0 layout [K, V]: block (K_seg=BK, V_seg=BV)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_h0_1 = m_i[:, None] & m_v[None, :]
                p_h0_1 = h0_ptr + o_i[:, None] * V + o_v[None, :]
                b_h1 += mload(p_h0_1, m_p_h0_1, MASK_DMA).to(tl.float32)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_2 = m_k[:, None] & m_v[None, :]
                    p_h0_2 = h0_ptr + o_k[:, None] * V + o_v[None, :]
                    b_h2 += mload(p_h0_2, m_p_h0_2, MASK_DMA).to(tl.float32)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_3 = m_k[:, None] & m_v[None, :]
                    p_h0_3 = h0_ptr + o_k[:, None] * V + o_v[None, :]
                    b_h3 += mload(p_h0_3, m_p_h0_3, MASK_DMA).to(tl.float32)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h0_4 = m_k[:, None] & m_v[None, :]
                    p_h0_4 = h0_ptr + o_k[:, None] * V + o_v[None, :]
                    b_h4 += mload(p_h0_4, m_p_h0_4, MASK_DMA).to(tl.float32)

        # main recurrence
        for i_t in range(NT):
            h_base = h_nh + tl.cast(i_t, tl.int64) * DH_CS

            if STATE_V_FIRST:
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_h1 = m_v[:, None] & m_i[None, :]
                p_h1 = h_base + o_v[:, None] * K + o_i[None, :]
                mstore(p_h1, b_h1.to(p_h1.dtype.element_ty), m_p_h1, MASK_DMA)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h2 = m_v[:, None] & m_k[None, :]
                    p_h2 = h_base + o_v[:, None] * K + o_k[None, :]
                    mstore(p_h2, b_h2.to(p_h2.dtype.element_ty), m_p_h2, MASK_DMA)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h3 = m_v[:, None] & m_k[None, :]
                    p_h3 = h_base + o_v[:, None] * K + o_k[None, :]
                    mstore(p_h3, b_h3.to(p_h3.dtype.element_ty), m_p_h3, MASK_DMA)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h4 = m_v[:, None] & m_k[None, :]
                    p_h4 = h_base + o_v[:, None] * K + o_k[None, :]
                    mstore(p_h4, b_h4.to(p_h4.dtype.element_ty), m_p_h4, MASK_DMA)
            else:
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_h1 = m_i[:, None] & m_v[None, :]
                p_h1 = h_base + o_i[:, None] * V + o_v[None, :]
                mstore(p_h1, b_h1.to(p_h1.dtype.element_ty), m_p_h1, MASK_DMA)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h2 = m_k[:, None] & m_v[None, :]
                    p_h2 = h_base + o_k[:, None] * V + o_v[None, :]
                    mstore(p_h2, b_h2.to(p_h2.dtype.element_ty), m_p_h2, MASK_DMA)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h3 = m_k[:, None] & m_v[None, :]
                    p_h3 = h_base + o_k[:, None] * V + o_v[None, :]
                    mstore(p_h3, b_h3.to(p_h3.dtype.element_ty), m_p_h3, MASK_DMA)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_h4 = m_k[:, None] & m_v[None, :]
                    p_h4 = h_base + o_k[:, None] * V + o_v[None, :]
                    mstore(p_h4, b_h4.to(p_h4.dtype.element_ty), m_p_h4, MASK_DMA)

            # load w (K-segmented), accumulate b_v = sum_k dot(b_w_k, b_h_k)
            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            m_t_2 = (o_t >= 0) & (o_t < T)
            o_i = tl.arange(0, BK)
            m_i = (o_i >= 0) & (o_i < K)
            m_p_w1 = m_t_2[:, None] & m_i[None, :]
            p_w1 = w_base + o_t[:, None] * stride_w + o_i[None, :]
            b_w = mload(p_w1, m_p_w1, MASK_DMA).to(tl.float32)
            if STATE_V_FIRST:
                b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
            else:
                b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > BK:
                o_k = BK + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_w2 = m_t_2[:, None] & m_k[None, :]
                p_w2 = w_base + o_t[:, None] * stride_w + o_k[None, :]
                b_w = mload(p_w2, m_p_w2, MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_v = tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype), b_v)
                else:
                    b_v = tl.dot(b_w, b_h2.to(b_w.dtype), b_v)
            if K > BK * 2:
                o_k = BK * 2 + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_w3 = m_t_2[:, None] & m_k[None, :]
                p_w3 = w_base + o_t[:, None] * stride_w + o_k[None, :]
                b_w = mload(p_w3, m_p_w3, MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_v = tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype), b_v)
                else:
                    b_v = tl.dot(b_w, b_h3.to(b_w.dtype), b_v)
            if K > BK * 3:
                o_k = BK * 3 + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_w4 = m_t_2[:, None] & m_k[None, :]
                p_w4 = w_base + o_t[:, None] * stride_w + o_k[None, :]
                b_w = mload(p_w4, m_p_w4, MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_v = tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype), b_v)
                else:
                    b_v = tl.dot(b_w, b_h4.to(b_w.dtype), b_v)

            if USE_G:
                if USE_G_PRECOMP:
                    p_g_ratio = g_ratio_ptr + o_t
                    b_g_ratio = mload(p_g_ratio, m_t_2, MASK_DMA).to(tl.float32)
                    b_g_last = tl.load(g_last_exp_ptr + i_t).to(tl.float32)
                else:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    # g is transposed to [B, HV, T] in wrapper for contiguous T-load.
                    if IS_VARLEN:
                        g_ptr = g + bos + i_h * T_max
                    else:
                        g_ptr = g + tl.cast(i_n, tl.int64) * HV * T_max + i_h * T_max
                    b_g_last = tl.load(g_ptr + last_idx).to(tl.float32)
                    p_g = g_ptr + o_t
                    b_g = mload(p_g, m_t_2, MASK_DMA).to(tl.float32)
                    m_t = (tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)) < T

            # load v and compute v_new = v - b_v
            o_v = v_start + tl.arange(0, BV)
            m_v = (o_v >= 0) & (o_v < V)
            m_p_v = m_t_2[:, None] & m_v[None, :]
            p_v = v_base + o_t[:, None] * stride_v + o_v[None, :]
            b_v = mload(p_v, m_p_v, MASK_DMA).to(tl.float32) - b_v

            if SAVE_NEW_VALUE:
                p_v_new = v_new_base + o_t[:, None] * stride_v + o_v[None, :]
                mstore(p_v_new, b_v.to(p_v_new.dtype.element_ty), m_p_v, MASK_DMA)

            if USE_G:
                if USE_G_PRECOMP:
                    b_v = b_v * b_g_ratio[:, None]
                else:
                    b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                b_h1 *= b_g_last
                if K > BK:
                    b_h2 *= b_g_last
                if K > BK * 2:
                    b_h3 *= b_g_last
                if K > BK * 3:
                    b_h4 *= b_g_last

            if USE_GK:
                o_k1 = tl.arange(0, BK)
                if USE_GK_PRECOMP:
                    # gk_last_exp layout [B, HV, NT, K]
                    gk_base = gk_last_exp_nh + tl.cast(i_t, tl.int64) * K
                    b_gk_last1 = mload(gk_base + o_k1, (o_k1 < K), MASK_DMA).to(tl.float32)
                else:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    gk_base = gk + (bos + last_idx) * HV * K + i_h * K
                    b_gk_last1 = exp2(mload(gk_base + o_k1, (o_k1 < K), MASK_DMA).to(tl.float32))
                if STATE_V_FIRST:
                    b_h1 *= b_gk_last1[None, :]
                else:
                    b_h1 *= b_gk_last1[:, None]
                if K > BK:
                    o_k2 = BK + o_k1
                    if USE_GK_PRECOMP:
                        b_gk_last2 = mload(gk_base + o_k2, (o_k2 < K), MASK_DMA).to(tl.float32)
                    else:
                        b_gk_last2 = exp2(mload(gk_base + o_k2, (o_k2 < K), MASK_DMA).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h2 *= b_gk_last2[None, :]
                    else:
                        b_h2 *= b_gk_last2[:, None]
                if K > BK * 2:
                    o_k3 = BK * 2 + o_k1
                    if USE_GK_PRECOMP:
                        b_gk_last3 = mload(gk_base + o_k3, (o_k3 < K), MASK_DMA).to(tl.float32)
                    else:
                        b_gk_last3 = exp2(mload(gk_base + o_k3, (o_k3 < K), MASK_DMA).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h3 *= b_gk_last3[None, :]
                    else:
                        b_h3 *= b_gk_last3[:, None]
                if K > BK * 3:
                    o_k4 = BK * 3 + o_k1
                    if USE_GK_PRECOMP:
                        b_gk_last4 = mload(gk_base + o_k4, (o_k4 < K), MASK_DMA).to(tl.float32)
                    else:
                        b_gk_last4 = exp2(mload(gk_base + o_k4, (o_k4 < K), MASK_DMA).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h4 *= b_gk_last4[None, :]
                    else:
                        b_h4 *= b_gk_last4[:, None]

            # load k (K-segmented), update b_h += dot(b_k_seg, b_v)
            m_p_k1 = m_i[:, None] & m_t_2[None, :]
            p_k1 = k_base + o_i[:, None] + o_t[None, :] * stride_k
            b_k = mload(p_k1, m_p_k1, MASK_DMA).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h1 = tl.dot(b_k, b_v, b_h1)
            if K > BK:
                o_k = BK + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_k2 = m_k[:, None] & m_t_2[None, :]
                p_k2 = k_base + o_k[:, None] + o_t[None, :] * stride_k
                b_k = mload(p_k2, m_p_k2, MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h2 = tl.dot(b_k, b_v, b_h2)
            if K > BK * 2:
                o_k = BK * 2 + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_k3 = m_k[:, None] & m_t_2[None, :]
                p_k3 = k_base + o_k[:, None] + o_t[None, :] * stride_k
                b_k = mload(p_k3, m_p_k3, MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h3 = tl.dot(b_k, b_v, b_h3)
            if K > BK * 3:
                o_k = BK * 3 + tl.arange(0, BK)
                m_k = (o_k >= 0) & (o_k < K)
                m_p_k4 = m_k[:, None] & m_t_2[None, :]
                p_k4 = k_base + o_k[:, None] + o_t[None, :] * stride_k
                b_k = mload(p_k4, m_p_k4, MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h4 = tl.dot(b_k, b_v, b_h4)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V

            if STATE_V_FIRST:
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_ht1 = m_v[:, None] & m_i[None, :]
                p_ht1 = ht_ptr + o_v[:, None] * K + o_i[None, :]
                mstore(p_ht1, b_h1.to(p_ht1.dtype.element_ty), m_p_ht1, MASK_DMA)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht2 = m_v[:, None] & m_k[None, :]
                    p_ht2 = ht_ptr + o_v[:, None] * K + o_k[None, :]
                    mstore(p_ht2, b_h2.to(p_ht2.dtype.element_ty), m_p_ht2, MASK_DMA)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht3 = m_v[:, None] & m_k[None, :]
                    p_ht3 = ht_ptr + o_v[:, None] * K + o_k[None, :]
                    mstore(p_ht3, b_h3.to(p_ht3.dtype.element_ty), m_p_ht3, MASK_DMA)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht4 = m_v[:, None] & m_k[None, :]
                    p_ht4 = ht_ptr + o_v[:, None] * K + o_k[None, :]
                    mstore(p_ht4, b_h4.to(p_ht4.dtype.element_ty), m_p_ht4, MASK_DMA)
            else:
                o_i = tl.arange(0, BK)
                m_i = (o_i >= 0) & (o_i < K)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_ht1 = m_i[:, None] & m_v[None, :]
                p_ht1 = ht_ptr + o_i[:, None] * V + o_v[None, :]
                mstore(p_ht1, b_h1.to(p_ht1.dtype.element_ty), m_p_ht1, MASK_DMA)
                if K > BK:
                    o_k = BK + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht2 = m_k[:, None] & m_v[None, :]
                    p_ht2 = ht_ptr + o_k[:, None] * V + o_v[None, :]
                    mstore(p_ht2, b_h2.to(p_ht2.dtype.element_ty), m_p_ht2, MASK_DMA)
                if K > BK * 2:
                    o_k = BK * 2 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht3 = m_k[:, None] & m_v[None, :]
                    p_ht3 = ht_ptr + o_k[:, None] * V + o_v[None, :]
                    mstore(p_ht3, b_h3.to(p_ht3.dtype.element_ty), m_p_ht3, MASK_DMA)
                if K > BK * 3:
                    o_k = BK * 3 + tl.arange(0, BK)
                    m_k = (o_k >= 0) & (o_k < K)
                    m_p_ht4 = m_k[:, None] & m_v[None, :]
                    p_ht4 = ht_ptr + o_k[:, None] * V + o_v[None, :]
                    mstore(p_ht4, b_h4.to(p_ht4.dtype.element_ty), m_p_ht4, MASK_DMA)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_ratio"] is not None,
        "USE_GK": lambda args: args["gk_last_exp"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_gated_delta_rule_h_fwd_kernel_single_slab(
    k,
    v,
    w,
    v_new,
    g_ratio,
    g_last_exp,
    gk_last_exp,
    h,
    h0,
    ht,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
):
    """Update K-first states for aligned dense sequences with one K slab and host-precomputed gates."""
    core_id = tl.program_id(0)
    DH_CS: tl.constexpr = HV * K * V
    stride_v: tl.constexpr = HV * V
    stride_k: tl.constexpr = H * K
    stride_w: tl.constexpr = HV * K
    NV: tl.constexpr = tl.cdiv(V, BV)
    NT = T // BT
    if USE_GK:
        o_k = tl.arange(0, BK)

    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_h = i_nh // HV, i_nh % HV
        bos = tl.cast(i_n, tl.int64) * T
        v_start = i_v * BV

        w_base = w + (bos * HV + i_h) * K
        k_base = k + (bos * H + i_h // (HV // H)) * K
        v_base = v + (bos * HV + i_h) * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_h) * V
        h_nh = h + (tl.cast(i_n, tl.int64) * NT * HV + i_h) * K * V

        if USE_G:
            g_ratio_ptr = g_ratio + (tl.cast(i_n, tl.int64) * HV + i_h) * T
            g_last_exp_ptr = g_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT
        if USE_GK:
            gk_last_exp_nh = gk_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT * K

        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            o_i = tl.arange(0, BK)
            o_v = v_start + tl.arange(0, BV)
            p_h0_1 = h0_ptr + o_i[:, None] * V + o_v[None, :]
            b_h1 += tl.load(p_h0_1).to(tl.float32)

        for i_t in range(NT):
            h_base = h_nh + tl.cast(i_t, tl.int64) * DH_CS
            o_i = tl.arange(0, BK)
            o_v = v_start + tl.arange(0, BV)
            p_h1 = h_base + o_i[:, None] * V + o_v[None, :]
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty))

            o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            p_w1 = w_base + o_t[:, None] * stride_w + o_i[None, :]
            b_w = tl.load(p_w1).to(tl.float32)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

            if USE_G:
                p_g_ratio = g_ratio_ptr + o_t
                b_g_ratio = tl.load(p_g_ratio).to(tl.float32)
                b_g_last = tl.load(g_last_exp_ptr + i_t).to(tl.float32)

            p_v = v_base + o_t[:, None] * stride_v + o_v[None, :]
            b_v = tl.load(p_v).to(tl.float32) - b_v
            if SAVE_NEW_VALUE:
                p_v_new = v_new_base + o_t[:, None] * stride_v + o_v[None, :]
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty))

            if USE_G:
                b_v = b_v * b_g_ratio[:, None]
                b_h1 *= b_g_last

            if USE_GK:
                gk_base = gk_last_exp_nh + tl.cast(i_t, tl.int64) * K
                b_gk_last1 = tl.load(gk_base + o_k).to(tl.float32)
                b_h1 *= b_gk_last1[:, None]

            p_k1 = k_base + o_i[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k1).to(tl.float32)
            b_h1 = tl.dot(b_k, b_v, b_h1)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V
            o_i = tl.arange(0, BK)
            o_v = v_start + tl.arange(0, BV)
            p_ht1 = ht_ptr + o_i[:, None] * V + o_v[None, :]
            tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty))


def _prepare_fwd_g_gates(
    g: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Prepare contiguous scalar gates, precomputing fp32 scales for aligned dense chunks."""
    if g is None:
        return None, None, None
    g_log = g.transpose(1, 2).contiguous()
    if cu_seqlens is None and (T % BT == 0):
        NT = T // BT
        g_f = g_log.float()
        g_view = g_f.view(B, HV, NT, BT)
        g_last = g_view[:, :, :, -1:]
        # fp32 required: bf16 precomp drifts through the recurrence at large NT.
        g_ratio = torch.exp2(g_last - g_view).reshape(B, HV, T)
        g_last_exp = torch.exp2(g_last).reshape(B, HV, NT)
        return None, g_ratio, g_last_exp
    return g_log, None, None


def _prepare_fwd_gk_last_exp(
    gk: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    K: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Precompute fp32 key-gate scales at chunk ends for aligned dense sequences."""
    if gk is None:
        return None, None
    if cu_seqlens is None and (T % BT == 0):
        NT = T // BT
        gk_last = gk.view(B, NT, BT, HV, K)[:, :, -1].permute(0, 2, 1, 3).contiguous()
        return None, torch.exp2(gk_last.float())
    return gk, None


@input_guard
def chunk_gated_delta_rule_fwd_h_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens=cu_seqlens, chunk_size=BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    g, g_ratio, g_last_exp = _prepare_fwd_g_gates(g=g, B=B, T=T, HV=HV, BT=BT, cu_seqlens=cu_seqlens)
    gk, gk_last_exp = _prepare_fwd_gk_last_exp(gk=gk, B=B, T=T, HV=HV, K=K, BT=BT, cu_seqlens=cu_seqlens)

    # blockdim64 keeps T/K/V masks (and inline exp2 when T % BT != 0).
    masked = cu_seqlens is not None or T % BT != 0
    BK, BV = _select_fwd_h_tiles(K, V, state_v_first, masked=masked)
    oneslab = (
        cu_seqlens is None
        and not state_v_first
        and T % BT == 0
        and K == BK
        and V % BV == 0
    )
    kwargs = dict(
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g_ratio=g_ratio,
        g_last_exp=g_last_exp,
        gk_last_exp=gk_last_exp,
        h=h,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    if oneslab:
        _launch_core_grid(
            kernel=chunk_gated_delta_rule_h_fwd_kernel_single_slab,
            task_num=N * HV * triton.cdiv(V, BV),
            kernel_kwargs={**kwargs},
            **_FWD_H_COMPILE,
        )
    else:
        _launch_core_grid(
            kernel=chunk_gated_delta_rule_h_fwd_kernel,
            task_num=N * HV * triton.cdiv(V, BV),
            kernel_kwargs={
                **kwargs,
                "g": g,
                "gk": gk,
                "cu_seqlens": cu_seqlens,
                "chunk_offsets": chunk_offsets,
                "STATE_V_FIRST": state_v_first,
            },
            **_FWD_H_COMPILE,
        )
    return h, v_new, final_state


# Keep scale a runtime fp32. A Python int (tests pass scale=1) is specialized
# and folded into `b_do * scale`; that compile hangs the masked D=100 kernel.
@triton.jit(do_not_specialize=["T", "task_num", "num_core", "scale"])
def chunk_gated_delta_rule_h_bwd_kernel(
    q,
    k,
    w,
    g,
    g_exp,
    g_ratio,
    g_last_exp,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    task_num,
    num_core,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_PRECOMP: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MASK_DMA: tl.constexpr,
):
    core_id = tl.program_id(0)
    T_max = T
    for task_id in tl.range(core_id, task_num, num_core):
        i_nh = task_id.to(tl.int64)
        i_n, i_h = i_nh // HV, i_nh % HV
        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            T = (eos - bos).to(tl.int32)
            NT = tl.cdiv(T, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int64)
        else:
            bos = tl.cast(i_n, tl.int64) * T_max
            eos = bos + T_max
            NT = tl.cdiv(T, BT)
            boh = tl.cast(i_n, tl.int64) * NT

        stride_v = HV * V
        stride_k = H * K
        stride_w = HV * K

        q_base = q + (bos * H + i_h // (HV // H)) * K
        k_base = k + (bos * H + i_h // (HV // H)) * K
        w_base = w + (bos * HV + i_h) * K
        do_base = do + (bos * HV + i_h) * V
        dv_base = dv + (bos * HV + i_h) * V
        dv2_base = dv2 + (bos * HV + i_h) * V
        dh_base = dh + (boh * HV + i_h) * K * V
        if USE_GK:
            gk_base = gk + (bos * HV + i_h) * K
        if USE_G:
            if USE_G_PRECOMP:
                g_exp_base = g_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_ratio_base = g_ratio + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_last_exp_base = g_last_exp + (tl.cast(i_n, tl.int64) * HV + i_h) * NT
                g_base = g
            elif IS_VARLEN:
                g_base = g + (bos + tl.cast(i_h, tl.int64) * T_max)
                g_exp_base = g_exp
                g_ratio_base = g_ratio
                g_last_exp_base = g_last_exp
            else:
                g_base = g + (tl.cast(i_n, tl.int64) * HV + i_h) * T_max
                g_exp_base = g_exp
                g_ratio_base = g_ratio
                g_last_exp_base = g_last_exp
        else:
            g_base = g
            g_exp_base = g_exp
            g_ratio_base = g_ratio
            g_last_exp_base = g_last_exp

        NV = tl.cdiv(V, BV)
        for i_v in range(NV):
            v_start = i_v * BV

            if STATE_V_FIRST:
                b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 64:
                    b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 128:
                    b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 192:
                    b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
            else:
                b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 64:
                    b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 128:
                    b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 192:
                    b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

            if USE_FINAL_STATE_GRADIENT:
                dht_base = dht + i_nh * K * V
                if STATE_V_FIRST:
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_dht1 = m_v[:, None] & m_i[None, :]
                    p_dht1 = dht_base + o_v[:, None] * K + o_i[None, :]
                else:
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dht1 = m_i[:, None] & m_v[None, :]
                    p_dht1 = dht_base + o_i[:, None] * V + o_v[None, :]
                b_dh1 += mload(p_dht1, m_p_dht1, MASK_DMA)
                if K > 64:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dht2 = m_v[:, None] & m_i[None, :]
                        p_dht2 = dht_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dht2 = m_i[:, None] & m_v[None, :]
                        p_dht2 = dht_base + o_i[:, None] * V + o_v[None, :]
                    b_dh2 += mload(p_dht2, m_p_dht2, MASK_DMA)
                if K > 128:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dht3 = m_v[:, None] & m_i[None, :]
                        p_dht3 = dht_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dht3 = m_i[:, None] & m_v[None, :]
                        p_dht3 = dht_base + o_i[:, None] * V + o_v[None, :]
                    b_dh3 += mload(p_dht3, m_p_dht3, MASK_DMA)
                if K > 192:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dht4 = m_v[:, None] & m_i[None, :]
                        p_dht4 = dht_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dht4 = m_i[:, None] & m_v[None, :]
                        p_dht4 = dht_base + o_i[:, None] * V + o_v[None, :]
                    b_dh4 += mload(p_dht4, m_p_dht4, MASK_DMA)

            if USE_GK:
                o_k1 = tl.arange(0, 64)

            DH_CS: tl.constexpr = HV * K * V
            dh_chunk = dh_base + (NT - 1).to(tl.int64) * DH_CS

            for i_t in range(NT - 1, -1, -1):
                if STATE_V_FIRST:
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_dh1 = m_v[:, None] & m_i[None, :]
                    p_dh1 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                else:
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dh1 = m_i[:, None] & m_v[None, :]
                    p_dh1 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                mstore(p_dh1,  b_dh1.to(p_dh1.dtype.element_ty), m_p_dh1, MASK_DMA)
                if K > 64:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh2 = m_v[:, None] & m_i[None, :]
                        p_dh2 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh2 = m_i[:, None] & m_v[None, :]
                        p_dh2 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh2,  b_dh2.to(p_dh2.dtype.element_ty), m_p_dh2, MASK_DMA)
                if K > 128:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh3 = m_v[:, None] & m_i[None, :]
                        p_dh3 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh3 = m_i[:, None] & m_v[None, :]
                        p_dh3 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh3,  b_dh3.to(p_dh3.dtype.element_ty), m_p_dh3, MASK_DMA)
                if K > 192:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh4 = m_v[:, None] & m_i[None, :]
                        p_dh4 = dh_chunk + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh4 = m_i[:, None] & m_v[None, :]
                        p_dh4 = dh_chunk + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh4,  b_dh4.to(p_dh4.dtype.element_ty), m_p_dh4, MASK_DMA)

                if USE_G:
                    if USE_G_PRECOMP:
                        bg_last_exp = tl.load(g_last_exp_base + i_t).to(tl.float32)
                        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
                        m_t_2 = (o_t >= 0) & (o_t < T)
                        p_g_exp = g_exp_base + o_t
                        p_g_ratio = g_ratio_base + o_t
                        b_g_exp = mload(p_g_exp, m_t_2, MASK_DMA).to(tl.float32)
                        b_g_ratio = mload(p_g_ratio, m_t_2, MASK_DMA).to(tl.float32)
                    else:
                        last_idx = min((i_t + 1) * BT, T) - 1
                        bg_last = tl.load(g_base + last_idx).to(tl.float32)
                        o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
                        m_t_2 = (o_t >= 0) & (o_t < T)
                        p_g = g_base + o_t
                        b_g = mload(p_g, m_t_2, MASK_DMA).to(tl.float32)
                        bg_last_exp = exp2(bg_last)
                        b_g_exp = exp2(b_g)
                        b_g_ratio = exp2(bg_last - b_g)
                o_t = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
                m_t_2 = (o_t >= 0) & (o_t < T)
                o_v = v_start + tl.arange(0, BV)
                m_v = (o_v >= 0) & (o_v < V)
                m_p_dv = m_t_2[:, None] & m_v[None, :]
                p_dv = dv_base + o_t[:, None] * stride_v + o_v[None, :]
                p_dv2 = dv2_base + o_t[:, None] * stride_v + o_v[None, :]
                p_do = do_base + o_t[:, None] * stride_v + o_v[None, :]
                b_do = mload(p_do, m_p_dv, MASK_DMA)

                o_i = tl.arange(0, 64)
                m_i = (o_i >= 0) & (o_i < K)
                m_p_k = m_t_2[:, None] & m_i[None, :]
                p_k = k_base + o_t[:, None] * stride_k + o_i[None, :]
                b_k = mload(p_k, m_p_k, MASK_DMA)
                if USE_GK:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    b_gk_last1 = mload(gk_base + last_idx * stride_w + o_k1, (o_k1 < K), MASK_DMA).to(tl.float32)
                if STATE_V_FIRST:
                    b_dv = tl.dot(b_dh1.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                    b_dv = tl.trans(b_dv)
                else:
                    b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), allow_tf32=False)

                if K > 64:
                    o_i_2 = 64 + tl.arange(0, 64)
                    m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                    m_p_k = m_t_2[:, None] & m_i_2[None, :]
                    p_k = k_base + o_t[:, None] * stride_k + o_i_2[None, :]
                    b_k = mload(p_k, m_p_k, MASK_DMA)
                    if USE_GK:
                        o_k2 = 64 + o_k1
                        b_gk_last2 = mload(gk_base + last_idx * stride_w + o_k2, (o_k2 < K), MASK_DMA).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv_part = tl.dot(b_dh2.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv += tl.trans(b_dv_part)
                    else:
                        b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), b_dv, allow_tf32=False)

                if K > 128:
                    o_i_2 = 128 + tl.arange(0, 64)
                    m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                    m_p_k = m_t_2[:, None] & m_i_2[None, :]
                    p_k = k_base + o_t[:, None] * stride_k + o_i_2[None, :]
                    b_k = mload(p_k, m_p_k, MASK_DMA)
                    if USE_GK:
                        o_k3 = 128 + o_k1
                        b_gk_last3 = mload(gk_base + last_idx * stride_w + o_k3, (o_k3 < K), MASK_DMA).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv_part = tl.dot(b_dh3.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv += tl.trans(b_dv_part)
                    else:
                        b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), b_dv, allow_tf32=False)

                if K > 192:
                    o_i_2 = 192 + tl.arange(0, 64)
                    m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                    m_p_k = m_t_2[:, None] & m_i_2[None, :]
                    p_k = k_base + o_t[:, None] * stride_k + o_i_2[None, :]
                    b_k = mload(p_k, m_p_k, MASK_DMA)
                    if USE_GK:
                        o_k4 = 192 + o_k1
                        b_gk_last4 = mload(gk_base + last_idx * stride_w + o_k4, (o_k4 < K), MASK_DMA).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv_part = tl.dot(b_dh4.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv += tl.trans(b_dv_part)
                    else:
                        b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), b_dv, allow_tf32=False)

                # Cube dots leave NaN in masked token lanes. Those lanes are the
                # reduction axis of the dh update, and NaN * 0 stays NaN, so drop
                # them with a select. Aligned launches have no such lanes; a select
                # there keeps an extra [BT, BV] tile and overflows UB at D128.
                if MASK_DMA:
                    if USE_G:
                        b_dv = tl.where(m_t_2[:, None], b_dv * b_g_ratio[:, None], 0)
                    else:
                        b_dv = tl.where(m_t_2[:, None], b_dv, 0)
                elif USE_G:
                    b_dv *= b_g_ratio[:, None]
                b_dv += mload(p_dv, m_p_dv, MASK_DMA)
                mstore(p_dv2,  b_dv.to(p_dv2.dtype.element_ty), m_p_dv, MASK_DMA)
                # v-first uses dv as lhs of the subtract dot, which clobbers it.
                if STATE_V_FIRST:
                    b_dv_pristine = b_dv + 0.0

                if USE_G:
                    b_dh1 *= bg_last_exp
                    if K > 64:
                        b_dh2 *= bg_last_exp
                    if K > 128:
                        b_dh3 *= bg_last_exp
                    if K > 192:
                        b_dh4 *= bg_last_exp
                    b_do = b_do * (b_g_exp * scale)[:, None]
                else:
                    b_do = b_do * scale
                # v-first feeds do as lhs, so each K-slab needs its own copy.
                # Non-v-first uses do only as rhs; one buffer is enough.
                if STATE_V_FIRST:
                    b_do_c = b_do + 0.0
                    if K > 128:
                        b_do_c2 = b_do + 0.0
                    if K > 192:
                        b_do_c3 = b_do + 0.0

                m_p_w = m_i[:, None] & m_t_2[None, :]
                p_w = w_base + o_i[:, None] + o_t[None, :] * stride_w
                p_q = q_base + o_i[:, None] + o_t[None, :] * stride_k
                b_w = mload(p_w, m_p_w, MASK_DMA)
                b_q = mload(p_q, m_p_w, MASK_DMA)
                if USE_GK:
                    if STATE_V_FIRST:
                        b_dh1 *= exp2(b_gk_last1)[None, :]
                    else:
                        b_dh1 *= exp2(b_gk_last1[:, None])
                if STATE_V_FIRST:
                    b_dh1 = tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), b_dh1, allow_tf32=False)
                    b_dv_i = b_dv_pristine + 0.0
                    b_dh1 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                else:
                    b_dh1 += (
                        tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False)
                        - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
                    )

                if K > 64:
                    o_i_2 = 64 + tl.arange(0, 64)
                    m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                    m_p_q = m_i_2[:, None] & m_t_2[None, :]
                    p_q = q_base + o_i_2[:, None] + o_t[None, :] * stride_k
                    p_w = w_base + o_i_2[:, None] + o_t[None, :] * stride_w
                    b_q = mload(p_q, m_p_q, MASK_DMA)
                    b_w = mload(p_w, m_p_q, MASK_DMA)
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh2 *= exp2(b_gk_last2)[None, :]
                        else:
                            b_dh2 *= exp2(b_gk_last2[:, None])
                    if STATE_V_FIRST:
                        b_dh2 = tl.dot(tl.trans(b_do_c.to(b_q.dtype)), tl.trans(b_q), b_dh2, allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh2 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                            b_w, b_dv.to(b_w.dtype), allow_tf32=False
                        )

                if K > 128:
                    o_i_2 = 128 + tl.arange(0, 64)
                    m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                    m_p_q = m_i_2[:, None] & m_t_2[None, :]
                    p_q = q_base + o_i_2[:, None] + o_t[None, :] * stride_k
                    p_w = w_base + o_i_2[:, None] + o_t[None, :] * stride_w
                    b_q = mload(p_q, m_p_q, MASK_DMA)
                    b_w = mload(p_w, m_p_q, MASK_DMA)
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh3 *= exp2(b_gk_last3)[None, :]
                        else:
                            b_dh3 *= exp2(b_gk_last3[:, None])
                    if STATE_V_FIRST:
                        b_dh3 = tl.dot(tl.trans(b_do_c2.to(b_q.dtype)), tl.trans(b_q), b_dh3, allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh3 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                            b_w, b_dv.to(b_w.dtype), allow_tf32=False
                        )

                if K > 192:
                    o_i_2 = 192 + tl.arange(0, 64)
                    m_i_2 = (o_i_2 >= 0) & (o_i_2 < K)
                    m_p_q = m_i_2[:, None] & m_t_2[None, :]
                    p_q = q_base + o_i_2[:, None] + o_t[None, :] * stride_k
                    p_w = w_base + o_i_2[:, None] + o_t[None, :] * stride_w
                    b_q = mload(p_q, m_p_q, MASK_DMA)
                    b_w = mload(p_w, m_p_q, MASK_DMA)
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh4 *= exp2(b_gk_last4)[None, :]
                        else:
                            b_dh4 *= exp2(b_gk_last4[:, None])
                    if STATE_V_FIRST:
                        b_dh4 = tl.dot(tl.trans(b_do_c3.to(b_q.dtype)), tl.trans(b_q), b_dh4, allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh4 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                            b_w, b_dv.to(b_w.dtype), allow_tf32=False
                        )

                dh_chunk -= DH_CS

            if USE_INITIAL_STATE:
                dh0_base = dh0 + i_nh * K * V
                if STATE_V_FIRST:
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    m_p_dh0 = m_v[:, None] & m_i[None, :]
                    p_dh0 = dh0_base + o_v[:, None] * K + o_i[None, :]
                else:
                    o_i = tl.arange(0, 64)
                    m_i = (o_i >= 0) & (o_i < K)
                    o_v = v_start + tl.arange(0, BV)
                    m_v = (o_v >= 0) & (o_v < V)
                    m_p_dh0 = m_i[:, None] & m_v[None, :]
                    p_dh0 = dh0_base + o_i[:, None] * V + o_v[None, :]
                mstore(p_dh0,  b_dh1.to(p_dh0.dtype.element_ty), m_p_dh0, MASK_DMA)
                if K > 64:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh1 = m_v[:, None] & m_i[None, :]
                        p_dh1 = dh0_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 64 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh1 = m_i[:, None] & m_v[None, :]
                        p_dh1 = dh0_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh1,  b_dh2.to(p_dh1.dtype.element_ty), m_p_dh1, MASK_DMA)
                if K > 128:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh2 = m_v[:, None] & m_i[None, :]
                        p_dh2 = dh0_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 128 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh2 = m_i[:, None] & m_v[None, :]
                        p_dh2 = dh0_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh2,  b_dh3.to(p_dh2.dtype.element_ty), m_p_dh2, MASK_DMA)
                if K > 192:
                    if STATE_V_FIRST:
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        m_p_dh3 = m_v[:, None] & m_i[None, :]
                        p_dh3 = dh0_base + o_v[:, None] * K + o_i[None, :]
                    else:
                        o_i = 192 + tl.arange(0, 64)
                        m_i = (o_i >= 0) & (o_i < K)
                        o_v = v_start + tl.arange(0, BV)
                        m_v = (o_v >= 0) & (o_v < V)
                        m_p_dh3 = m_i[:, None] & m_v[None, :]
                        p_dh3 = dh0_base + o_i[:, None] * V + o_v[None, :]
                    mstore(p_dh3,  b_dh4.to(p_dh3.dtype.element_ty), m_p_dh3, MASK_DMA)


def _prepare_dhu_gate_tensors(
    g: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool]:
    """Prepare contiguous backward gates, precomputing fp32 scales for aligned dense chunks."""
    if g is None:
        return None, None, None, None, False
    g_log = g.transpose(1, 2).contiguous()
    if cu_seqlens is None and (T % BT == 0):
        NT = T // BT
        g_f = g_log.float()
        g_view = g_f.view(B, HV, NT, BT)
        g_last = g_view[:, :, :, -1:]
        # fp32 required: bf16 precomp drifts through the reverse recurrence at large NT.
        g_ratio = torch.exp2(g_last - g_view).reshape(B, HV, T)
        g_exp = torch.exp2(g_f)
        g_last_exp = torch.exp2(g_last).reshape(B, HV, NT)
        return None, g_exp, g_ratio, g_last_exp, True
    return g_log, None, None, None, False


@input_guard
def chunk_gated_delta_rule_bwd_dhu_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens=cu_seqlens, chunk_size=BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    # keep the caller's dv unchanged while updating the value gradient.
    dv2 = torch.empty_like(dv)
    g_log, g_exp, g_ratio, g_last_exp, use_g_precomp = _prepare_dhu_gate_tensors(
        g=g,
        B=B,
        T=T,
        HV=HV,
        BT=BT,
        cu_seqlens=cu_seqlens,
    )
    gate_inline = g is not None and not use_g_precomp
    masked = cu_seqlens is not None or T % BT != 0
    BK, BV = _select_bwd_dhu_tiles(
        K=K,
        V=V,
        state_v_first=state_v_first,
        gate_inline=gate_inline,
        masked=masked,
    )
    # unit_flag overlaps cube and fixpipe. On the v-first transpose path that
    # overlap deadlocks when the sequence is ragged or packed, and the device
    # sync then times out. Aligned dense launches keep the flag.
    compile_opts = dict(_DHU_COMPILE)
    if state_v_first and masked:
        compile_opts["unit_flag"] = False
    _launch_core_grid(
        kernel=chunk_gated_delta_rule_h_bwd_kernel,
        task_num=N * HV,
        **compile_opts,
        kernel_kwargs={
            "q": q,
            "k": k,
            "w": w,
            "g": g_log,
            "g_exp": g_exp,
            "g_ratio": g_ratio,
            "g_last_exp": g_last_exp,
            "gk": gk,
            "dht": dht,
            "dh0": dh0,
            "do": do,
            "dh": dh,
            "dv": dv,
            "dv2": dv2,
            "cu_seqlens": cu_seqlens,
            "chunk_offsets": chunk_offsets,
            "scale": float(scale),
            "T": T,
            "H": H,
            "HV": HV,
            "K": K,
            "V": V,
            "BT": BT,
            "BK": BK,
            "BV": BV,
            "USE_G": g is not None,
            "USE_G_PRECOMP": use_g_precomp,
            "USE_GK": gk is not None,
            "USE_INITIAL_STATE": h0 is not None,
            "USE_FINAL_STATE_GRADIENT": dht is not None,
            "STATE_V_FIRST": state_v_first,
            "IS_VARLEN": cu_seqlens is not None,
            "MASK_DMA": need_dma_mask(
                T, BT, K, BK, V, BV, cu_seqlens is not None,
            ),
        },
    )
    return dh, dh0, dv2
