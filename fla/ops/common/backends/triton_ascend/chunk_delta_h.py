# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""chunk_gated_delta_rule_fwd_h adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    get_ub_manager,
    max_grid_axis_chunks,
)

_NUM_WARPS = 4
# b_h[64,BV] fp32 + b_w[64,64] + b_v[64,BV] + b_k[64,64] peak during recurrence.
_FWD_H_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BV = 16
_MAX_BV = 64
_DHU_BT = 64
# Soft UB over-admit for host-precomputed gates (live-range can beat peak estimate).
_DHU_UB_SOFT = 1.15
# Tighter fraction when in-kernel exp2(g) inflates live UB (~1.7× analytical).
_DHU_UB_GATE_INLINE = 0.60


def _get_bv(K: int, V: int) -> int:
    return compute_row_tile_block_size(
        min(K, 64),
        V,
        _FWD_H_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BV,
        min_block=16,
        max_block=min(_MAX_BV, triton.next_power_of_2(V)),
    )


def _dhu_peak_bytes(BK: int, BV: int, n_slabs: int, BT: int = _DHU_BT) -> int:
    # All K-slabs of dh stay live; q/w are one-slab at a time.
    return (
        n_slabs * BK * BV * 4
        + 2 * BK * BT * 2
        + BT * BV * 2
        + BT * BV * 4
        + 12 * BT
    )


def _dhu_tile_cost(K: int, V: int, BK: int, BV: int) -> int:
    """Reverse-loop scalar proxy: nv × (base ptrs + ~4 per K-slab)."""
    return triton.cdiv(V, BV) * (3 + 4 * triton.cdiv(K, BK))


def _select_bwd_dhu_tiles(
    K: int,
    V: int,
    state_v_first: bool,
    *,
    gate_inline: bool = False,
) -> tuple[int, int]:
    """Pick (BK, BV) minimizing launch×ptr cost under a soft UB cap.

    ``gate_inline`` covers USE_G without host-precomputed exp2 (unaligned T / varlen).
    """
    if state_v_first:
        return 64, _get_bv(K, V)

    soft_cap = int(get_ub_manager().ub_capacity_bytes * (
        _DHU_UB_GATE_INLINE if gate_inline else _DHU_UB_SOFT
    ))
    # bwd kernel is blockdim64: K is covered by up to four BK=64 slabs only.
    max_bk = 64
    desired_v = triton.next_power_of_2(V)
    best: tuple[int, int, int] | None = None  # cost, BK, BV

    bk = 64
    while bk <= max_bk:
        n_slabs = triton.cdiv(K, bk)
        bv = 16
        while bv <= min(desired_v, 256):
            if _dhu_peak_bytes(bk, bv, n_slabs) <= soft_cap:
                cost = _dhu_tile_cost(K, V, bk, bv)
                if best is None or cost < best[0] or (cost == best[0] and bk > best[1]):
                    best = (cost, bk, bv)
            bv *= 2
        bk *= 2

    if best is None:
        return 64, _get_bv(K, V)
    return best[1], best[2]


def _launch_fwd_h_kernel(kernel, *, nv_chunks: int, nh_total: int, kernel_kwargs: dict) -> None:
    max_nv = max_grid_axis_chunks(nv_chunks, nh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for v_off in range(0, nv_chunks, max_nv):
        v_len = min(max_nv, nv_chunks - v_off)
        kernel_kwargs['V_OFFSET'] = v_off
        max_nh = max_grid_axis_chunks(nh_total, v_len, max_grid=ASCEND_MAX_GRID_DIM)
        for nh_off in range(0, nh_total, max_nh):
            nh_len = min(max_nh, nh_total - nh_off)
            kernel_kwargs['NH_OFFSET'] = nh_off
            kernel[(v_len, nh_len)](num_warps=_NUM_WARPS, **kernel_kwargs)


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': 32}),
        triton.Config({'BV': 64}),
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu(
    k,
    v,
    w,
    v_new,
    g,
    gk,
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
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    core_id = tl.program_id(0)
    T_max = T
    for task_id in tl.range(core_id, task_num, num_core):
        i_nh = task_id
        i_n, i_h = i_nh // HV, i_nh % HV
        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int32),
                tl.load(cu_seqlens + i_n + 1).to(tl.int32),
            )
            T = eos - bos
            NT = tl.cdiv(T, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            bos, eos = i_n * T, i_n * T + T
            NT = tl.cdiv(T, BT)
            boh = i_n * NT

        stride_v = HV * V
        stride_k = H * K
        stride_w = HV * K

        # V-dimension tiling loop: BV autotune (32/64), NV chunks to support V>128
        NV = tl.cdiv(V, BV)
        for i_v in range(NV):
            v_start = i_v * BV

            # K-dimension tiling: 4 segments of K_seg=64, supporting K<=256
            # b_h shape: [K_seg=64, BV] (default) or [BV, K_seg=64] (STATE_V_FIRST)
            if STATE_V_FIRST:
                b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 64:
                    b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 128:
                    b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
                if K > 192:
                    b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
            else:
                b_h1 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 64:
                    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 128:
                    b_h3 = tl.zeros([64, BV], dtype=tl.float32)
                if K > 192:
                    b_h4 = tl.zeros([64, BV], dtype=tl.float32)

            # load initial state for this V segment (K-segmented)
            if USE_INITIAL_STATE:
                h0_ptr = h0 + i_nh * K * V
                if STATE_V_FIRST:
                    # h0 layout [V, K]: block (V_seg=BV, K_seg=64)
                    p_h0_1 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, 0), (BV, 64), (1, 0))
                    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                    if K > 64:
                        p_h0_2 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, 64), (BV, 64), (1, 0))
                        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                    if K > 128:
                        p_h0_3 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, 128), (BV, 64), (1, 0))
                        b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                    if K > 192:
                        p_h0_4 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, 192), (BV, 64), (1, 0))
                        b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
                else:
                    # h0 layout [K, V]: block (K_seg=64, V_seg=BV)
                    p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (64, BV), (1, 0))
                    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                    if K > 64:
                        p_h0_2 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (64, v_start), (64, BV), (1, 0))
                        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                    if K > 128:
                        p_h0_3 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (128, v_start), (64, BV), (1, 0))
                        b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                    if K > 192:
                        p_h0_4 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (192, v_start), (64, BV), (1, 0))
                        b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

            # main recurrence
            for i_t in range(NT):
                h_base = h + (boh + i_t) * HV * K * V + i_h * K * V

                # store h for this V segment (K-segmented)
                if STATE_V_FIRST:
                    p_h1 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, 0), (BV, 64), (1, 0))
                    tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                    if K > 64:
                        p_h2 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, 64), (BV, 64), (1, 0))
                        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
                    if K > 128:
                        p_h3 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, 128), (BV, 64), (1, 0))
                        tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
                    if K > 192:
                        p_h4 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, 192), (BV, 64), (1, 0))
                        tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
                else:
                    p_h1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (64, BV), (1, 0))
                    tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                    if K > 64:
                        p_h2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (64, v_start), (64, BV), (1, 0))
                        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
                    if K > 128:
                        p_h3 = tl.make_block_ptr(h_base, (K, V), (V, 1), (128, v_start), (64, BV), (1, 0))
                        tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
                    if K > 192:
                        p_h4 = tl.make_block_ptr(h_base, (K, V), (V, 1), (192, v_start), (64, BV), (1, 0))
                        tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

                # load w (K-segmented), accumulate b_v = sum_k dot(b_w_k, b_h_k)
                w_base = w + bos * HV * K + i_h * K
                p_w1 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
                b_w = tl.load(p_w1, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
                else:
                    b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
                if K > 64:
                    p_w2 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                    b_w = tl.load(p_w2, boundary_check=(0, 1)).to(tl.float32)
                    if STATE_V_FIRST:
                        b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
                    else:
                        b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
                if K > 128:
                    p_w3 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                    b_w = tl.load(p_w3, boundary_check=(0, 1)).to(tl.float32)
                    if STATE_V_FIRST:
                        b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
                    else:
                        b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
                if K > 192:
                    p_w4 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                    b_w = tl.load(p_w4, boundary_check=(0, 1)).to(tl.float32)
                    if STATE_V_FIRST:
                        b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
                    else:
                        b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

                v_new_base = v_new + bos * HV * V + i_h * V

                last_idx = min((i_t + 1) * BT, T) - 1
                if USE_G:
                    # g is transposed to [B, HV, T] in wrapper for contiguous T-load.
                    # Non-varlen: g_ptr = g + i_n * HV * T_max + i_h * T_max (i_n is batch index)
                    # Varlen (B=1): g_ptr = g + bos + i_h * T_max (bos is absolute token offset)
                    if IS_VARLEN:
                        g_ptr = g + bos + i_h * T_max
                    else:
                        g_ptr = g + i_n * HV * T_max + i_h * T_max
                    b_g_last = tl.load(g_ptr + last_idx).to(tl.float32)
                    p_g = tl.make_block_ptr(g_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                    m_t = (i_t * BT + tl.arange(0, BT)) < T

                # load v and compute v_new = v - b_v
                p_v = tl.make_block_ptr(v + bos * HV * V + i_h * V, (T, V), (stride_v, 1),
                                        (i_t * BT, v_start), (BT, BV), (1, 0))
                b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32) - b_v

                if SAVE_NEW_VALUE:
                    p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
                    tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

                if USE_G:
                    b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                    b_h1 *= b_g_last
                    if K > 64:
                        b_h2 *= b_g_last
                    if K > 128:
                        b_h3 *= b_g_last
                    if K > 192:
                        b_h4 *= b_g_last

                if USE_GK:
                    # gk layout [B, T, HV, K]: load last token's per-key gate (K-segmented)
                    gk_base = gk + (bos + last_idx) * HV * K + i_h * K
                    o_k1 = tl.arange(0, 64)
                    b_gk_last1 = tl.load(gk_base + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h1 *= exp2(b_gk_last1)[None, :]
                    else:
                        b_h1 *= exp2(b_gk_last1)[:, None]
                    if K > 64:
                        o_k2 = 64 + o_k1
                        b_gk_last2 = tl.load(gk_base + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                        if STATE_V_FIRST:
                            b_h2 *= exp2(b_gk_last2)[None, :]
                        else:
                            b_h2 *= exp2(b_gk_last2)[:, None]
                    if K > 128:
                        o_k3 = 128 + o_k1
                        b_gk_last3 = tl.load(gk_base + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                        if STATE_V_FIRST:
                            b_h3 *= exp2(b_gk_last3)[None, :]
                        else:
                            b_h3 *= exp2(b_gk_last3)[:, None]
                    if K > 192:
                        o_k4 = 192 + o_k1
                        b_gk_last4 = tl.load(gk_base + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                        if STATE_V_FIRST:
                            b_h4 *= exp2(b_gk_last4)[None, :]
                        else:
                            b_h4 *= exp2(b_gk_last4)[:, None]

                # load k (K-segmented), update b_h += dot(b_k_seg, b_v)
                k_base = k + bos * H * K + (i_h // (HV // H)) * K
                p_k1 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_h1 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    p_k2 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h2 += tl.trans(tl.dot(b_k, b_v))
                    else:
                        b_h2 += tl.dot(b_k, b_v)
                if K > 128:
                    p_k3 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h3 += tl.trans(tl.dot(b_k, b_v))
                    else:
                        b_h3 += tl.dot(b_k, b_v)
                if K > 192:
                    p_k4 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k4, boundary_check=(0, 1)).to(tl.float32)
                    if STATE_V_FIRST:
                        b_h4 += tl.trans(tl.dot(b_k, b_v))
                    else:
                        b_h4 += tl.dot(b_k, b_v)

            # epilogue: store final state for this V segment (K-segmented)
            if STORE_FINAL_STATE:
                ht_ptr = ht + i_nh * K * V

                if STATE_V_FIRST:
                    p_ht1 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, 0), (BV, 64), (1, 0))
                    tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                    if K > 64:
                        p_ht2 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, 64), (BV, 64), (1, 0))
                        tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
                    if K > 128:
                        p_ht3 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, 128), (BV, 64), (1, 0))
                        tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
                    if K > 192:
                        p_ht4 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, 192), (BV, 64), (1, 0))
                        tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))
                else:
                    p_ht1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (64, BV), (1, 0))
                    tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                    if K > 64:
                        p_ht2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (64, v_start), (64, BV), (1, 0))
                        tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
                    if K > 128:
                        p_ht3 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (128, v_start), (64, BV), (1, 0))
                        tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
                    if K > 192:
                        p_ht4 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (192, v_start), (64, BV), (1, 0))
                        tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices) if chunk_indices is not None else chunk_offsets[-1].item()
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    if g is not None:
        g = g.transpose(1, 2).contiguous()

    num_core = get_npu_properties()["num_aicore"]
    task_num = N * HV
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu[(num_core,)](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        task_num=task_num,
        num_core=num_core,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state


@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu(
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
    cu_seqlens,
    chunk_offsets,
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
    USE_G_PRECOMP: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    V_OFFSET: tl.constexpr,
    NH_OFFSET: tl.constexpr,
):
    i_v = tl.program_id(0) + V_OFFSET
    i_nh = tl.program_id(1) + NH_OFFSET
    i_n, i_h = i_nh // HV, i_nh % HV
    # Keep tensor-width T for g strides; T_cur is the active sequence length.
    T_cur = T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_cur = eos - bos
        NT = tl.cdiv(T_cur, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T_cur, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_dh1 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > BK:
            b_dh2 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > 2 * BK:
            b_dh3 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > 3 * BK:
            b_dh4 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > BK:
            b_dh2 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > 2 * BK:
            b_dh3 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > 3 * BK:
            b_dh4 = tl.zeros([BK, BV], dtype=tl.float32)

    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K * V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K
    # Host g layout is [B, HV, T] (or [1, HV, T] for varlen).
    if USE_G:
        if USE_G_PRECOMP:
            g_exp += (i_n * HV + i_h).to(tl.int64) * T
            g_ratio += (i_n * HV + i_h).to(tl.int64) * T
            g_last_exp += (i_n * HV + i_h).to(tl.int64) * NT
        elif IS_VARLEN:
            g += (bos + i_h * T).to(tl.int64)
        else:
            g += (i_n * HV + i_h).to(tl.int64) * T

    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht1 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > BK:
            if STATE_V_FIRST:
                p_dht2 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, BK), (BV, BK), (1, 0))
            else:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (BK, i_v * BV), (BK, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 2 * BK:
            if STATE_V_FIRST:
                p_dht3 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 2 * BK), (BV, BK), (1, 0))
            else:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (2 * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 3 * BK:
            if STATE_V_FIRST:
                p_dht4 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 3 * BK), (BV, BK), (1, 0))
            else:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (3 * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    if USE_GK:
        o_k1 = tl.arange(0, BK)

    DH_CS: tl.constexpr = HV * K * V
    dh_chunk = dh + (NT - 1).to(tl.int64) * DH_CS

    for i_t in range(NT - 1, -1, -1):
        if STATE_V_FIRST:
            p_dh1 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_dh1 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > BK:
            if STATE_V_FIRST:
                p_dh2 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (i_v * BV, BK), (BV, BK), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 2 * BK:
            if STATE_V_FIRST:
                p_dh3 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (i_v * BV, 2 * BK), (BV, BK), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (2 * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 3 * BK:
            if STATE_V_FIRST:
                p_dh4 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (i_v * BV, 3 * BK), (BV, BK), (1, 0))
            else:
                p_dh4 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (3 * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            if USE_G_PRECOMP:
                bg_last_exp = tl.load(g_last_exp + i_t).to(tl.float32)
                p_g_exp = tl.make_block_ptr(g_exp, (T_cur,), (1,), (i_t * BT,), (BT,), (0,))
                p_g_ratio = tl.make_block_ptr(g_ratio, (T_cur,), (1,), (i_t * BT,), (BT,), (0,))
                b_g_exp = tl.load(p_g_exp, boundary_check=(0,)).to(tl.float32)
                b_g_ratio = tl.load(p_g_ratio, boundary_check=(0,)).to(tl.float32)
            else:
                last_idx = min((i_t + 1) * BT, T_cur) - 1
                bg_last = tl.load(g + last_idx).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T_cur,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
                b_g_ratio = exp2(bg_last - b_g)
        # In-place dv correction (callers rebind the returned buffer).
        p_dv = tl.make_block_ptr(dv, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T_cur, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k, (T_cur, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            last_idx = min((i_t + 1) * BT, T_cur) - 1
            b_gk_last1 = tl.load(gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        if STATE_V_FIRST:
            b_dv = tl.dot(b_dh1.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
            b_dv = tl.trans(b_dv)
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), allow_tf32=False)

        if K > BK:
            p_k = tl.make_block_ptr(k, (T_cur, K), (H * K, 1), (i_t * BT, BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = BK + o_k1
                b_gk_last2 = tl.load(gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv_part = tl.dot(b_dh2.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                b_dv += tl.trans(b_dv_part)
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype), allow_tf32=False)

        if K > 2 * BK:
            p_k = tl.make_block_ptr(k, (T_cur, K), (H * K, 1), (i_t * BT, 2 * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 2 * BK + o_k1
                b_gk_last3 = tl.load(gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv_part = tl.dot(b_dh3.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                b_dv += tl.trans(b_dv_part)
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype), allow_tf32=False)

        if K > 3 * BK:
            p_k = tl.make_block_ptr(k, (T_cur, K), (H * K, 1), (i_t * BT, 3 * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 3 * BK + o_k1
                b_gk_last4 = tl.load(gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv_part = tl.dot(b_dh4.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                b_dv += tl.trans(b_dv_part)
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype), allow_tf32=False)

        if USE_G:
            if USE_G_PRECOMP:
                b_dv *= b_g_ratio[:, None]
            else:
                m_t = (i_t * BT + tl.arange(0, BT)) < T_cur
                b_dv *= tl.where(m_t, b_g_ratio, 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # Decay dh once; fold gate*scale into do so K-slabs skip per-slab gating.
        if USE_G:
            b_dh1 *= bg_last_exp
            if K > BK:
                b_dh2 *= bg_last_exp
            if K > 2 * BK:
                b_dh3 *= bg_last_exp
            if K > 3 * BK:
                b_dh4 *= bg_last_exp
            b_do = b_do * (b_g_exp * scale)[:, None]
        else:
            b_do = b_do * scale

        p_w = tl.make_block_ptr(w, (K, T_cur), (1, HV * K), (0, i_t * BT), (BK, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T_cur), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_GK:
            if STATE_V_FIRST:
                b_dh1 *= exp2(b_gk_last1)[None, :]
            else:
                b_dh1 *= exp2(b_gk_last1[:, None])
        if STATE_V_FIRST:
            b_dh1 += tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
            b_dh1 -= tl.dot(tl.trans(b_dv.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
        else:
            b_dh1 += (
                tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False)
                - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
            )

        if K > BK:
            p_q = tl.make_block_ptr(q, (K, T_cur), (1, H * K), (BK, i_t * BT), (BK, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T_cur), (1, HV * K), (BK, i_t * BT), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_dh2 *= exp2(b_gk_last2[:, None])
            if STATE_V_FIRST:
                b_dh2 += tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                b_dh2 -= tl.dot(tl.trans(b_dv.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
            else:
                b_dh2 += (
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False)
                    - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
                )

        if K > 2 * BK:
            p_q = tl.make_block_ptr(q, (K, T_cur), (1, H * K), (2 * BK, i_t * BT), (BK, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T_cur), (1, HV * K), (2 * BK, i_t * BT), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_dh3 *= exp2(b_gk_last3[:, None])
            if STATE_V_FIRST:
                b_dh3 += tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                b_dh3 -= tl.dot(tl.trans(b_dv.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
            else:
                b_dh3 += (
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False)
                    - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
                )

        if K > 3 * BK:
            p_q = tl.make_block_ptr(q, (K, T_cur), (1, H * K), (3 * BK, i_t * BT), (BK, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T_cur), (1, HV * K), (3 * BK, i_t * BT), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_dh4 *= exp2(b_gk_last4[:, None])
            if STATE_V_FIRST:
                b_dh4 += tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                b_dh4 -= tl.dot(tl.trans(b_dv.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
            else:
                b_dh4 += (
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False)
                    - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
                )

        dh_chunk -= DH_CS

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > BK:
            if STATE_V_FIRST:
                p_dh1 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, BK), (BV, BK), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 2 * BK:
            if STATE_V_FIRST:
                p_dh2 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 2 * BK), (BV, BK), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (2 * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 3 * BK:
            if STATE_V_FIRST:
                p_dh3 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 3 * BK), (BV, BK), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (3 * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def _prepare_dhu_gate_tensors(
    g: torch.Tensor | None,
    *,
    B: int,
    T: int,
    HV: int,
    BT: int,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool]:
    """Transpose g to [B, HV, T]; if T%BT==0, also precompute fp32 exp2 scales."""
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, 'current kernel does not support head dimension being larger than 256.'

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    g_log, g_exp, g_ratio, g_last_exp, use_g_precomp = _prepare_dhu_gate_tensors(
        g, B=B, T=T, HV=HV, BT=BT, cu_seqlens=cu_seqlens,
    )
    gate_inline = g is not None and not use_g_precomp
    BK, BV = _select_bwd_dhu_tiles(K, V, state_v_first, gate_inline=gate_inline)
    nv_chunks = triton.cdiv(V, BV)
    _launch_fwd_h_kernel(
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu,
        nv_chunks=nv_chunks,
        nh_total=N * HV,
        kernel_kwargs={
            'q': q,
            'k': k,
            'w': w,
            'g': g_log,
            'g_exp': g_exp,
            'g_ratio': g_ratio,
            'g_last_exp': g_last_exp,
            'gk': gk,
            'dht': dht,
            'dh0': dh0,
            'do': do,
            'dh': dh,
            'dv': dv,
            'cu_seqlens': cu_seqlens,
            'chunk_offsets': chunk_offsets,
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
            'USE_G_PRECOMP': use_g_precomp,
            'USE_GK': gk is not None,
            'USE_INITIAL_STATE': h0 is not None,
            'USE_FINAL_STATE_GRADIENT': dht is not None,
            'STATE_V_FIRST': state_v_first,
            'IS_VARLEN': cu_seqlens is not None,
            'V_OFFSET': 0,
            'NH_OFFSET': 0,
        },
    )
    return dh, dh0, dv
