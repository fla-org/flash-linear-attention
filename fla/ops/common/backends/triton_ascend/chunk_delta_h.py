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
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    compute_row_tile_block_size,
    get_ub_manager,
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
# Soft over-admit: D256 BK=256/BV=128 live-range peak is 1.18× 192KiB (cost 14 vs 22 for BK=128/BV=128).
_FWD_UB_SOFT = 1.20
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


def _fwd_h_peak_bytes(BK: int, BV: int, n_slabs: int, BT: int = _FWD_H_BT) -> int:
    """b_h all slabs stay live (fp32); w and k are sequential one-slab tiles.

    Recurrence is store-h → load-w/dot → load-v → load-k/dot. Coupled with
    ``enable_ubuf_saving`` on the fwd launch so D256 BK=256/BV=128 compiles
    (256KiB naive live set packed into 192KiB UB).
    """
    return n_slabs * BK * BV * 4 + BK * BT * 4 + BT * BV * 4 + BK * 4 + 12 * BT


def _fwd_h_tile_cost(K: int, V: int, BK: int, BV: int) -> int:
    """Scalar proxy: nv × (base ptrs + ~4 per K-slab), same family as bwd dhu."""
    return triton.cdiv(V, BV) * (3 + 4 * triton.cdiv(K, BK))


def _select_fwd_h_tiles(K: int, V: int, state_v_first: bool) -> tuple[int, int]:
    """Pick (BK, BV) minimizing ptr/scalar cost under a UB cap.

    ``STATE_V_FIRST`` keeps BK=64 because ``tl.trans(b_h)`` copies inflate live UB.
    At most four K-slabs (kernel unroll); BK starts at 64.
    """
    if state_v_first:
        return 64, _get_bv(K, V)

    soft_cap = int(get_ub_manager().ub_capacity_bytes * _FWD_UB_SOFT)
    max_bk = min(256, triton.next_power_of_2(max(K, 64)))
    desired_v = triton.next_power_of_2(V)
    best: tuple[int, int, int] | None = None  # cost, BK, BV

    bk = 64
    while bk <= max_bk:
        n_slabs = triton.cdiv(K, bk)
        if n_slabs <= 4:
            bv = 16
            while bv <= min(desired_v, 256):
                if _fwd_h_peak_bytes(bk, bv, n_slabs) <= soft_cap:
                    cost = _fwd_h_tile_cost(K, V, bk, bv)
                    if (
                        best is None
                        or cost < best[0]
                        or (cost == best[0] and (bk > best[1] or (bk == best[1] and bv > best[2])))
                    ):
                        best = (cost, bk, bv)
                bv *= 2
        bk *= 2

    if best is None:
        return 64, _get_bv(K, V)
    return best[1], best[2]


def _dhu_peak_bytes(BK: int, BV: int, n_slabs: int, BT: int = _DHU_BT) -> int:
    # All K-slabs of dh stay live; q/w are one-slab at a time.
    return n_slabs * BK * BV * 4 + 2 * BK * BT * 2 + BT * BV * 2 + BT * BV * 4 + 12 * BT


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

    soft_cap = int(get_ub_manager().ub_capacity_bytes * (_DHU_UB_GATE_INLINE if gate_inline else _DHU_UB_SOFT))
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


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def _launch_core_grid(kernel, *, task_num: int, kernel_kwargs: dict, **compile_opts) -> None:
    num_core = get_npu_properties()["num_aicore"]
    kernel[(num_core,)](task_num=task_num, num_core=num_core, **compile_opts, **kernel_kwargs)


_FWD_H_COMPILE = dict(enable_ubuf_saving=True, unit_flag=True)

_FWD_H_HEURISTICS = {
    "USE_G": lambda args: args["g"] is not None or args["g_ratio"] is not None,
    "USE_G_PRECOMP": lambda args: args["g_ratio"] is not None,
    "USE_GK": lambda args: args["gk"] is not None or args["gk_last_exp"] is not None,
    "USE_GK_PRECOMP": lambda args: args["gk_last_exp"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
}


@triton.heuristics(_FWD_H_HEURISTICS)
@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu(
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
):
    core_id = tl.program_id(0)
    NV: tl.constexpr = tl.cdiv(V, BV)
    DH_CS: tl.constexpr = HV * K * V
    stride_v: tl.constexpr = HV * V
    stride_k: tl.constexpr = H * K
    stride_w: tl.constexpr = HV * K
    T_max = T
    for task_id in tl.range(core_id, task_num, num_core):
        # One V-tile per task, matching CUDA grid NV * N * HV.
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
            bos = i_n * T
            NT = tl.cdiv(T, BT)
            boh = i_n * NT

        v_start = i_v * BV
        # Rebind GM bases each task; do not in-place ``ptr +=`` across ``i_t``
        # (Ascend MTE OOB). ``h`` rebases each chunk via ``i_t * DH_CS``.
        w_base = w + (bos * HV + i_h).to(tl.int64) * K
        k_base = k + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        v_base = v + (bos * HV + i_h).to(tl.int64) * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_h).to(tl.int64) * V
        h_nh = h + (boh * HV + i_h).to(tl.int64) * K * V

        if USE_G:
            if USE_G_PRECOMP:
                g_ratio_ptr = g_ratio + (i_n * HV + i_h).to(tl.int64) * T_max
                g_last_exp_ptr = g_last_exp + (i_n * HV + i_h).to(tl.int64) * NT
        if USE_GK:
            if USE_GK_PRECOMP:
                gk_last_exp_nh = gk_last_exp + (i_n * HV + i_h).to(tl.int64) * NT * K

        # b_h shape: [BK, BV] (default) or [BV, BK] (STATE_V_FIRST)
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

        # load initial state for this V segment (K-segmented)
        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            if STATE_V_FIRST:
                # h0 layout [V, K]: block (V_seg=BV, K_seg=BK)
                p_h0_1 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, 0), (BV, BK), (1, 0))
                b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                if K > BK:
                    p_h0_2 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, BK), (BV, BK), (1, 0))
                    b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                if K > BK * 2:
                    p_h0_3 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, BK * 2), (BV, BK), (1, 0))
                    b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                if K > BK * 3:
                    p_h0_4 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (v_start, BK * 3), (BV, BK), (1, 0))
                    b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
            else:
                # h0 layout [K, V]: block (K_seg=BK, V_seg=BV)
                p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
                b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                if K > BK:
                    p_h0_2 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (BK, v_start), (BK, BV), (1, 0))
                    b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                if K > BK * 2:
                    p_h0_3 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (BK * 2, v_start), (BK, BV), (1, 0))
                    b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                if K > BK * 3:
                    p_h0_4 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (BK * 3, v_start), (BK, BV), (1, 0))
                    b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

        # main recurrence
        for i_t in range(NT):
            h_base = h_nh + tl.cast(i_t, tl.int64) * DH_CS

            # store h for this V segment (K-segmented)
            if STATE_V_FIRST:
                p_h1 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, 0), (BV, BK), (1, 0))
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                if K > BK:
                    p_h2 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, BK), (BV, BK), (1, 0))
                    tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 2:
                    p_h3 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, BK * 2), (BV, BK), (1, 0))
                    tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 3:
                    p_h4 = tl.make_block_ptr(h_base, (V, K), (K, 1), (v_start, BK * 3), (BV, BK), (1, 0))
                    tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
            else:
                p_h1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                if K > BK:
                    p_h2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (BK, v_start), (BK, BV), (1, 0))
                    tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 2:
                    p_h3 = tl.make_block_ptr(h_base, (K, V), (V, 1), (BK * 2, v_start), (BK, BV), (1, 0))
                    tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 3:
                    p_h4 = tl.make_block_ptr(h_base, (K, V), (V, 1), (BK * 3, v_start), (BK, BV), (1, 0))
                    tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

            # load w (K-segmented), accumulate b_v = sum_k dot(b_w_k, b_h_k)
            p_w1 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_w = tl.load(p_w1, boundary_check=(0, 1)).to(tl.float32)
            if STATE_V_FIRST:
                b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
            else:
                b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > BK:
                p_w2 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, BK), (BT, BK), (1, 0))
                b_w = tl.load(p_w2, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
                else:
                    b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > BK * 2:
                p_w3 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, BK * 2), (BT, BK), (1, 0))
                b_w = tl.load(p_w3, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
                else:
                    b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > BK * 3:
                p_w4 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, BK * 3), (BT, BK), (1, 0))
                b_w = tl.load(p_w4, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
                else:
                    b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

            if USE_G:
                if USE_G_PRECOMP:
                    p_g_ratio = tl.make_block_ptr(g_ratio_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_g_ratio = tl.load(p_g_ratio, boundary_check=(0,)).to(tl.float32)
                    b_g_last = tl.load(g_last_exp_ptr + i_t).to(tl.float32)
                else:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    # g is transposed to [B, HV, T] in wrapper for contiguous T-load.
                    if IS_VARLEN:
                        g_ptr = g + bos + i_h * T_max
                    else:
                        g_ptr = g + i_n * HV * T_max + i_h * T_max
                    b_g_last = tl.load(g_ptr + last_idx).to(tl.float32)
                    p_g = tl.make_block_ptr(g_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
                    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                    m_t = (i_t * BT + tl.arange(0, BT)) < T

            # load v and compute v_new = v - b_v
            p_v = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32) - b_v

            if SAVE_NEW_VALUE:
                p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

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
                    b_gk_last1 = tl.load(gk_base + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
                else:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    gk_base = gk + (bos + last_idx) * HV * K + i_h * K
                    b_gk_last1 = exp2(tl.load(gk_base + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32))
                if STATE_V_FIRST:
                    b_h1 *= b_gk_last1[None, :]
                else:
                    b_h1 *= b_gk_last1[:, None]
                if K > BK:
                    o_k2 = BK + o_k1
                    if USE_GK_PRECOMP:
                        b_gk_last2 = tl.load(gk_base + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
                    else:
                        b_gk_last2 = exp2(tl.load(gk_base + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h2 *= b_gk_last2[None, :]
                    else:
                        b_h2 *= b_gk_last2[:, None]
                if K > BK * 2:
                    o_k3 = BK * 2 + o_k1
                    if USE_GK_PRECOMP:
                        b_gk_last3 = tl.load(gk_base + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32)
                    else:
                        b_gk_last3 = exp2(tl.load(gk_base + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h3 *= b_gk_last3[None, :]
                    else:
                        b_h3 *= b_gk_last3[:, None]
                if K > BK * 3:
                    o_k4 = BK * 3 + o_k1
                    if USE_GK_PRECOMP:
                        b_gk_last4 = tl.load(gk_base + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32)
                    else:
                        b_gk_last4 = exp2(tl.load(gk_base + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32))
                    if STATE_V_FIRST:
                        b_h4 *= b_gk_last4[None, :]
                    else:
                        b_h4 *= b_gk_last4[:, None]

            # load k (K-segmented), update b_h += dot(b_k_seg, b_v)
            p_k1 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (BK, BT), (0, 1))
            b_k = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h1 += tl.dot(b_k, b_v)
            if K > BK:
                p_k2 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (BK, i_t * BT), (BK, BT), (0, 1))
                b_k = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h2 += tl.dot(b_k, b_v)
            if K > BK * 2:
                p_k3 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (BK * 2, i_t * BT), (BK, BT), (0, 1))
                b_k = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h3 += tl.dot(b_k, b_v)
            if K > BK * 3:
                p_k4 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (BK * 3, i_t * BT), (BK, BT), (0, 1))
                b_k = tl.load(p_k4, boundary_check=(0, 1)).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 += tl.trans(tl.dot(b_k, b_v))
                else:
                    b_h4 += tl.dot(b_k, b_v)

        # epilogue: store final state for this V segment (K-segmented)
        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V

            if STATE_V_FIRST:
                p_ht1 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, 0), (BV, BK), (1, 0))
                tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                if K > BK:
                    p_ht2 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, BK), (BV, BK), (1, 0))
                    tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 2:
                    p_ht3 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, BK * 2), (BV, BK), (1, 0))
                    tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 3:
                    p_ht4 = tl.make_block_ptr(ht_ptr, (V, K), (K, 1), (v_start, BK * 3), (BV, BK), (1, 0))
                    tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))
            else:
                p_ht1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
                tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                if K > BK:
                    p_ht2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (BK, v_start), (BK, BV), (1, 0))
                    tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 2:
                    p_ht3 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (BK * 2, v_start), (BK, BV), (1, 0))
                    tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
                if K > BK * 3:
                    p_ht4 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (BK * 3, v_start), (BK, BV), (1, 0))
                    tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_ratio"] is not None,
        "USE_GK": lambda args: args["gk_last_exp"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    }
)
@triton.jit(do_not_specialize=["task_num", "num_core"])
def chunk_gated_delta_rule_fwd_kernel_h_oneslab_npu(
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
    NT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
):
    """Aligned oneslab path: host NT constexpr, no K-slab/varlen/STATE_V_FIRST.

    Oneslab is only launched when ``T % BT == 0`` and not varlen, so gates are
    always host-precomputed (``g``/``gk`` are never passed).
    """
    core_id = tl.program_id(0)
    DH_CS: tl.constexpr = HV * K * V
    stride_v: tl.constexpr = HV * V
    stride_k: tl.constexpr = H * K
    stride_w: tl.constexpr = HV * K
    NV: tl.constexpr = tl.cdiv(V, BV)
    if USE_GK:
        o_k = tl.arange(0, BK)

    for task_id in tl.range(core_id, task_num, num_core):
        i_v, i_nh = task_id % NV, (task_id // NV).to(tl.int64)
        i_n, i_h = i_nh // HV, i_nh % HV
        bos = i_n * T
        v_start = i_v * BV

        w_base = w + (bos * HV + i_h).to(tl.int64) * K
        k_base = k + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        v_base = v + (bos * HV + i_h).to(tl.int64) * V
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * HV + i_h).to(tl.int64) * V
        h_nh = h + (i_n * NT * HV + i_h).to(tl.int64) * K * V

        if USE_G:
            g_ratio_ptr = g_ratio + (i_n * HV + i_h).to(tl.int64) * T
            g_last_exp_ptr = g_last_exp + (i_n * HV + i_h).to(tl.int64) * NT
        if USE_GK:
            gk_last_exp_nh = gk_last_exp + (i_n * HV + i_h).to(tl.int64) * NT * K

        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
            p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
            b_h1 += tl.load(p_h0_1).to(tl.float32)

        for i_t in range(NT):
            h_base = h_nh + tl.cast(i_t, tl.int64) * DH_CS
            p_h1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty))

            p_w1 = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_w = tl.load(p_w1).to(tl.float32)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))

            if USE_G:
                p_g_ratio = tl.make_block_ptr(g_ratio_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g_ratio = tl.load(p_g_ratio).to(tl.float32)
                b_g_last = tl.load(g_last_exp_ptr + i_t).to(tl.float32)

            p_v = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
            b_v = tl.load(p_v).to(tl.float32) - b_v
            if SAVE_NEW_VALUE:
                p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty))

            if USE_G:
                b_v = b_v * b_g_ratio[:, None]
                b_h1 *= b_g_last

            if USE_GK:
                gk_base = gk_last_exp_nh + tl.cast(i_t, tl.int64) * K
                b_gk_last1 = tl.load(gk_base + o_k).to(tl.float32)
                b_h1 *= b_gk_last1[:, None]

            p_k1 = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (BK, BT), (0, 1))
            b_k = tl.load(p_k1).to(tl.float32)
            b_h1 += tl.dot(b_k, b_v)

        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V
            p_ht1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (BK, BV), (1, 0))
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
    """Transpose g to [B, HV, T]; if T%BT==0, precompute fp32 exp2 scales.

    Returns ``(g_log, g_ratio, g_last_exp)``. Aligned non-varlen paths pass
    ``g_log=None`` so the kernel loads only the precomputed tensors.
    """
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
    """If T%BT==0, precompute ``exp2(gk)`` at each chunk's last token.

    Returns ``(gk, gk_last_exp)`` with layout ``[B, HV, NT, K]`` fp32.
    """
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

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    g, g_ratio, g_last_exp = _prepare_fwd_g_gates(
        g, B=B, T=T, HV=HV, BT=BT, cu_seqlens=cu_seqlens,
    )
    gk, gk_last_exp = _prepare_fwd_gk_last_exp(
        gk, B=B, T=T, HV=HV, K=K, BT=BT, cu_seqlens=cu_seqlens,
    )

    BK, BV = _select_fwd_h_tiles(K, V, state_v_first)
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
            chunk_gated_delta_rule_fwd_kernel_h_oneslab_npu,
            task_num=N * HV * triton.cdiv(V, BV),
            kernel_kwargs={**kwargs, "NT": T // BT},
            **_FWD_H_COMPILE,
        )
    else:
        _launch_core_grid(
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu,
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


@triton.jit(do_not_specialize=["T", "task_num", "num_core"])
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
            bos, eos = i_n * T_max, i_n * T_max + T_max
            NT = tl.cdiv(T, BT)
            boh = i_n * NT

        stride_v = HV * V
        stride_k = H * K
        stride_w = HV * K

        q_base = q + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        k_base = k + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        w_base = w + (bos * HV + i_h).to(tl.int64) * K
        do_base = do + (bos * HV + i_h).to(tl.int64) * V
        dv_base = dv + (bos * HV + i_h).to(tl.int64) * V
        dv2_base = dv2 + (bos * HV + i_h).to(tl.int64) * V
        dh_base = dh + (boh * HV + i_h).to(tl.int64) * K * V
        if USE_GK:
            gk_base = gk + (bos * HV + i_h).to(tl.int64) * K
        if USE_G:
            if USE_G_PRECOMP:
                g_exp_base = g_exp + (i_n * HV + i_h).to(tl.int64) * T_max
                g_ratio_base = g_ratio + (i_n * HV + i_h).to(tl.int64) * T_max
                g_last_exp_base = g_last_exp + (i_n * HV + i_h).to(tl.int64) * NT
                g_base = g
            elif IS_VARLEN:
                g_base = g + (bos + i_h * T_max).to(tl.int64)
                g_exp_base = g_exp
                g_ratio_base = g_ratio
                g_last_exp_base = g_last_exp
            else:
                g_base = g + (i_n * HV + i_h).to(tl.int64) * T_max
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
                    p_dht1 = tl.make_block_ptr(dht_base, (V, K), (K, 1), (v_start, 0), (BV, 64), (1, 0))
                else:
                    p_dht1 = tl.make_block_ptr(dht_base, (K, V), (V, 1), (0, v_start), (64, BV), (1, 0))
                b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
                if K > 64:
                    if STATE_V_FIRST:
                        p_dht2 = tl.make_block_ptr(dht_base, (V, K), (K, 1), (v_start, 64), (BV, 64), (1, 0))
                    else:
                        p_dht2 = tl.make_block_ptr(dht_base, (K, V), (V, 1), (64, v_start), (64, BV), (1, 0))
                    b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
                if K > 128:
                    if STATE_V_FIRST:
                        p_dht3 = tl.make_block_ptr(dht_base, (V, K), (K, 1), (v_start, 128), (BV, 64), (1, 0))
                    else:
                        p_dht3 = tl.make_block_ptr(dht_base, (K, V), (V, 1), (128, v_start), (64, BV), (1, 0))
                    b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
                if K > 192:
                    if STATE_V_FIRST:
                        p_dht4 = tl.make_block_ptr(dht_base, (V, K), (K, 1), (v_start, 192), (BV, 64), (1, 0))
                    else:
                        p_dht4 = tl.make_block_ptr(dht_base, (K, V), (V, 1), (192, v_start), (64, BV), (1, 0))
                    b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

            if USE_GK:
                o_k1 = tl.arange(0, 64)

            DH_CS: tl.constexpr = HV * K * V
            dh_chunk = dh_base + (NT - 1).to(tl.int64) * DH_CS

            for i_t in range(NT - 1, -1, -1):
                if STATE_V_FIRST:
                    p_dh1 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (v_start, 0), (BV, 64), (1, 0))
                else:
                    p_dh1 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (0, v_start), (64, BV), (1, 0))
                tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    if STATE_V_FIRST:
                        p_dh2 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (v_start, 64), (BV, 64), (1, 0))
                    else:
                        p_dh2 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (64, v_start), (64, BV), (1, 0))
                    tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
                if K > 128:
                    if STATE_V_FIRST:
                        p_dh3 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (v_start, 128), (BV, 64), (1, 0))
                    else:
                        p_dh3 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (128, v_start), (64, BV), (1, 0))
                    tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
                if K > 192:
                    if STATE_V_FIRST:
                        p_dh4 = tl.make_block_ptr(dh_chunk, (V, K), (K, 1), (v_start, 192), (BV, 64), (1, 0))
                    else:
                        p_dh4 = tl.make_block_ptr(dh_chunk, (K, V), (V, 1), (192, v_start), (64, BV), (1, 0))
                    tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

                if USE_G:
                    if USE_G_PRECOMP:
                        bg_last_exp = tl.load(g_last_exp_base + i_t).to(tl.float32)
                        p_g_exp = tl.make_block_ptr(g_exp_base, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        p_g_ratio = tl.make_block_ptr(g_ratio_base, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        b_g_exp = tl.load(p_g_exp, boundary_check=(0,)).to(tl.float32)
                        b_g_ratio = tl.load(p_g_ratio, boundary_check=(0,)).to(tl.float32)
                    else:
                        last_idx = min((i_t + 1) * BT, T) - 1
                        bg_last = tl.load(g_base + last_idx).to(tl.float32)
                        p_g = tl.make_block_ptr(g_base, (T,), (1,), (i_t * BT,), (BT,), (0,))
                        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                        bg_last_exp = exp2(bg_last)
                        b_g_exp = exp2(b_g)
                        b_g_ratio = exp2(bg_last - b_g)
                p_dv = tl.make_block_ptr(dv_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
                p_do = tl.make_block_ptr(do_base, (T, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
                b_do = tl.load(p_do, boundary_check=(0, 1))

                p_k = tl.make_block_ptr(k_base, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    last_idx = min((i_t + 1) * BT, T) - 1
                    b_gk_last1 = tl.load(gk_base + last_idx * stride_w + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
                if STATE_V_FIRST:
                    b_dv = tl.dot(b_dh1.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                    b_dv = tl.trans(b_dv)
                else:
                    b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), allow_tf32=False)

                if K > 64:
                    p_k = tl.make_block_ptr(k_base, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    if USE_GK:
                        o_k2 = 64 + o_k1
                        b_gk_last2 = tl.load(gk_base + last_idx * stride_w + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv_part = tl.dot(b_dh2.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv += tl.trans(b_dv_part)
                    else:
                        b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype), allow_tf32=False)

                if K > 128:
                    p_k = tl.make_block_ptr(k_base, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    if USE_GK:
                        o_k3 = 128 + o_k1
                        b_gk_last3 = tl.load(gk_base + last_idx * stride_w + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv_part = tl.dot(b_dh3.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv += tl.trans(b_dv_part)
                    else:
                        b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype), allow_tf32=False)

                if K > 192:
                    p_k = tl.make_block_ptr(k_base, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    if USE_GK:
                        o_k4 = 192 + o_k1
                        b_gk_last4 = tl.load(gk_base + last_idx * stride_w + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32)
                    if STATE_V_FIRST:
                        b_dv_part = tl.dot(b_dh4.to(b_k.dtype), tl.trans(b_k), allow_tf32=False)
                        b_dv += tl.trans(b_dv_part)
                    else:
                        b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype), allow_tf32=False)

                if USE_G:
                    if USE_G_PRECOMP:
                        b_dv *= b_g_ratio[:, None]
                    else:
                        m_t = (i_t * BT + tl.arange(0, BT)) < T
                        b_dv *= tl.where(m_t, b_g_ratio, 0)[:, None]
                b_dv += tl.load(p_dv, boundary_check=(0, 1))
                tl.store(p_dv2, b_dv.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
                # Ascend tl.dot clobbers lhs; b_dv is lhs in the subtract dot across K-slabs.
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
                # Ascend tl.dot clobbers lhs; b_do is lhs across K-slabs (STATE_V_FIRST path).
                b_do_c = b_do + 0.0
                b_do_c2 = b_do + 0.0
                b_do_c3 = b_do + 0.0

                p_w = tl.make_block_ptr(w_base, (K, T), (1, stride_w), (0, i_t * BT), (64, BT), (0, 1))
                p_q = tl.make_block_ptr(q_base, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                if USE_GK:
                    if STATE_V_FIRST:
                        b_dh1 *= exp2(b_gk_last1)[None, :]
                    else:
                        b_dh1 *= exp2(b_gk_last1[:, None])
                if STATE_V_FIRST:
                    b_dh1 += tl.dot(tl.trans(b_do.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                    b_dv_i = b_dv_pristine + 0.0
                    b_dh1 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                else:
                    b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=False) - tl.dot(
                        b_w, b_dv.to(b_w.dtype), allow_tf32=False
                    )

                if K > 64:
                    p_q = tl.make_block_ptr(q_base, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    p_w = tl.make_block_ptr(w_base, (K, T), (1, stride_w), (64, i_t * BT), (64, BT), (0, 1))
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh2 *= exp2(b_gk_last2)[None, :]
                        else:
                            b_dh2 *= exp2(b_gk_last2[:, None])
                    if STATE_V_FIRST:
                        b_dh2 += tl.dot(tl.trans(b_do_c.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh2 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do_c.to(b_q.dtype), allow_tf32=False) - tl.dot(
                            b_w, b_dv.to(b_w.dtype), allow_tf32=False
                        )

                if K > 128:
                    p_q = tl.make_block_ptr(q_base, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    p_w = tl.make_block_ptr(w_base, (K, T), (1, stride_w), (128, i_t * BT), (64, BT), (0, 1))
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh3 *= exp2(b_gk_last3)[None, :]
                        else:
                            b_dh3 *= exp2(b_gk_last3[:, None])
                    if STATE_V_FIRST:
                        b_dh3 += tl.dot(tl.trans(b_do_c2.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh3 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do_c2.to(b_q.dtype), allow_tf32=False) - tl.dot(
                            b_w, b_dv.to(b_w.dtype), allow_tf32=False
                        )

                if K > 192:
                    p_q = tl.make_block_ptr(q_base, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    p_w = tl.make_block_ptr(w_base, (K, T), (1, stride_w), (192, i_t * BT), (64, BT), (0, 1))
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    if USE_GK:
                        if STATE_V_FIRST:
                            b_dh4 *= exp2(b_gk_last4)[None, :]
                        else:
                            b_dh4 *= exp2(b_gk_last4[:, None])
                    if STATE_V_FIRST:
                        b_dh4 += tl.dot(tl.trans(b_do_c3.to(b_q.dtype)), tl.trans(b_q), allow_tf32=False)
                        b_dv_i = b_dv_pristine + 0.0
                        b_dh4 -= tl.dot(tl.trans(b_dv_i.to(b_w.dtype)), tl.trans(b_w), allow_tf32=False)
                    else:
                        b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do_c3.to(b_q.dtype), allow_tf32=False) - tl.dot(
                            b_w, b_dv.to(b_w.dtype), allow_tf32=False
                        )

                dh_chunk -= DH_CS

            if USE_INITIAL_STATE:
                dh0_base = dh0 + i_nh * K * V
                if STATE_V_FIRST:
                    p_dh0 = tl.make_block_ptr(dh0_base, (V, K), (K, 1), (v_start, 0), (BV, 64), (1, 0))
                else:
                    p_dh0 = tl.make_block_ptr(dh0_base, (K, V), (V, 1), (0, v_start), (64, BV), (1, 0))
                tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    if STATE_V_FIRST:
                        p_dh1 = tl.make_block_ptr(dh0_base, (V, K), (K, 1), (v_start, 64), (BV, 64), (1, 0))
                    else:
                        p_dh1 = tl.make_block_ptr(dh0_base, (K, V), (V, 1), (64, v_start), (64, BV), (1, 0))
                    tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
                if K > 128:
                    if STATE_V_FIRST:
                        p_dh2 = tl.make_block_ptr(dh0_base, (V, K), (K, 1), (v_start, 128), (BV, 64), (1, 0))
                    else:
                        p_dh2 = tl.make_block_ptr(dh0_base, (K, V), (V, 1), (128, v_start), (64, BV), (1, 0))
                    tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
                if K > 192:
                    if STATE_V_FIRST:
                        p_dh3 = tl.make_block_ptr(dh0_base, (V, K), (K, 1), (v_start, 192), (BV, 64), (1, 0))
                    else:
                        p_dh3 = tl.make_block_ptr(dh0_base, (K, V), (V, 1), (192, v_start), (64, BV), (1, 0))
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
    chunk_offsets: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    # Separate output, matching the CUDA kernel: callers must not observe a
    # mutated `dv`. Distinct from the #1113 in-register `+ 0.0` lhs copies.
    dv2 = torch.empty_like(dv)
    g_log, g_exp, g_ratio, g_last_exp, use_g_precomp = _prepare_dhu_gate_tensors(
        g,
        B=B,
        T=T,
        HV=HV,
        BT=BT,
        cu_seqlens=cu_seqlens,
    )
    gate_inline = g is not None and not use_g_precomp
    BK, BV = _select_bwd_dhu_tiles(K, V, state_v_first, gate_inline=gate_inline)
    _launch_core_grid(
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu,
        task_num=N * HV,
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
            "scale": scale,
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
        },
    )
    return dh, dh0, dv2
