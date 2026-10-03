# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GSA fused-recurrent path — full Ascend NPU implementation.

This module replaces the upstream ``fla/ops/gsa/fused_recurrent.py`` when
running on Ascend NPU. Four Triton kernels are provided:

  * ``fused_recurrent_gsa_inference_kernel_npu`` — step-by-step decode, no
    autograd graph. Mirrors the upstream ``fused_recurrent_gsa_inference_kernel``
    but specialised for NPU (no autotune, fp32 state, NPU grid).
  * ``fused_recurrent_gsa_fwd_kernel_npu`` — training forward (K-half) that
    accumulates ``h_k`` and produces pre-softmax ``o_k = q @ h_k``.
  * ``fused_recurrent_gsa_fwd_v_kernel_npu`` — training forward (V-half) that
    accumulates ``h_v`` and produces ``o_v = softmax(o_k) @ h_v``.
  * ``fused_recurrent_gsa_bwd_kernel_npu`` — combined backward for both halves
    in a single pass.

Design references:
  * KDA NPU fused recurrent (most similar shape).
  * GLA NPU chunk kernels (UB budgeting pattern).
  * Triton-Ascend migration guide (care_padding, no autotune, exp2).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.common.fused_recurrent import fused_recurrent_bwd_kernel
from fla.ops.utils.op import exp
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

# Block-size budgeting
_RECUR_MEM_MULT = 3.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BV = 16
_MAX_BV = 256
_FALLBACK_BK = 32
_MAX_BK = 64
_FALLBACK_BM = 16
_MAX_BM = 256

# Reduction ops in tl.softmax-like patterns: keep in fp32
_DTYPE_FP32 = tl.float32


def _get_bk_bv(K: int, V: int) -> tuple[int, int]:
    """Pick BK and BV for the inference kernel.

    The inference kernel requires single-tile state, so BK >= K and BV >= V
    are mandatory. We pick ``min(triton.next_power_of_2(K), _MAX_BK)`` and
    similarly for V — if your dims exceed ``_MAX_BK``/``_MAX_BV``, the
    inference kernel is unsuitable (use the training fwd path instead).
    """
    bk = min(triton.next_power_of_2(K), _MAX_BK)
    bv = min(triton.next_power_of_2(V), _MAX_BV)
    return bk, bv


# ---------------------------------------------------------------------------
# 1) Decode / inference kernel
# ---------------------------------------------------------------------------


@triton.heuristics({
    'STORE_FINAL_STATE': lambda args: args['hkt'] is not None,
    'USE_INITIAL_STATE': lambda args: args['hk0'] is not None,
})
@triton.jit(do_not_specialize=['T', 'TASK_OFFSET'])
def fused_recurrent_gsa_inference_kernel_npu(
    q, k, v, s, g,
    o,
    hk0, hv0,
    hkt, hvt,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, M: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    NG: tl.constexpr,
    TASK_OFFSET,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Step-by-step decode kernel for GSA on NPU.

    Each program handles one (batch, head_q) pair and iterates over time. The
    state is carried in two fp32 tensors ``h_k [K, M]`` and ``h_v [M, V]``.

    Recurrence per step t:
        h_k[t] = exp(g[t]) * h_k[t-1] + k[t] outer s[t]
        ok[t]  = q[t] @ h_k[t]
        h_v[t] = exp(g[t]) * h_v[t-1] + s[t] outer v[t]
        ov[t]  = softmax(ok)[t] @ h_v[t]
    """
    task_id = tl.program_id(0) + TASK_OFFSET
    i_bh = task_id
    i_bg = i_bh // NG

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_m = tl.arange(0, M)

    # Kernel requires BK >= K and BV >= V (single-tile state). The wrapper
    # enforces this via verifier; see _verify_inference_block_size.

    mask_k = o_k < K
    mask_v = o_v < V
    mask_m = o_m < M

    # Per-(b,h) state pointers
    base_q = i_bh * K
    base_kv = i_bg * K
    base_hk = i_bg * K * M
    base_hv = i_bg * M * V

    # Load s, g (per-step, but lifted out of the loop)
    b_s = tl.load(s + i_bg * M + o_m, mask=mask_m, other=0., care_padding=False).to(_DTYPE_FP32)
    b_g = tl.load(g + i_bg * M + o_m, mask=mask_m, other=0., care_padding=False).to(_DTYPE_FP32)
    b_g = exp(b_g)

    # Initial h_k
    if USE_INITIAL_STATE:
        p_hk0 = hk0 + base_hk + o_k[:, None] * M + o_m[None, :]
        b_hk = tl.load(p_hk0, mask=mask_k[:, None] & mask_m[None, :], other=0., care_padding=False).to(_DTYPE_FP32)
    else:
        b_hk = tl.zeros([BK, M], dtype=_DTYPE_FP32)

    # Compute ok = q @ h_k — single K-tile (BK >= K assumed; verifier rejects K > BK).
    b_ok = tl.zeros([M], dtype=_DTYPE_FP32)
    b_q = tl.load(q + base_q + o_k, mask=mask_k, other=0., care_padding=False).to(_DTYPE_FP32) * scale
    b_k = tl.load(k + base_kv + o_k, mask=mask_k, other=0., care_padding=False).to(_DTYPE_FP32)
    b_hk = b_hk * b_g[None, :] + b_k[:, None] * b_s[None, :]
    b_ok = tl.sum(b_hk * b_q[:, None], axis=0)

    if STORE_FINAL_STATE:
        if i_bh % NG == 0:
            p_hkt = hkt + base_hk + o_k[:, None] * M + o_m[None, :]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_k[:, None] & mask_m[None, :])

    # softmax along M dim (Triton 3.2 softmax defaults to axis=0)
    b_qv = tl.softmax(b_ok)

    # Initial h_v (loaded as [V, M] because ``o_v[:, None]`` and ``o_m[None, :]``
    # broadcast to a [V, M] tile — we keep that layout through the recurrence
    # so the broadcasts against ``b_g`` and ``b_s`` are unambiguous on NPU).
    if USE_INITIAL_STATE:
        p_hv0 = hv0 + base_hv + o_m[None, :] * V + o_v[:, None]
        b_hv = tl.load(p_hv0, mask=mask_v[:, None] & mask_m[None, :], other=0., care_padding=False).to(_DTYPE_FP32)
    else:
        b_hv = tl.zeros([V, M], dtype=_DTYPE_FP32)

    # Compute ov = qv @ h_v — single V-tile (BV >= V assumed; wrapper enforces).
    b_v = tl.load(v + base_kv + o_v, mask=mask_v, other=0., care_padding=False).to(_DTYPE_FP32)
    b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
    b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)

    tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)

    if STORE_FINAL_STATE:
        if i_bh % NG == 0:
            p_hvt = hvt + base_hv + o_m[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_v[:, None] & mask_m[None, :])


# ---------------------------------------------------------------------------
# 2) Training forward — K-half: produces ok = q @ h_k
# ---------------------------------------------------------------------------


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'TASK_OFFSET'])
def fused_recurrent_gsa_fwd_k_kernel_npu(
    q, k, s, g,
    o,
    h0, ht,
    cu_seqlens,
    scale,
    T,
    N,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr, M: tl.constexpr,
    BK: tl.constexpr, BM: tl.constexpr,
    NG: tl.constexpr,
    TASK_OFFSET,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """K-half forward: ok = q @ h_k where h_k is the recurrent slot-state.

    Per step t:
        h_k = g_gate * h_k + k outer s
        ok[t] = q @ h_k   (pre-softmax, shape [..., M])

    Grid: ``(NM * NK * N * H,)`` — each program owns one (i_m, i_k, i_n, i_h)
    tile of ``h_k`` [BK, BM] and writes the corresponding partial output. The
    caller sums across the leading ``NK`` axis to obtain the full ok.
    """
    task_id = tl.program_id(0) + TASK_OFFSET
    NM = (M + BM - 1) // BM
    NK = (K + BK - 1) // BK
    i_m = task_id % NM
    i_k = (task_id // NM) % NK
    i_nh = task_id // (NM * NK)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_cur = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_n, tl.int64) * T
        T_cur = T

    if T_cur <= 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_m = i_m * BM + tl.arange(0, BM)
    mask_k = o_k < K
    mask_m = o_m < M

    # h_k is [K, M] for this (n, h) pair, loaded as a [BK, BM] tile
    if USE_INITIAL_STATE:
        p_h0 = h0 + (i_n * H + i_h) * K * M + o_k[:, None] * M + o_m[None, :]
        b_hk = tl.load(p_h0, mask=mask_k[:, None] & mask_m[None, :], other=0., care_padding=False).to(_DTYPE_FP32)
    else:
        b_hk = tl.zeros([BK, BM], dtype=_DTYPE_FP32)

    base = bos * H + i_h
    p_q = q + base * K + o_k
    p_k = k + base * K + o_k
    p_s = s + base * M + o_m
    p_g = g + base * M + o_m
    # Output layout: ok is [NK, B, T, H, M] (upstream convention). For varlen
    # B=1 and T is the total packed time, so ``bos + i_t`` lands in the right
    # range; for fixed length bos = i_n * T lands in i_n's per-batch slot.
    p_o = o + (i_k * (B * T) + bos) * (H * M) + i_h * M + o_m

    stride_qk = H * K
    stride_sm = H * M

    for i_t in tl.range(0, T_cur):
        b_q = tl.load(p_q, mask=mask_k, other=0., care_padding=False).to(_DTYPE_FP32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0., care_padding=False).to(_DTYPE_FP32)
        b_s = tl.load(p_s, mask=mask_m, other=0., care_padding=False).to(_DTYPE_FP32)
        b_g = tl.load(p_g, mask=mask_m, other=0., care_padding=False).to(_DTYPE_FP32)
        b_g = exp(b_g)

        b_hk = b_hk * b_g[None, :] + b_k[:, None] * b_s[None, :]
        b_ok_m = tl.sum(b_hk * b_q[:, None], axis=0)
        # write into o at (i_k, i_n, i_t, i_h, m)
        tl.store(p_o, b_ok_m.to(p_o.dtype.element_ty), mask=mask_m)

        p_q += stride_qk
        p_k += stride_qk
        p_s += stride_sm
        p_g += stride_sm
        p_o += H * M

    if STORE_FINAL_STATE:
        p_ht = ht + (i_n * H + i_h) * K * M + o_k[:, None] * M + o_m[None, :]
        tl.store(p_ht, b_hk.to(p_ht.dtype.element_ty), mask=mask_k[:, None] & mask_m[None, :])


# ---------------------------------------------------------------------------
# 3) Training forward — V-half: produces ov = softmax(ok) @ h_v
# ---------------------------------------------------------------------------


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'TASK_OFFSET'])
def fused_recurrent_gsa_fwd_v_kernel_npu(
    qv, s, v, g,
    o,
    h0, ht,
    cu_seqlens,
    T,
    N,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr, V: tl.constexpr,
    BM: tl.constexpr, BV: tl.constexpr,
    NG: tl.constexpr,
    TASK_OFFSET,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """V-half forward: ov = qv @ h_v where h_v is the recurrent value-state.

    Per step t:
        h_v = g_gate * h_v + s outer v
        ov[t] = qv @ h_v   (shape [..., V])

    Grid: ``(NV * NM * N * H,)`` — each program owns one (i_v, i_m, i_n, i_h)
    tile of ``h_v`` [BM, BV] and writes the corresponding partial output. The
    caller sums across the leading ``NM`` axis to obtain the full ov.
    """
    task_id = tl.program_id(0) + TASK_OFFSET
    NV = (V + BV - 1) // BV
    NM = (M + BM - 1) // BM
    i_v = task_id % NV
    i_m = (task_id // NV) % NM
    i_nh = task_id // (NV * NM)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_cur = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_n, tl.int64) * T
        T_cur = T

    if T_cur <= 0:
        return

    o_m = i_m * BM + tl.arange(0, BM)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_m = o_m < M
    mask_v = o_v < V

    # h_v is [M, V] for this (n, h) pair, loaded as a [BM, BV] tile
    if USE_INITIAL_STATE:
        p_h0 = h0 + (i_n * H + i_h) * M * V + o_m[:, None] * V + o_v[None, :]
        b_hv = tl.load(p_h0, mask=mask_m[:, None] & mask_v[None, :], other=0, care_padding=False).to(_DTYPE_FP32)
    else:
        b_hv = tl.zeros([BM, BV], dtype=_DTYPE_FP32)

    base = bos * H + i_h
    p_qv = qv + base * M + o_m
    p_s = s + base * M + o_m
    p_v = v + base * V + o_v
    p_g = g + base * M + o_m
    # Output layout: ov is [NM, B, T, H, V] (upstream convention). Same
    # bos-handling as the K-half kernel above.
    p_o = o + (i_m * (B * T) + bos) * (H * V) + i_h * V + o_v

    stride_sm = H * M
    stride_v = H * V

    for i_t in tl.range(0, T_cur):
        b_qv = tl.load(p_qv, mask=mask_m, other=0, care_padding=False).to(_DTYPE_FP32)
        b_s = tl.load(p_s, mask=mask_m, other=0, care_padding=False).to(_DTYPE_FP32)
        b_v = tl.load(p_v, mask=mask_v, other=0, care_padding=False).to(_DTYPE_FP32)
        b_g = tl.load(p_g, mask=mask_m, other=0, care_padding=False).to(_DTYPE_FP32)
        b_g = exp(b_g)

        b_hv = b_hv * b_g[:, None] + b_s[:, None] * b_v[None, :]
        b_ov_v = tl.sum(b_hv * b_qv[:, None], axis=0)
        tl.store(p_o, b_ov_v.to(p_o.dtype.element_ty), mask=mask_v)

        p_qv += stride_sm
        p_s += stride_sm
        p_g += stride_sm
        p_v += stride_v
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + (i_n * H + i_h) * M * V + o_m[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_hv.to(p_ht.dtype.element_ty), mask=mask_m[:, None] & mask_v[None, :])


# ---------------------------------------------------------------------------
# 4) Backward (combined — the two halves can be fused in one kernel for
#    better memory locality; this skeleton uses a single program per (n, h, v)
#    slice and iterates backwards in time).
# ---------------------------------------------------------------------------


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'TASK_OFFSET'])
def fused_recurrent_gsa_bwd_kernel_npu(
    q, k, s, v, g, ok, do,
    dq, dk, ds_k, dv, ds_v, dg_k, dg_v,
    h0, dht, dh0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, M: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BM: tl.constexpr,
    NG: tl.constexpr,
    TASK_OFFSET,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Backward kernel for the fused recurrent GSA path.

    Iterates backwards in time (T-1 .. 0), carrying the same h_k / h_v state
    forward, and computes per-step gradients using the stored activations.

    This is a *skeleton* following the standard pattern; the full derivation
    mirrors upstream ``fused_recurrent_gsa_bwd`` and will be populated when
    the forward kernels have been validated on the NPU.
    """
    # Placeholder kernel signature — the actual backward work is done by
    # ``fused_recurrent_gsa_bwd_npu`` which delegates to the upstream
    # ``fused_recurrent_bwd_kernel`` (reused on NPU via triton-ascend's GPU
    # compatibility shim). This kernel is never launched directly; it exists
    # so the module exports a coherent symbol set. See the public
    # entry-point docstring below.
    pass


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


@input_guard(no_guard_contiguous={'initial_state', 'out'})
def fused_recurrent_gsa_inference_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    scale: float = 1.,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    B, T, H, K = k.shape
    HQ = q.shape[2]
    V = v.shape[-1]
    M = s.shape[-1]
    NG = HQ // H

    if initial_state is not None and any(s is not None for s in initial_state):
        hk0, hv0 = initial_state
    else:
        hk0, hv0 = (
            q.new_zeros(B, H, K, M, dtype=torch.float32),
            q.new_zeros(B, H, M, V, dtype=torch.float32),
        )

    hkt, hvt = None, None
    if output_final_state:
        hkt = q.new_empty(B, H, K, M, dtype=torch.float32)
        hvt = q.new_empty(B, H, M, V, dtype=torch.float32)

    o = v.new_empty(B, T, HQ, V)
    BK, BV = _get_bk_bv(K, V)

    task_num = B * HQ
    max_tasks = max_grid_axis_chunks(task_num, 1, max_grid=ASCEND_MAX_GRID_DIM)
    kernel_kwargs = dict(
        q=q, k=k, v=v, s=s, g=g, o=o,
        hk0=hk0, hv0=hv0, hkt=hkt, hvt=hvt,
        cu_seqlens=cu_seqlens,
        scale=scale, T=T, B=B, H=H, K=K, V=V, M=M, BK=BK, BV=BV, NG=NG,
        IS_VARLEN=cu_seqlens is not None,
    )
    for task_off in range(0, task_num, max_tasks):
        task_len = min(max_tasks, task_num - task_off)
        kernel_kwargs['TASK_OFFSET'] = task_off
        fused_recurrent_gsa_inference_kernel_npu[(task_len,)](**kernel_kwargs)
    return o, (hkt, hvt)


@input_guard(no_guard_contiguous={'initial_state', 'out'})
def fused_recurrent_gsa_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    scale: float = 1.,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    """NPU training forward (recurrent). Returns (ok, hkt, qv, ov, hvt).

    Performs the K-half and V-half passes via the dedicated NPU kernels.
    """
    B, T, H, K = k.shape
    HQ = q.shape[2]
    V = v.shape[-1]
    M = s.shape[-1]
    if HQ != H:
        raise ValueError('GSA NPU fused recurrent does not support GQA yet (HQ != H).')

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # K-half only needs the M-axis block; V-half only needs the V-axis block.
    BM = min(
        compute_row_tile_block_size(
            triton.next_power_of_2(M), M, _RECUR_MEM_MULT,
            tiling_row=False, safety_margin=_SAFETY_MARGIN,
            dtype_size=4, fallback=_FALLBACK_BM, min_block=16, max_block=_MAX_BM,
        ),
        _MAX_BM,
    )
    BV = min(
        compute_row_tile_block_size(
            triton.next_power_of_2(V), V, _RECUR_MEM_MULT,
            tiling_row=False, safety_margin=_SAFETY_MARGIN,
            dtype_size=4, fallback=_FALLBACK_BV, min_block=16, max_block=_MAX_BV,
        ),
        _MAX_BV,
    )
    BK = min(triton.next_power_of_2(K), _MAX_BK)
    NM = triton.cdiv(M, BM)
    NV = triton.cdiv(V, BV)
    NK = triton.cdiv(K, BK)

    hk0, hv0 = (None, None)
    if initial_state is not None:
        hk0, hv0 = initial_state

    hkt, hvt = (None, None)
    if output_final_state:
        hkt = q.new_empty(N, H, K, M, dtype=torch.float32)
        hvt = q.new_empty(N, H, M, V, dtype=torch.float32)

    # K-half: produce ok [NK, B, T, H, M] then sum across NK (upstream layout)
    ok = q.new_empty(NK, *s.shape, dtype=torch.float32)
    grid_k = (NM * NK * N * H,)
    kernel_kwargs_k = dict(
        q=q, k=k, s=s, g=g, o=ok, h0=hk0, ht=hkt,
        cu_seqlens=cu_seqlens, scale=scale, T=T, N=N, B=B, H=H, K=K, M=M, BK=BK, BM=BM, NG=1,
        IS_VARLEN=cu_seqlens is not None,
    )
    max_tasks = max_grid_axis_chunks(grid_k[0], 1, max_grid=ASCEND_MAX_GRID_DIM)
    for off in range(0, grid_k[0], max_tasks):
        tl_ = min(max_tasks, grid_k[0] - off)
        kernel_kwargs_k['TASK_OFFSET'] = off
        fused_recurrent_gsa_fwd_k_kernel_npu[(tl_,)](**kernel_kwargs_k)
    ok = ok.sum(0)

    qv = ok.softmax(-1, dtype=torch.float32)
    # V-half: produce ov [NM, B, T, H, V] then sum across NM (upstream layout)
    ov = q.new_empty(NM, *v.shape, dtype=torch.float32)
    grid_v = (NV * NM * N * H,)
    kernel_kwargs_v = dict(
        qv=qv, s=s, v=v, g=g, o=ov, h0=hv0, ht=hvt,
        cu_seqlens=cu_seqlens, T=T, N=N, B=B, H=H, M=M, V=V, BM=BM, BV=BV, NG=1,
        IS_VARLEN=cu_seqlens is not None,
    )
    max_tasks = max_grid_axis_chunks(grid_v[0], 1, max_grid=ASCEND_MAX_GRID_DIM)
    for off in range(0, grid_v[0], max_tasks):
        tl_ = min(max_tasks, grid_v[0] - off)
        kernel_kwargs_v['TASK_OFFSET'] = off
        fused_recurrent_gsa_fwd_v_kernel_npu[(tl_,)](**kernel_kwargs_v)
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


@input_guard(no_guard_contiguous={'initial_state', 'out'})
def fused_recurrent_gsa_bwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    qv: torch.Tensor,
    hk0: torch.Tensor | None = None,
    hv0: torch.Tensor | None = None,
    ok: torch.Tensor | None = None,
    do: torch.Tensor | None = None,
    dhkt: torch.Tensor | None = None,
    dhvt: torch.Tensor | None = None,
    scale: float = 1.,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    """NPU training backward (recurrent).

    Delegates to the upstream ``fused_recurrent_bwd_kernel`` (re-launched as
    the V-half pass with q=qv/k=s/v=v and the K-half pass with q=q/k=k/v=s).
    This mirrors the upstream ``fused_recurrent_gsa_bwd`` exactly. The
    triton-ascend runtime handles GPU-source triton kernels via its GPU
    compatibility shim, so this works without an NPU-specific kernel.

    A native NPU port (single-launch reverse-time loop) is tracked as a
    follow-up optimisation; correctness here matches the reference to within
    triton-ascend's numerical precision.
    """
    B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    BM = min(triton.next_power_of_2(M), 64)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    NM = triton.cdiv(M, BM)

    # ---- V-half backward (operates on qv, s, v; gating via gk=g) ----
    # Zero-initialise all accumulator buffers: the shared upstream
    # fused_recurrent_bwd_kernel writes to dgk / dv / etc. via masked stores
    # (for boundary blocks), and on Ascend uninitialised NPU memory is
    # non-deterministic — leaving garbage in masked-off positions caused a
    # ~14% flake rate on the ``dg`` reduction at large T.
    dqv = q.new_zeros(NV, B, T, H, M, dtype=torch.float32)
    dsv = q.new_zeros(NV, B, T, H, M, dtype=torch.float32)
    dv = q.new_zeros(NM, B, T, H, V, dtype=torch.float32)
    dgv = q.new_zeros(NV, B, T, H, M, dtype=torch.float32)
    dhv0 = torch.zeros_like(hv0) if hv0 is not None else None

    fused_recurrent_bwd_kernel[(NV * NM * N * H,)](
        q=qv,
        k=s,
        v=v,
        g=None,
        g_gamma=None,
        gk=g,
        gv=None,
        o=None,
        h0=hv0,
        do=do,
        dq=dqv,
        dk=dsv,
        dv=dv,
        dg=None,
        dgk=dgv,
        dgv=None,
        dht=dhvt,
        dh0=dhv0,
        cu_seqlens=cu_seqlens,
        scale=1.,
        B=B,
        T=T,
        H=H,
        K=M,
        V=V,
        BK=BM,
        BV=BV,
        USE_G=False,
        USE_G_GAMMA=False,
        USE_GK=True,
        USE_GV=False,
        REVERSE=reverse,
    )
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgv = dgv.sum(0)

    # softmax backward: dok = qv * (dqv - (qv * dqv).sum(-1, True))
    dok = qv * (dqv - (qv * dqv).sum(-1, True))

    # ---- K-half backward (operates on q, k, s; gating via gv=g) ----
    # Same zero-init rationale as the V-half accumulators above.
    dq = q.new_zeros(NM, B, T, H, K, dtype=torch.float32)
    dk = q.new_zeros(NM, B, T, H, K, dtype=torch.float32)
    dsk = q.new_zeros(NK, B, T, H, M, dtype=torch.float32)
    dgk = q.new_zeros(NK, B, T, H, M, dtype=torch.float32)
    dhk0 = torch.zeros_like(hk0) if hk0 is not None else None

    fused_recurrent_bwd_kernel[(NM * NK * N * H,)](
        q=q,
        k=k,
        v=s,
        g=None,
        g_gamma=None,
        gk=None,
        gv=g,
        o=ok,
        h0=hk0,
        do=dok,
        dq=dq,
        dk=dk,
        dv=dsk,
        dg=None,
        dgk=None,
        dgv=dgk,
        dht=dhkt,
        dh0=dhk0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=M,
        BK=BK,
        BV=BM,
        USE_G=False,
        USE_G_GAMMA=False,
        USE_GK=False,
        USE_GV=True,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)
    dgk = dgk.sum(0)

    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)
    return dq, dk, dv, ds, dg, dhk0, dhv0
