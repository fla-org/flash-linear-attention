# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunked ATK backward for triton-ascend on Ascend NPU."""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils.ascend_ub_manager import ASCEND_LAUNCH_BLOCK_BUDGET

_NPU_CHUNK_LEN = 32


@triton.jit
def _reverse_cumsum_chunk(x, CHUNK_LEN: tl.constexpr):
    return tl.cumsum(x, reverse=True)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'chunk_offset'])
def _atk_backward_chunk_out_npu(
    k,                # *f32 [B, T, H, D]
    log_g,            # *f32 [B, T, H]
    beta,             # *f32 [B, T, H] - beta scaling
    ac,               # *f32 [B, C, H, D]  (forward prefix contexts)
    h0,               # *f32 [N, H, D] or None - initial ATK state
    g_kp,             # *f32 [B, T, H, D]  upstream grad on k_precond
    # outputs (accumulators; +=)
    gk_out,           # *f32 [B, T, H, D]
    gg_out,           # *f32 [B, T, H]
    gbeta_out,        # *f32 [B, T, H] - gradient for beta
    gac_prev,         # *f32 [B, C, H, D]  grad wrt ac_{i-1}, stored at index i-1
    dh0,              # *f32 [N, H, D] or None - gradient for initial ATK state
    g_log_atk_scale_chunks,  # *f32 [N, C, H] per-chunk log_atk_scale grads
    cu_seqlens,       # *i32 [N+1] - cumulative sequence lengths
    chunk_indices,    # *i32 [NT, 2] - (seq_idx, chunk_idx) pairs
    log_atk_scale,    # *f32 [H] - per-head log-space center (learnable or fixed)
    logx,             # scalar float32 - log(x) for squash range
    eps,              # scalar float32 - epsilon for log safety
    B: tl.constexpr, T, H: tl.constexpr, D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    k_stride_b, k_stride_t, k_stride_h, k_stride_d,
    log_g_stride_b, log_g_stride_t, log_g_stride_h,
    beta_stride_b, beta_stride_t, beta_stride_h,
    ac_stride_b, ac_stride_c, ac_stride_h, ac_stride_d,
    gkp_stride_b, gkp_stride_t, gkp_stride_h, gkp_stride_d,
    gk_stride_b, gk_stride_t, gk_stride_h, gk_stride_d,
    gg_stride_b, gg_stride_t, gg_stride_h,
    gbeta_stride_b, gbeta_stride_t, gbeta_stride_h,
    gac_stride_b, gac_stride_c, gac_stride_h, gac_stride_d,
    gscale_stride_b, gscale_stride_c, gscale_stride_h,
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    chunk_offset,
):
    """
    Per-chunk backward kernel for ATK. K-tiled variant.
    """
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2).to(tl.int64) + chunk_offset

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n.to(tl.int64)
    else:
        b = tl.program_id(0).to(tl.int64)
        i_n = b
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    C_range = tl.arange(0, CHUNK_LEN)
    center = tl.load(log_atk_scale + h).to(tl.float32)
    gg_val = tl.zeros([CHUNK_LEN], dtype=tl.float32)
    gbeta_val = tl.zeros([CHUNK_LEN], dtype=tl.float32)
    g_center = tl.zeros((), dtype=tl.float32)

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        ac_ptr = ac + b * ac_stride_b + h * ac_stride_h + (chunk_id - 1) * ac_stride_c + D_range * ac_stride_d
        if chunk_id == 0:
            if USE_INITIAL_STATE:
                ac_val = tl.load(h0 + (i_n * H + h) * D + D_range, mask=mask_D, other=0).to(tl.float32)
            else:
                ac_val = tl.zeros([BK], dtype=tl.float32)
        else:
            ac_val = tl.load(ac_ptr, mask=mask_D).to(tl.float32)

        ac_hist = tl.zeros([CHUNK_LEN, BK], dtype=tl.float32)
        for t in tl.range(CHUNK_LEN):
            t_idx = chunk_id * CHUNK_LEN + t
            valid = t_idx < T
            row = C_range == t
            ac_hist = tl.where(row[:, None], ac_val[None, :], ac_hist)
            if IS_VARLEN:
                beta_t = tl.load(beta + (bos + t_idx) * beta_stride_t + h *
                                 beta_stride_h, mask=valid, other=0.0).to(tl.float32)
                log_g_t = tl.load(log_g + (bos + t_idx) * log_g_stride_t + h *
                                  log_g_stride_h, mask=valid, other=0.0).to(tl.float32)
                k_ptr = k + (bos + t_idx) * k_stride_t + h * k_stride_h + D_range * k_stride_d
            else:
                beta_t = tl.load(beta + b * beta_stride_b + t_idx * beta_stride_t +
                                 h * beta_stride_h, mask=valid, other=0.0).to(tl.float32)
                log_g_t = tl.load(log_g + b * log_g_stride_b + t_idx * log_g_stride_t +
                                  h * log_g_stride_h, mask=valid, other=0.0).to(tl.float32)
                k_ptr = k + b * k_stride_b + t_idx * k_stride_t + h * k_stride_h + D_range * k_stride_d
            k_val = tl.load(k_ptr, mask=mask_D & valid, other=0.0).to(tl.float32)
            A_t = tl.exp(log_g_t) * ac_val + beta_t * k_val * k_val
            ac_val = tl.where(valid, A_t, ac_val)

        g_ac = tl.zeros([BK], dtype=tl.float32)
        for t in tl.range(CHUNK_LEN - 1, -1, -1):
            t_idx = chunk_id * CHUNK_LEN + t
            valid = t_idx < T
            row = C_range == t
            ac_before = tl.sum(tl.where(row[:, None], ac_hist, 0.0), 0)
            if IS_VARLEN:
                beta_t = tl.load(beta + (bos + t_idx) * beta_stride_t + h *
                                 beta_stride_h, mask=valid, other=0.0).to(tl.float32)
                log_g_t = tl.load(log_g + (bos + t_idx) * log_g_stride_t + h *
                                  log_g_stride_h, mask=valid, other=0.0).to(tl.float32)
                k_ptr = k + (bos + t_idx) * k_stride_t + h * k_stride_h + D_range * k_stride_d
                gkp_ptr = g_kp + (bos + t_idx) * gkp_stride_t + h * gkp_stride_h + D_range * gkp_stride_d
                gk_ptr = gk_out + (bos + t_idx) * gk_stride_t + h * gk_stride_h + D_range * gk_stride_d
            else:
                beta_t = tl.load(beta + b * beta_stride_b + t_idx * beta_stride_t +
                                 h * beta_stride_h, mask=valid, other=0.0).to(tl.float32)
                log_g_t = tl.load(log_g + b * log_g_stride_b + t_idx * log_g_stride_t +
                                  h * log_g_stride_h, mask=valid, other=0.0).to(tl.float32)
                k_ptr = k + b * k_stride_b + t_idx * k_stride_t + h * k_stride_h + D_range * k_stride_d
                gkp_ptr = g_kp + b * gkp_stride_b + t_idx * gkp_stride_t + h * gkp_stride_h + D_range * gkp_stride_d
                gk_ptr = gk_out + b * gk_stride_b + t_idx * gk_stride_t + h * gk_stride_h + D_range * gk_stride_d
            k_val = tl.load(k_ptr, mask=mask_D & valid, other=0.0).to(tl.float32)
            gkp_val = tl.load(gkp_ptr, mask=mask_D & valid, other=0.0).to(tl.float32)
            g_exp = tl.exp(log_g_t)
            u = beta_t * k_val * k_val
            A_t = g_exp * ac_before + u
            r = tl.log(A_t + eps) - center
            abs_r = tl.abs(r)
            one_plus = 1.0 + abs_r
            s = r / one_plus
            ds_dr = 1.0 / (one_plus * one_plus)
            M_mult = tl.exp(-logx * s)
            gr = (gkp_val * k_val) * (-logx * M_mult) * ds_dr
            g_center += -tl.sum(tl.where(valid, gr, 0.0))
            gA_local = gr / (A_t + eps)
            gA = tl.where(valid, gA_local + g_ac, 0.0)
            g_ac = tl.where(valid, gA * g_exp, g_ac)
            gg_add = tl.sum(tl.where(valid, gA * g_exp * ac_before, 0.0))
            gbeta_add = tl.sum(tl.where(valid, gA * k_val * k_val, 0.0))
            gg_val = tl.where(row, gg_val + gg_add, gg_val)
            gbeta_val = tl.where(row, gbeta_val + gbeta_add, gbeta_val)
            gk_val = gkp_val * M_mult + 2.0 * k_val * beta_t * gA
            tl.store(gk_ptr, gk_val, mask=mask_D & valid)

        gac_ptr = gac_prev + i_n * gac_stride_b + h * gac_stride_h + (chunk_id - 1) * gac_stride_c + D_range * gac_stride_d
        if chunk_id > 0:
            tl.store(gac_ptr, g_ac, mask=mask_D)
        elif USE_INITIAL_STATE:
            tl.store(dh0 + (i_n * H + h) * D + D_range, g_ac, mask=mask_D)

    for t in tl.range(CHUNK_LEN):
        t_idx = chunk_id * CHUNK_LEN + t
        valid = t_idx < T
        gg_t = tl.sum(tl.where(C_range == t, gg_val, 0.0))
        gbeta_t = tl.sum(tl.where(C_range == t, gbeta_val, 0.0))
        if IS_VARLEN:
            tl.store(gg_out + (bos + t_idx) * gg_stride_t + h * gg_stride_h, gg_t, mask=valid)
            tl.store(gbeta_out + (bos + t_idx) * gbeta_stride_t + h * gbeta_stride_h, gbeta_t, mask=valid)
        else:
            tl.store(gg_out + b * gg_stride_b + t_idx * gg_stride_t + h * gg_stride_h, gg_t, mask=valid)
            tl.store(gbeta_out + b * gbeta_stride_b + t_idx * gbeta_stride_t + h * gbeta_stride_h, gbeta_t, mask=valid)
    gscale_ptr = g_log_atk_scale_chunks + i_n * gscale_stride_b + chunk_id * gscale_stride_c + h * gscale_stride_h
    tl.store(gscale_ptr, g_center)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'chunk_id'])
def _atk_backward_pass_one_chunk_npu(
    a, sa, ac,            # forward buffers
    h0,                   # *f32 [N, H, D] or None - initial ATK state
    gac_from_out,         # grad entering each ac[i] from later usage
    ga, gsa,              # outputs (+=)
    dh0,                  # *f32 [N, H, D] or None - gradient for initial ATK state (accumulated)
    cu_seqlens,           # *i32 [N+1] - cumulative sequence lengths
    B: tl.constexpr, T, H: tl.constexpr, D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    a_stride_b, a_stride_c, a_stride_h, a_stride_d,
    sa_stride_b, sa_stride_c, sa_stride_h,
    ac_stride_b, ac_stride_c, ac_stride_h, ac_stride_d,
    gac_stride_b, gac_stride_c, gac_stride_h, gac_stride_d,
    ga_stride_b, ga_stride_c, ga_stride_h, ga_stride_d,
    gsa_stride_b, gsa_stride_c, gsa_stride_h,
    BLOCK_D: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    chunk_id,
):
    """
    One chunk of the backward state scan. Host loops chunks in reverse so the
    carry is not inside an unreliable runtime tl.range.
    Not K-tiled due to cross-D reduction in gsa_val computation.
    """
    i_nh = tl.program_id(0).to(tl.int64)

    if IS_VARLEN:
        i_n = i_nh // H
        h = i_nh % H
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n.to(tl.int64)
    else:
        b = i_nh // H
        h = i_nh % H
        i_n = b
        if b >= B:
            return

    if h >= H or chunk_id * CHUNK_LEN >= T:
        return

    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < D
    gac_ptr = gac_from_out + b * gac_stride_b + h * gac_stride_h + chunk_id * gac_stride_c + D_range * gac_stride_d
    ga_ptr = ga + b * ga_stride_b + h * ga_stride_h + chunk_id * ga_stride_c + D_range * ga_stride_d
    gsa_ptr = gsa + b * gsa_stride_b + h * gsa_stride_h + chunk_id * gsa_stride_c
    sa_ptr = sa + b * sa_stride_b + h * sa_stride_h + chunk_id * sa_stride_c

    gac_val = tl.load(gac_ptr, mask=D_mask, other=0).to(tl.float32)
    # Later chunks add their propagated state grad into this slot before launch.
    tl.store(ga_ptr, gac_val, mask=D_mask)

    sa_val = tl.load(sa_ptr).to(tl.float32)
    gac_next = gac_val * tl.exp(sa_val)
    if chunk_id > 0:
        ac_prev_ptr = ac + b * ac_stride_b + h * ac_stride_h + (chunk_id - 1) * ac_stride_c + D_range * ac_stride_d
        ac_prev_val = tl.load(ac_prev_ptr, mask=D_mask, other=0).to(tl.float32)
        gsa_val = tl.sum(gac_val * ac_prev_val) * tl.exp(sa_val)
        prev_gac = gac_from_out + b * gac_stride_b + h * gac_stride_h + (chunk_id - 1) * gac_stride_c + D_range * gac_stride_d
        tl.store(prev_gac, tl.load(prev_gac, mask=D_mask, other=0).to(tl.float32) + gac_next, mask=D_mask)
    elif USE_INITIAL_STATE:
        h0_val = tl.load(h0 + i_nh * D + D_range, mask=D_mask, other=0).to(tl.float32)
        gsa_val = tl.sum(gac_val * h0_val) * tl.exp(sa_val)
        p_dh0 = dh0 + i_nh * D + D_range
        tl.store(p_dh0, tl.load(p_dh0, mask=D_mask, other=0).to(tl.float32) + gac_next, mask=D_mask)
    else:
        gsa_val = 0.0
    tl.store(gsa_ptr, gsa_val)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'chunk_offset'])
def _atk_backward_chunk_summary_npu(
    k, log_g, beta,    # inputs for recompute
    ga, gsa,           # grads from scan-backward
    gk_out, gg_out, gbeta_out,
    cu_seqlens,        # *i32 [N+1] - cumulative sequence lengths
    chunk_indices,     # *i32 [NT, 2] - (seq_idx, chunk_idx) pairs
    B: tl.constexpr, T, H: tl.constexpr, D: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    k_stride_b, k_stride_t, k_stride_h, k_stride_d,
    lg_stride_b, lg_stride_t, lg_stride_h,
    beta_stride_b, beta_stride_t, beta_stride_h,
    ga_stride_b, ga_stride_c, ga_stride_h, ga_stride_d,
    gsa_stride_b, gsa_stride_c, gsa_stride_h,
    gk_stride_b, gk_stride_t, gk_stride_h, gk_stride_d,
    gg_stride_b, gg_stride_t, gg_stride_h,
    gbeta_stride_b, gbeta_stride_t, gbeta_stride_h,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    chunk_offset,
):
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2).to(tl.int64) + chunk_offset

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n.to(tl.int64)
    else:
        b = tl.program_id(0).to(tl.int64)
        i_n = b
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    T_range = chunk_id * CHUNK_LEN + tl.arange(0, CHUNK_LEN)
    mask_T = T_range < T

    if IS_VARLEN:
        log_g_ptr = log_g + (bos + T_range) * lg_stride_t + h * lg_stride_h
        beta_ptr = beta + (bos + T_range) * beta_stride_t + h * beta_stride_h
    else:
        log_g_ptr = log_g + b * lg_stride_b + T_range * lg_stride_t + h * lg_stride_h
        beta_ptr = beta + b * beta_stride_b + T_range * beta_stride_t + h * beta_stride_h

    log_g_val = tl.load(log_g_ptr, mask=mask_T, other=0.0).to(tl.float32)
    beta_val = tl.load(beta_ptr, mask=mask_T, other=0.0).to(tl.float32)

    sa_val = tl.sum(log_g_val)
    decays = tl.exp(sa_val - tl.cumsum(log_g_val))

    gsa_ptr = gsa + i_n * gsa_stride_b + h * gsa_stride_h + chunk_id * gsa_stride_c
    gsa_val = tl.load(gsa_ptr).to(tl.float32)

    gdecays = tl.zeros([CHUNK_LEN], dtype=tl.float32)
    gbeta_val = tl.zeros([CHUNK_LEN], dtype=tl.float32)

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        if IS_VARLEN:
            k_ptr = k + (bos + T_range)[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d
            gk_ptr = gk_out + (bos + T_range)[:, None] * gk_stride_t + h * gk_stride_h + D_range[None, :] * gk_stride_d
        else:
            k_ptr = k + b * k_stride_b + T_range[:, None] * k_stride_t + h * k_stride_h + D_range[None, :] * k_stride_d
            gk_ptr = gk_out + b * gk_stride_b + T_range[:, None] * \
                gk_stride_t + h * gk_stride_h + D_range[None, :] * gk_stride_d

        ga_ptr = ga + i_n * ga_stride_b + h * ga_stride_h + chunk_id * ga_stride_c + (ga_stride_d) * D_range

        k_val = tl.load(k_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)
        ga_val = tl.load(ga_ptr, mask=mask_D, other=0.0).to(tl.float32)

        k_sq = k_val * k_val
        U = beta_val[:, None] * k_sq

        gdecays += tl.sum(ga_val[None, :] * U, 1)

        gU = ga_val[None, :] * decays[:, None]
        gbeta_val += tl.sum(gU * k_sq, 1)

        gk_sq = gU * beta_val[:, None]
        gk_val = gk_sq * k_val * 2

        tl.atomic_add(gk_ptr, gk_val, mask=mask_T[:, None] * mask_D[None, :])

    gdecays_exp = decays * gdecays

    gsa_val += tl.sum(gdecays_exp)
    glog_g_val = -1 * _reverse_cumsum_chunk(gdecays_exp, CHUNK_LEN)

    glog_g_val += gsa_val

    if IS_VARLEN:
        gg_ptr = gg_out + (bos + T_range) * gg_stride_t + h * gg_stride_h
        gbeta_ptr = gbeta_out + (bos + T_range) * gbeta_stride_t + h * gbeta_stride_h
    else:
        gg_ptr = gg_out + b * gg_stride_b + T_range * gg_stride_t + h * gg_stride_h
        gbeta_ptr = gbeta_out + b * gbeta_stride_b + T_range * gbeta_stride_t + h * gbeta_stride_h

    tl.atomic_add(gg_ptr, glog_g_val, mask=mask_T)
    tl.atomic_add(gbeta_ptr, gbeta_val, mask=mask_T)


def _launch_chunk_grid(kernel, grid_b, grid_h, num_chunks, args):
    max_c = max(1, ASCEND_LAUNCH_BLOCK_BUDGET // max(grid_b * grid_h, 1))
    for off in range(0, num_chunks, max_c):
        n = min(max_c, num_chunks - off)
        kernel[(grid_b, grid_h, n)](*args, chunk_offset=off)


def chunk_atk_bwd_npu(
    k: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    dk_precond: torch.Tensor,
    ac: torch.Tensor,
    a: torch.Tensor,
    sa: torch.Tensor,
    chunk_size: int = 64,
    initial_A_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor = None,
    dat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    del chunk_size
    B, T, H, K = k.shape
    CHUNK_LEN = _NPU_CHUNK_LEN

    is_varlen = cu_seqlens is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_LEN) if is_varlen else None
    NT = triton.cdiv(T, CHUNK_LEN) if not is_varlen else len(chunk_indices)
    N = len(cu_seqlens) - 1 if is_varlen else B

    # Single K-tile on NPU keeps chunk_out backward within UB budget.
    BK = K if os.environ.get('ATK_NO_KTILE', '1') else 32
    BLOCK_D = triton.next_power_of_2(K)

    k = k.contiguous()
    g_raw = g_raw.contiguous()
    beta = beta.contiguous()
    dk_precond = dk_precond.contiguous()

    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32, device=k.device)
    else:
        log_atk_scale = log_atk_scale.contiguous()

    logx = math.log(x) if x > 0 else 0.0

    gk = torch.zeros_like(k, dtype=torch.float32)
    g_log_atk_scale_chunks = torch.zeros(N, ac.shape[1], H, device=k.device, dtype=torch.float32)
    gg = torch.zeros_like(g_raw, dtype=torch.float32)
    gbeta = torch.zeros_like(beta, dtype=torch.float32)
    gac_prev = torch.zeros_like(ac, dtype=torch.float32)
    if dat is not None:
        if cu_seqlens is None:
            gac_prev[:, NT - 1] += dat
        else:
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            last_chunk = torch.clamp((seqlens + CHUNK_LEN - 1) // CHUNK_LEN - 1, min=0)
            valid = seqlens > 0
            idx = torch.arange(N, device=k.device)
            gac_prev[idx[valid], last_chunk[valid]] += dat[valid]
    dh0 = torch.zeros(N, H, K, device=k.device, dtype=torch.float32) if initial_A_state is not None else None
    ga = torch.zeros_like(a, dtype=torch.float32)
    gsa = torch.zeros_like(sa, dtype=torch.float32)

    if cu_seqlens is None:
        grid_b, grid_h, num_chunks = B, H, triton.cdiv(T, CHUNK_LEN)
        grid2 = (B * H,)
    else:
        grid_b, grid_h, num_chunks = NT, H, 1
        grid2 = (N * H,)

    chunk_out_args = (
        k, g_raw, beta, ac, initial_A_state, dk_precond,
        gk, gg, gbeta, gac_prev, dh0,
        g_log_atk_scale_chunks,
        cu_seqlens, chunk_indices,
        log_atk_scale, logx, eps,
        B, T, H, K, CHUNK_LEN,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g_raw.stride(0), g_raw.stride(1), g_raw.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        ac.stride(0), ac.stride(1), ac.stride(2), ac.stride(3),
        dk_precond.stride(0), dk_precond.stride(1), dk_precond.stride(2), dk_precond.stride(3),
        gk.stride(0), gk.stride(1), gk.stride(2), gk.stride(3),
        gg.stride(0), gg.stride(1), gg.stride(2),
        gbeta.stride(0), gbeta.stride(1), gbeta.stride(2),
        gac_prev.stride(0), gac_prev.stride(1), gac_prev.stride(2), gac_prev.stride(3),
        g_log_atk_scale_chunks.stride(0), g_log_atk_scale_chunks.stride(1), g_log_atk_scale_chunks.stride(2),
        BK,
    )
    summary_args = (
        k, g_raw, beta,
        ga, gsa,
        gk, gg, gbeta,
        cu_seqlens, chunk_indices,
        B, T, H, K, CHUNK_LEN,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g_raw.stride(0), g_raw.stride(1), g_raw.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        ga.stride(0), ga.stride(1), ga.stride(2), ga.stride(3),
        gsa.stride(0), gsa.stride(1), gsa.stride(2),
        gk.stride(0), gk.stride(1), gk.stride(2), gk.stride(3),
        gg.stride(0), gg.stride(1), gg.stride(2),
        gbeta.stride(0), gbeta.stride(1), gbeta.stride(2),
        BK,
    )

    if is_varlen:
        _atk_backward_chunk_out_npu[(NT, H, 1)](*chunk_out_args, chunk_offset=0)
    else:
        _launch_chunk_grid(_atk_backward_chunk_out_npu, grid_b, grid_h, num_chunks, chunk_out_args)

    pass_args = (
        a, sa, ac,
        initial_A_state,
        gac_prev,
        ga, gsa,
        dh0,
        cu_seqlens,
        B if not is_varlen else N, T, H, K, CHUNK_LEN,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        sa.stride(0), sa.stride(1), sa.stride(2),
        ac.stride(0), ac.stride(1), ac.stride(2), ac.stride(3),
        gac_prev.stride(0), gac_prev.stride(1), gac_prev.stride(2), gac_prev.stride(3),
        ga.stride(0), ga.stride(1), ga.stride(2), ga.stride(3),
        gsa.stride(0), gsa.stride(1), gsa.stride(2),
        BLOCK_D,
    )
    for chunk_id in range(ac.shape[1] - 1, -1, -1):
        _atk_backward_pass_one_chunk_npu[grid2](*pass_args, chunk_id=chunk_id)

    if is_varlen:
        _atk_backward_chunk_summary_npu[(NT, H, 1)](*summary_args, chunk_offset=0)
    else:
        _launch_chunk_grid(_atk_backward_chunk_summary_npu, grid_b, grid_h, num_chunks, summary_args)

    g_log_atk_scale = g_log_atk_scale_chunks.sum(dim=(0, 1))
    return gk, gbeta, gg, g_log_atk_scale, dh0
