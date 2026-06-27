# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2023-2025

import math
import os

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _atk_backward_chunk_out(
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
    g_log_atk_scale_out,  # *f32 [H] - per-head log_atk_scale gradients (atomic add)
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
    BK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Per-chunk backward kernel for ATK. K-tiled variant.
    """
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n
    else:
        b = tl.program_id(0)
        i_n = b
        bos = 0
        eos = T

    if h >= H:
        return

    if chunk_id * CHUNK_LEN >= T:
        return

    T_range = chunk_id * CHUNK_LEN + tl.arange(0, CHUNK_LEN)
    mask_T = T_range < T
    C_range = tl.arange(0, CHUNK_LEN)

    if IS_VARLEN:
        log_g_ptr = log_g + h * log_g_stride_h + (log_g_stride_t * (bos + T_range))
        beta_ptr = beta + h * beta_stride_h + (beta_stride_t * (bos + T_range))
    else:
        log_g_ptr = log_g + b * log_g_stride_b + h * log_g_stride_h + (log_g_stride_t * T_range)
        beta_ptr = beta + b * beta_stride_b + h * beta_stride_h + (beta_stride_t * T_range)

    log_g_val = tl.load(log_g_ptr, mask=mask_T, other=0.0).to(tl.float32)
    beta_val = tl.load(beta_ptr, mask=mask_T, other=0.0).to(tl.float32)
    g_val = tl.exp(log_g_val)

    la_cumsum = tl.cumsum(log_g_val)

    roll_mat = (C_range[:, None] == (C_range[None, :] + 1)).to(tl.float32)
    la_cumsum_roll = tl.sum(roll_mat[:, :] * la_cumsum[None, :], 1)
    M = tl.exp(la_cumsum_roll[:, None] - la_cumsum[None, :])
    M = tl.where(C_range[:, None] > C_range[None, :], M, 0.0)

    base_decays = tl.exp(la_cumsum_roll * (C_range > 0).to(tl.float32))

    center = tl.load(log_atk_scale + h).to(tl.float32)

    gg_val = tl.zeros([CHUNK_LEN], dtype=tl.float32)
    gbeta_val = tl.zeros([CHUNK_LEN], dtype=tl.float32)
    g_center = tl.zeros([], dtype=tl.float32)
    gbase_decays = tl.zeros([CHUNK_LEN], dtype=tl.float32)
    gM = tl.zeros([CHUNK_LEN, CHUNK_LEN], dtype=tl.float32)

    for i_k in range(tl.cdiv(D, BK)):
        d_offset = i_k * BK
        D_range = d_offset + tl.arange(0, BK)
        mask_D = D_range < D

        if IS_VARLEN:
            k_ptr = k + h * k_stride_h + (k_stride_t * (bos + T_range))[:, None] + (k_stride_d) * D_range[None, :]
            gkp_ptr = g_kp + h * gkp_stride_h + (gkp_stride_t * (bos + T_range))[:, None] + (gkp_stride_d) * D_range[None, :]
        else:
            k_ptr = k + b * k_stride_b + h * k_stride_h + (k_stride_t * T_range)[:, None] + (k_stride_d) * D_range[None, :]
            gkp_ptr = g_kp + b * gkp_stride_b + h * gkp_stride_h + \
                (gkp_stride_t * T_range)[:, None] + (gkp_stride_d) * D_range[None, :]

        k_val = tl.load(k_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)
        gkp_val = tl.load(gkp_ptr, mask=mask_T[:, None] * mask_D[None, :], other=0.0).to(tl.float32)

        k_sq = k_val * k_val
        U = beta_val[:, None] * k_sq

        ac_ptr = ac + b * ac_stride_b + h * ac_stride_h + (chunk_id - 1) * ac_stride_c + D_range * ac_stride_d
        if chunk_id == 0:
            if USE_INITIAL_STATE:
                ac_val = tl.load(h0 + (i_n * H + h) * D + D_range, mask=mask_D, other=0).to(tl.float32)
            else:
                ac_val = tl.zeros([BK], dtype=tl.float32)
        else:
            ac_val = tl.load(ac_ptr, mask=mask_D)

        raw_state = base_decays[:, None] * ac_val[None, :] + tl.dot(M, U)
        A_t = g_val[:, None] * raw_state + U

        ell = tl.log(A_t + eps)
        r = ell - center
        abs_r = tl.abs(r)
        one_plus_abs_r = 1.0 + abs_r
        s = r / one_plus_abs_r
        ds_dr = 1.0 / (one_plus_abs_r * one_plus_abs_r)
        M_mult = tl.exp(-logx * s)

        gk_val = gkp_val * M_mult
        gM_from_kprecond = gkp_val * k_val
        gs = gM_from_kprecond * (-logx * M_mult)

        gr_local = gs * ds_dr
        g_center += -tl.sum(gr_local)
        gA_t = gr_local / (A_t + eps)
        graw_state = gA_t * g_val[:, None]
        gg_val += tl.sum(gA_t * raw_state, 1)
        gU = gA_t

        gU_from_M = tl.dot(tl.trans(M), graw_state)

        gk_sq = (gU + gU_from_M) * beta_val[:, None]
        gbeta_val += tl.sum(gU * k_sq, 1) + tl.sum(gU_from_M * k_sq, 1)
        gk_val += 2 * k_val * gk_sq

        gac_tile = tl.sum(graw_state * base_decays[:, None], 0)
        gac_ptr = gac_prev + i_n * gac_stride_b + h * gac_stride_h + (chunk_id - 1) * gac_stride_c + D_range * gac_stride_d
        if chunk_id > 0:
            tl.store(gac_ptr, gac_tile, mask=mask_D)
        elif USE_INITIAL_STATE:
            tl.store(dh0 + (i_n * H + h) * D + D_range, gac_tile, mask=mask_D)

        gbase_decays += tl.sum(graw_state * ac_val[None, :], 1)
        gM += tl.dot(graw_state, tl.trans(U))

        if IS_VARLEN:
            gk_ptr = gk_out + (bos + T_range)[:, None] * gk_stride_t + h * gk_stride_h + D_range[None, :] * gk_stride_d
        else:
            gk_ptr = gk_out + b * gk_stride_b + T_range[:, None] * \
                gk_stride_t + h * gk_stride_h + D_range[None, :] * gk_stride_d
        tl.store(gk_ptr, gk_val, mask=mask_T[:, None] * mask_D[None, :])

    glog_g_val = gg_val * g_val

    gbase_decays = gbase_decays * (C_range > 0)
    gla_cumsum_roll = gbase_decays * base_decays

    gM = tl.where(C_range[:, None] > C_range[None, :], gM, 0.0)
    ginner_sum = M * gM
    gla_cumsum_roll += tl.sum(ginner_sum, 1)
    gla_cumsum = -1 * tl.sum(ginner_sum, 0)

    roll_mat = (C_range[:, None] == (C_range[None, :] + 1)).to(tl.float32)
    gla_cumsum += tl.sum(roll_mat * gla_cumsum_roll[:, None], 0) * (C_range < (CHUNK_LEN - 1)).to(tl.float32)

    glog_g_val += tl.cumsum(gla_cumsum, reverse=True)

    if IS_VARLEN:
        gg_ptr = gg_out + (bos + T_range) * gg_stride_t + h * gg_stride_h
        gbeta_ptr = gbeta_out + (bos + T_range) * gbeta_stride_t + h * gbeta_stride_h
    else:
        gg_ptr = gg_out + b * gg_stride_b + T_range * gg_stride_t + h * gg_stride_h
        gbeta_ptr = gbeta_out + b * gbeta_stride_b + T_range * gbeta_stride_t + h * gbeta_stride_h

    tl.store(gg_ptr, glog_g_val, mask=mask_T)
    tl.store(gbeta_ptr, gbeta_val, mask=mask_T)
    tl.atomic_add(g_log_atk_scale_out + h, g_center)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _atk_backward_pass_chunks(
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
):
    """
    Sequential backward pass across chunks.
    Not K-tiled due to cross-D reduction in gsa_val computation.
    """
    i_nh = tl.program_id(0)

    if IS_VARLEN:
        i_n = i_nh // H
        h = i_nh % H
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n
    else:
        b = i_nh // H
        h = i_nh % H
        i_n = b
        if b >= B:
            return

    if h >= H:
        return

    N_chunks = tl.cdiv(T, CHUNK_LEN)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < D

    sa_ptr = sa + b * sa_stride_b + h * sa_stride_h + (N_chunks - 1) * sa_stride_c
    a_ptr = a + b * a_stride_b + h * a_stride_h + D_range * a_stride_d + (N_chunks - 1) * a_stride_c
    ac_ptr = ac + b * ac_stride_b + h * ac_stride_h + D_range * ac_stride_d + (N_chunks - 1) * ac_stride_c
    gac_ptr = gac_from_out + b * gac_stride_b + h * gac_stride_h + D_range * gac_stride_d + (N_chunks - 1) * gac_stride_c
    gsa_ptr = gsa + b * gsa_stride_b + h * gsa_stride_h + (N_chunks - 1) * gsa_stride_c
    ga_ptr = ga + b * ga_stride_b + h * ga_stride_h + D_range * ga_stride_d + (N_chunks - 1) * ga_stride_c

    gac_val = tl.zeros([BLOCK_D], dtype=tl.float32)

    for chunk_id in tl.range(N_chunks - 1, -1, -1):
        gac_val += tl.load(gac_ptr, mask=D_mask)

        tl.store(ga_ptr, gac_val, mask=D_mask)

        sa_val = tl.load(sa_ptr).to(tl.float32)

        if chunk_id > 0:
            ac_prev_ptr = ac + b * ac_stride_b + h * ac_stride_h + D_range * ac_stride_d + (chunk_id - 1) * ac_stride_c
            ac_prev_val = tl.load(ac_prev_ptr, mask=D_mask)
            gsa_val = tl.sum(gac_val * ac_prev_val) * tl.exp(sa_val)
        else:
            if USE_INITIAL_STATE:
                h0_val = tl.load(h0 + i_nh * D + D_range, mask=D_mask, other=0).to(tl.float32)
                gsa_val = tl.sum(gac_val * h0_val) * tl.exp(sa_val)
            else:
                gsa_val = 0.0

        gac_val = gac_val * tl.exp(sa_val)

        tl.store(gsa_ptr, gsa_val)

        sa_ptr -= sa_stride_c
        a_ptr -= a_stride_c
        ac_ptr -= ac_stride_c
        gac_ptr -= gac_stride_c
        ga_ptr -= ga_stride_c
        gsa_ptr -= gsa_stride_c

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * D + D_range
        tl.store(p_dh0, tl.load(p_dh0, mask=D_mask, other=0).to(tl.float32) + gac_val, mask=D_mask)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _atk_backward_chunk_summary(
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
):
    i_t = tl.program_id(0)
    h = tl.program_id(1)
    chunk_id = tl.program_id(2)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        b = i_n
    else:
        b = tl.program_id(0)
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
    glog_g_val = -1 * tl.cumsum(gdecays_exp, reverse=True)

    glog_g_val += gsa_val

    if IS_VARLEN:
        gg_ptr = gg_out + (bos + T_range) * gg_stride_t + h * gg_stride_h
        gbeta_ptr = gbeta_out + (bos + T_range) * gbeta_stride_t + h * gbeta_stride_h
    else:
        gg_ptr = gg_out + b * gg_stride_b + T_range * gg_stride_t + h * gg_stride_h
        gbeta_ptr = gbeta_out + b * gbeta_stride_b + T_range * gbeta_stride_t + h * gbeta_stride_h

    tl.atomic_add(gg_ptr, glog_g_val, mask=mask_T)
    tl.atomic_add(gbeta_ptr, gbeta_val, mask=mask_T)


def chunk_atk_bwd(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    r"""
    ATK backward pass. Takes precomputed forward intermediates from ``recompute_atk_fwd``.

    Args:
        k: Keys ``[B, T, H, K]``.
        g_raw: Raw gate values (NOT cumsum) ``[B, T, H]``.
        beta: Beta scaling ``[B, T, H]``.
        dk_precond: Gradient w.r.t. k_precond ``[B, T, H, K]``.
        ac: Prefix-sum ATK states from ``recompute_atk_fwd`` ``[B, C, H, K]``.
        a: Per-chunk summaries from ``recompute_atk_fwd`` ``[B, C, H, K]``.
        sa: Per-chunk log-gate sums from ``recompute_atk_fwd`` ``[B, C, H]``.
        chunk_size: Chunk size.
        initial_A_state: Initial ATK diagonal state ``[N, H, K]``.
        cu_seqlens: Cumulative sequence lengths ``[N+1]`` for varlen.
        x: Squash range parameter.
        eps: Epsilon for numerical stability.
        log_atk_scale: Per-head log-space center ``[H]``.

    Returns:
        dk_atk: Gradient contribution to k ``[B, T, H, K]``.
        dbeta_atk: Gradient for beta ``[B, T, H]``.
        dg_atk: Gradient for g ``[B, T, H]``.
        d_log_atk_scale: Gradient for log_atk_scale ``[H]``.
        dh0_atk: Gradient for initial_A_state ``[N, H, K]`` or ``None``.
    """
    B, T, H, K = k.shape
    CHUNK_LEN = chunk_size

    is_varlen = cu_seqlens is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_LEN) if is_varlen else None
    NT = triton.cdiv(T, CHUNK_LEN) if not is_varlen else len(chunk_indices)
    N = len(cu_seqlens) - 1 if is_varlen else B

    BK = K if os.environ.get('ATK_NO_KTILE') else 32  # K-tile size (set ATK_NO_KTILE=1 to disable)
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
    g_log_atk_scale = torch.zeros(H, device=k.device, dtype=torch.float32)
    gg = torch.zeros_like(g_raw, dtype=torch.float32)
    gbeta = torch.zeros_like(beta, dtype=torch.float32)
    gac_prev = torch.zeros_like(ac, dtype=torch.float32)
    dh0 = torch.zeros(N, H, K, device=k.device, dtype=torch.float32) if initial_A_state is not None else None
    ga = torch.empty_like(a, dtype=torch.float32)
    gsa = torch.empty_like(sa, dtype=torch.float32)

    if cu_seqlens is None:
        grid = (B, H, triton.cdiv(T, CHUNK_LEN))
        grid2 = (B * H,)
    else:
        grid = (NT, H, 1)
        grid2 = (N * H,)

    _atk_backward_chunk_out[grid](
        k, g_raw, beta, ac, initial_A_state, dk_precond,
        gk, gg, gbeta, gac_prev, dh0,
        g_log_atk_scale,
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
        BK=BK, num_warps=4,
    )

    _atk_backward_pass_chunks[grid2](
        a, sa, ac,
        initial_A_state,
        gac_prev,
        ga, gsa,
        dh0,
        cu_seqlens,
        B, T, H, K, CHUNK_LEN,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        sa.stride(0), sa.stride(1), sa.stride(2),
        ac.stride(0), ac.stride(1), ac.stride(2), ac.stride(3),
        gac_prev.stride(0), gac_prev.stride(1), gac_prev.stride(2), gac_prev.stride(3),
        ga.stride(0), ga.stride(1), ga.stride(2), ga.stride(3),
        gsa.stride(0), gsa.stride(1), gsa.stride(2),
        BLOCK_D, num_warps=4
    )

    _atk_backward_chunk_summary[grid](
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
        BK, num_warps=4
    )

    return gk, gbeta, gg, g_log_atk_scale, dh0
