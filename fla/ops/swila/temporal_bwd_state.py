# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Shared temporal memory reverse kernel and its autotune reset hook."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from fla.ops.swila.atoms import dot_3d_axis2, log_softmax_2d, logaddexp, logsigmoid_pair, softmax_backward_2d, state_update
from fla.ops.swila.atoms.activations import log_gate_mix
from fla.ops.swila.autotune import prune_temporal_backward_configs
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs


def _zero_temporal_bwd_outputs(args) -> None:
    """Reset reduction buffers before each Triton autotune trial."""
    args["dq"].zero_()
    args["dk"].zero_()
    dlog_decay = args["dlog_decay"]
    if dlog_decay is not None:
        dlog_decay.zero_()


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages,
                      pre_hook=_zero_temporal_bwd_outputs)
        for BV in [8, 16, 32, 64]
        for num_warps in ([1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8])
        for num_stages in [2, 3, 4]
    ],
    key=[
        'D_k', 'D_v', 'E', 'RESOURCE_SAFE',
        'PRODUCTION_FAST', 'SPLIT_Q_TEMPORAL',
        'SKIP_REPLAY',
    ],
    prune_configs_by={"early_config_prune": prune_temporal_backward_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_swila_temporal_bwd_kernel(
    # Inputs (saved from forward)
    q,       # [B, T, H, D_k]
    k,       # [B, T, H, D_k]
    v,       # [B, T, H, D_v]
    beta,    # [B, T, H, D_v, E]
    pi_q,    # [B, T, H, D_v, E]
    pi_k,    # [B, T, H, D_v, E]
    log_decay,  # [B, T, H, D_k] (optional)
    # Saved intermediates from forward
    r,       # [B, T, H, D_v, E] - responsibilities
    u,       # [B, T, H, D_v, E] - error signals
    pi_bar,  # [B, T, H, D_v, E] - predicted prior (temporal) / r_prior (gate mode)
    rho,     # [B, T, H, D_v, E] - Q filtered state (temporal)
    qS,      # [B, T, H, D_v, E] - per-expert scaled output (temporal)
    h0,      # [B, H, D_v, E, D_k] initial state (optional)
    r_prev_init,   # [B, H, D_v, E] initial r_prev (temporal)
    rho_prev_init,  # [B, H, D_v, E] initial rho_prev (temporal)
    # Gradient inputs
    do,      # [B, T, H, D_v]
    dht,     # [B, H, D_v, E, D_k] final state gradient (optional)
    ht,      # [B, H, D_v, E, D_k] final state value (optional)
    d_r_prev_final,    # [B, H, D_v, E] final r-state gradient (optional)
    d_rho_prev_final,  # [B, H, D_v, E] final rho-state gradient (optional)
    # Gradient outputs
    dq,      # [NV, B, T, H, D_k]
    dk,      # [NV, B, T, H, D_k]
    dv,      # [NK, B, T, H, D_v]
    dbeta,   # [NK, B, T, H, D_v, E] - FIX: removed NV dimension (each i_v writes to its D_v slice)
    dpi_q,   # [NK, B, T, H, D_v, E] - FIX: removed NV dimension
    dpi_k,   # [NK, B, T, H, D_v, E] - FIX: removed NV dimension
    du_save,  # [B, T, H, D_v, E] - stores grad_u from Phase 1 for Phase 2 correction
    dlog_decay,  # shape matches log_decay; stores local terms then reverse scan
    dh0,     # [B, H, D_v, E, D_k] initial state gradient (optional)
    # FIX: Add gradient outputs for initial temporal states
    dr_prev_init,   # [B, H, D_v, E] gradient w.r.t. initial r_prev (optional)
    drho_prev_init,  # [B, H, D_v, E] gradient w.r.t. initial rho_prev (optional)
    # Auxiliary loss gradients (load balancing)
    d_r_seq,        # [B, T, H, D_v, E] gradient from aux loss w.r.t. r (optional)
    d_rho_seq,      # [B, T, H, D_v, E] gradient from aux loss w.r.t. rho (optional)
    d_u_seq,        # [B, T, H, D_v, E] direct gradient w.r.t. saved u (optional)
    d_pi_bar_seq,   # [B, T, H, D_v, E] direct gradient w.r.t. pi_bar (optional)
    # Gate inputs for reset_gate temporal coupling (optional)
    gate_kv,        # [B, T, H, D_v] - raw gate logits KV-side (optional)
    gate_q,         # [B, T, H, D_v] - raw gate logits Q-side (optional)
    d_gate_kv,      # [B, T, H, D_v] - gate gradient output (w.r.t. raw logit) KV-side
    d_gate_q,       # [B, T, H, D_v] - gate gradient output (w.r.t. raw logit) Q-side
    # Debug buffers (optional, for intermediate value extraction)
    debug_dh,       # [B, T, H, D_v, E, D_k] - stores b_dh after Section B (optional)
    cu_seqlens,     # [N+1] cumulative sequence lengths (optional, for variable-length)
    # Parameters
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_k: tl.constexpr,
    D_v: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,   # Block size for E (must be power of 2, >= E)
    NK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_INITIAL_TEMPORAL_STATE: tl.constexpr,
    STORE_TEMPORAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_R_GRADIENT: tl.constexpr,
    USE_FINAL_RHO_GRADIENT: tl.constexpr,
    USE_AUX_LOSS_GRAD: tl.constexpr,
    USE_U_AUX_GRAD: tl.constexpr,
    USE_PI_BAR_AUX_GRAD: tl.constexpr,
    DEBUG_INTERMEDIATES: tl.constexpr,
    SKIP_REPLAY: tl.constexpr,
    SPLIT_Q_TEMPORAL: tl.constexpr,
    REUSE_DK_COEF: tl.constexpr,
    RESOURCE_SAFE: tl.constexpr,
    PRODUCTION_FAST: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    IS_VARLEN: tl.constexpr = False,
):
    """
    Fused recurrent SwiLA backward kernel with temporal (rSLDS) transitions.

    Phase 1: Backward-in-time (t = T-1 to 0)
        - Section A: Q-side temporal backward
        - Section B: Accumulate b_dh (using rho)
        - Section C: Weight update gradients (unchanged)
        - Section D: KV-side temporal backward
        - Section E: Remaining gradients and per-key write local term

    Phase 2: Forward-in-time (t = 0 to T-1)
        - Reconstruct W (state)
        - Compute grad_q using rho
        - Correct grad_k using W_before
        - Accumulate per-key readout and prediction local terms

    A small follow-up kernel performs the reverse cumulative sum of those local
    terms, segmented by ``cu_seqlens``, to produce ``dlog_decay``.
    """
    tl.static_assert(
        not (RESOURCE_SAFE or PRODUCTION_FAST) or SPLIT_Q_TEMPORAL,
        "specialized temporal backward requires the split Q reverse",
    )
    # Grid indices
    T = T.to(tl.int64)
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_bt = T
        T = eos - bos
    else:
        bos = i_n * T
        all_bt = B * T

    # Masks - use BE for block size, check actual E for bounds
    e_range = tl.arange(0, BE).to(tl.int64)
    k_range = i_k * BK + tl.arange(0, BK).to(tl.int64)
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    mask_e = e_range < E
    mask_k = k_range < D_k
    mask_v = v_range < D_v
    mask_ve = mask_v[:, None] & mask_e[None, :]
    mask_vek = mask_v[:, None, None] & mask_e[None, :, None] & mask_k[None, None, :]

    # Exact forward keeps temporal carries in log space, whereas the public
    # r/rho outputs are FP32 probabilities and can round a finite carry to
    # zero. Recreate those exact carries in scratch that backward already owns:
    # du_save for log-r and (in the unsplit path) dpi_q for log(E*rho). The reverse
    # scan consumes slot t-1 before overwriting slot t, so this needs no extra
    # B*T*H*D_v*E allocation. Keeping the prepass inside the kernel also makes
    # every Triton autotune trial self-contained.
    log_E = tl.log(tl.cast(E, tl.float32))
    rp_pre_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    if USE_INITIAL_TEMPORAL_STATE:
        rp_pre_base = r_prev_init + i_nh * D_v * E
        b_r_initial_pre = tl.load(
            rp_pre_base + rp_pre_offset,
            mask=mask_ve,
            other=0.0,
        ).to(tl.float32)
        b_log_r_carry_pre = tl.log(b_r_initial_pre)
        if not SPLIT_Q_TEMPORAL:
            rhop_pre_base = rho_prev_init + i_nh * D_v * E
            b_rho_initial_pre = tl.load(
                rhop_pre_base + rp_pre_offset,
                mask=mask_ve,
                other=0.0,
            ).to(tl.float32)
            b_log_rho_carry_pre = tl.log(b_rho_initial_pre) + log_E
    else:
        b_log_uniform_pre = -tl.log(tl.cast(E, tl.float32))
        b_log_r_carry_pre = tl.full(
            [BV, BE], b_log_uniform_pre, dtype=tl.float32
        )
        b_log_r_carry_pre = tl.where(
            mask_e[None, :], b_log_r_carry_pre, float('-inf')
        )
        if not SPLIT_Q_TEMPORAL:
            b_log_rho_carry_pre = tl.zeros([BV, BE], dtype=tl.float32)
            b_log_rho_carry_pre = tl.where(
                mask_e[None, :], b_log_rho_carry_pre, float('-inf')
            )

    base_pre = (bos * H + i_h) * D_v * E + \
        (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_pi_k_pre = pi_k + base_pre
    p_u_pre = u + base_pre
    p_gate_kv_pre = gate_kv + (bos * H + i_h) * D_v + \
        i_v * BV + tl.arange(0, BV).to(tl.int64)
    p_log_r_out = du_save + base_pre
    if not SPLIT_Q_TEMPORAL:
        p_pi_q_pre = pi_q + base_pre
        p_gate_q_pre = gate_q + (bos * H + i_h) * D_v + \
            i_v * BV + tl.arange(0, BV).to(tl.int64)
        p_log_rho_out = dpi_q + base_pre

    for _ in range(T):
        b_pi_k_pre = tl.load(
            p_pi_k_pre, mask=mask_ve, other=0.0
        ).to(tl.float32)
        b_u_pre = tl.load(
            p_u_pre, mask=mask_ve, other=0.0
        ).to(tl.float32)
        b_gate_kv_pre = tl.load(
            p_gate_kv_pre, mask=mask_v, other=0.0
        ).to(tl.float32)
        b_log_g_kv_pre, b_log_1mg_kv_pre = logsigmoid_pair(
            b_gate_kv_pre
        )
        b_log_r_reset_pre = log_softmax_2d(b_pi_k_pre, mask_e)
        b_log_r_prior_pre = logaddexp(
            b_log_1mg_kv_pre[:, None] + b_log_r_carry_pre,
            b_log_g_kv_pre[:, None] + b_log_r_reset_pre,
        )
        b_log_r_carry_pre = log_softmax_2d(
            b_log_r_prior_pre - 0.5 * b_u_pre * b_u_pre,
            mask_e,
        )
        tl.store(p_log_r_out, b_log_r_carry_pre, mask=mask_ve)

        if not SPLIT_Q_TEMPORAL:
            b_pi_q_pre = tl.load(
                p_pi_q_pre, mask=mask_ve, other=0.0
            ).to(tl.float32)
            b_gate_q_pre = tl.load(
                p_gate_q_pre, mask=mask_v, other=0.0
            ).to(tl.float32)
            b_log_rho_reset_pre = log_softmax_2d(b_pi_q_pre, mask_e) + log_E
            b_log_rho_carry_pre = log_gate_mix(
                b_log_rho_carry_pre,
                b_log_rho_reset_pre,
                b_gate_q_pre[:, None],
            )
            b_log_rho_carry_pre = tl.where(
                mask_e[None, :],
                b_log_rho_carry_pre,
                float('-inf'),
            )
            tl.store(
                p_log_rho_out, b_log_rho_carry_pre, mask=mask_ve
            )
            p_pi_q_pre += H * D_v * E
            p_gate_q_pre += H * D_v
            p_log_rho_out += H * D_v * E

        p_pi_k_pre += H * D_v * E
        p_u_pre += H * D_v * E
        p_gate_kv_pre += H * D_v
        p_log_r_out += H * D_v * E

    # ===== PHASE 1: Backward-in-time =====
    # Initialize pointers at END of sequence (t = T-1)
    p_q = q + (bos * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64) + (T - 1) * H * D_k
    p_k = k + (bos * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64) + (T - 1) * H * D_k
    p_do = do + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v
    base_pi_end = ((bos + T - 1) * H + i_h) * D_v * E + (i_v * BV +
                                                         tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_pi_q = pi_q + base_pi_end
    p_pi_k = pi_k + base_pi_end
    p_beta = beta + base_pi_end
    p_r = r + base_pi_end
    p_u = u + base_pi_end
    p_rho = rho + base_pi_end
    p_qS = qS + base_pi_end
    if USE_LOG_DECAY:
        log_decay_base_end = ((bos + T - 1) * H + i_h) * D_k
        p_log_decay = log_decay + log_decay_base_end + k_range
        p_dlog_decay = dlog_decay + log_decay_base_end + k_range

    # Pointer for r_{t-1} (one step behind main pointers)
    # At t=T-1, r_{t-1} = r[T-2], so we need base_pi_end - H * D_v * E
    p_r_prev = r + base_pi_end - H * D_v * E
    p_rho_prev = rho + base_pi_end - H * D_v * E

    # Output pointers at END
    p_dv = dv + ((i_k * all_bt + bos + T - 1) * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64)
    # FIX: Simplified pointer computation after removing NV dimension from allocation
    # Shape is now [NK, B, T, H, D_v, E] - each i_v block writes to its D_v slice directly
    dbeta_base_end = (i_k * all_bt + bos + T - 1) * H * D_v * E + i_h * D_v * E
    p_dbeta = dbeta + dbeta_base_end + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_dpi_k = dpi_k + dbeta_base_end + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_dpi_q = dpi_q + dbeta_base_end + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_dk = dk + ((i_v * all_bt + bos + T - 1) * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
    # Pointer for saving du (for Phase 2 correction)
    p_du_save = du_save + base_pi_end

    # Pointers for auxiliary loss gradients (d_r_seq, d_rho_seq) - same layout as r, rho
    if USE_AUX_LOSS_GRAD:
        p_d_r_seq = d_r_seq + base_pi_end
        if not SPLIT_Q_TEMPORAL:
            p_d_rho_seq = d_rho_seq + base_pi_end
    if USE_U_AUX_GRAD:
        p_d_u_seq = d_u_seq + base_pi_end
    if USE_PI_BAR_AUX_GRAD:
        p_d_pi_bar_seq = d_pi_bar_seq + base_pi_end

    # Initialize accumulated state gradient (grad_W in Python code) - use BE
    b_dh = tl.zeros([BV, BE, BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * D_v * E * D_k + \
            (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
            e_range[None, :, None] * D_k + \
            (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
        b_dht = tl.load(p_dht, mask=mask_vek, other=0.0).to(tl.float32)
        b_dh += b_dht
        if USE_LOG_DECAY:
            p_ht = ht + i_nh * D_v * E * D_k + \
                (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
                e_range[None, :, None] * D_k + \
                (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
            b_ht = tl.load(p_ht, mask=mask_vek, other=0.0).to(tl.float32)
            b_tail_e = tl.sum(b_dht * b_ht, axis=0)
            b_tail = tl.sum(b_tail_e, axis=0)
            tl.atomic_add(p_dlog_decay, b_tail, mask=mask_k)

    # Legacy carries a probability-coordinate adjoint. Exact mode carries the
    # adjoint of the internal log responsibility so a finite log carry that
    # rounded to a zero public probability never forms 0 * inf at the previous
    # softmax.
    b_d_r_temporal = tl.zeros([BV, BE], dtype=tl.float32)
    b_d_log_r_temporal = tl.zeros([BV, BE], dtype=tl.float32)
    b_d_r_final = tl.zeros([BV, BE], dtype=tl.float32)
    final_temporal_offset = i_nh * D_v * E + \
        (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    if USE_FINAL_R_GRADIENT:
        b_d_r_final = tl.load(
            d_r_prev_final + final_temporal_offset,
            mask=mask_ve,
            other=0.0,
        ).to(tl.float32)
    if not SPLIT_Q_TEMPORAL:
        b_d_rho_temporal = tl.zeros([BV, BE], dtype=tl.float32)
        b_d_log_rho_temporal = tl.zeros([BV, BE], dtype=tl.float32)
        b_d_rho_final = tl.zeros([BV, BE], dtype=tl.float32)
        if USE_FINAL_RHO_GRADIENT:
            b_d_rho_final = tl.load(
                d_rho_prev_final + final_temporal_offset,
                mask=mask_ve,
                other=0.0,
            ).to(tl.float32)

    # Gate pointers and precomputed transition matrices for reset_gate backward
    # Pointers at END of sequence (t = T-1)
    p_gate_kv = gate_kv + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v
    p_gate_q = gate_q + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v
    p_d_gate_kv = d_gate_kv + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v
    p_d_gate_q = d_gate_q + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v

    # Counter for handling t=0 boundary
    remaining = T

    # Offset for loading initial temporal states
    rp_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]

    # Backward loop: compute gradients using saved intermediates
    for _ in range(T):
        remaining -= 1

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_pi_k = tl.load(p_pi_k, mask=mask_ve, other=0).to(tl.float32)
        b_beta = tl.load(p_beta, mask=mask_ve, other=0).to(tl.float32)
        if not SPLIT_Q_TEMPORAL:
            b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0).to(tl.float32)

        # Load saved intermediates
        b_r = tl.load(p_r, mask=mask_ve, other=0).to(tl.float32)          # [BV, E]
        b_u = tl.load(p_u, mask=mask_ve, other=0).to(tl.float32)          # [BV, E]
        b_rho_val = tl.load(p_rho, mask=mask_ve, other=0).to(tl.float32)        # [BV, E]
        if not SPLIT_Q_TEMPORAL:
            b_qS_val = tl.load(p_qS, mask=mask_ve, other=0).to(tl.float32)      # [BV, E]
        if USE_LOG_DECAY:
            b_log_decay = tl.load(p_log_decay, mask=mask_k, other=0.0).to(tl.float32)
            b_decay = tl.exp(b_log_decay)[None, None, :]

        # Load r_{t-1} and rho_{t-1}
        if remaining > 0:
            b_log_r_prev_exact = tl.load(
                p_du_save - H * D_v * E,
                mask=mask_ve,
                other=float('-inf'),
            ).to(tl.float32)
            b_r_prev = tl.exp(b_log_r_prev_exact)
            if not SPLIT_Q_TEMPORAL:
                b_log_rho_prev_exact = tl.load(
                    p_dpi_q - H * D_v * E,
                    mask=mask_ve,
                    other=float('-inf'),
                ).to(tl.float32)
                b_rho_prev = tl.exp(b_log_rho_prev_exact - log_E)
        else:
            # At t=0, use initial state
            if USE_INITIAL_TEMPORAL_STATE:
                rp_base = r_prev_init + i_nh * D_v * E
                b_r_prev = tl.load(rp_base + rp_offset, mask=mask_ve, other=0).to(tl.float32)
                if not SPLIT_Q_TEMPORAL:
                    rhop_base = rho_prev_init + i_nh * D_v * E
                    b_rho_prev = tl.load(rhop_base + rp_offset, mask=mask_ve, other=0).to(tl.float32)
            else:
                # Use BE for block size, but initialize with 1/E (uniform over actual experts)
                b_r_prev = tl.full([BV, BE], 1.0 / E, dtype=tl.float32)
                # Zero out elements beyond E
                b_r_prev = tl.where(mask_e[None, :], b_r_prev, 0.0)
                if not SPLIT_Q_TEMPORAL:
                    b_rho_prev = tl.full([BV, BE], 1.0 / E, dtype=tl.float32)
                    b_rho_prev = tl.where(mask_e[None, :], b_rho_prev, 0.0)
            b_log_r_prev_exact = tl.log(b_r_prev)
            if not SPLIT_Q_TEMPORAL:
                if USE_INITIAL_TEMPORAL_STATE:
                    b_log_rho_prev_exact = tl.log(b_rho_prev) + log_E
                else:
                    b_log_rho_prev_exact = tl.zeros([BV, BE], dtype=tl.float32)
                    b_log_rho_prev_exact = tl.where(
                        mask_e[None, :], b_log_rho_prev_exact, float('-inf')
                    )

        # ═══════════════════════════════════════════════════════
        # SECTION A: Q-SIDE TEMPORAL BACKWARD
        # ═══════════════════════════════════════════════════════

        if not SPLIT_Q_TEMPORAL:
            b_d_rho = b_do[:, None] * b_qS_val

            # Add auxiliary loss gradient (load balancing loss w.r.t. rho)
            if USE_AUX_LOSS_GRAD:
                b_d_rho_aux = tl.load(p_d_rho_seq, mask=mask_ve, other=0).to(tl.float32)
                b_d_rho = b_d_rho + b_d_rho_aux

            if USE_FINAL_RHO_GRADIENT and remaining == T - 1:
                b_d_rho += b_d_rho_final

        if not SPLIT_Q_TEMPORAL:
            b_gate_q_raw = tl.load(
                p_gate_q, mask=mask_v, other=0
            ).to(tl.float32)
            # Use the same Q-only gate normalizer as log_gate_mix.
            b_log_gate_normalizer_q = libdevice.log1p(tl.exp(-tl.abs(b_gate_q_raw)))
            b_log_g_q = -tl.maximum(-b_gate_q_raw, 0.0) - b_log_gate_normalizer_q
            b_log_1mg_q = -tl.maximum(b_gate_q_raw, 0.0) - b_log_gate_normalizer_q
            b_g_q = tl.exp(b_log_g_q)
            b_1mg_q = tl.exp(b_log_1mg_q)
            b_log_rho_reset = log_softmax_2d(b_pi_q, mask_e)
            b_rho_reset = tl.exp(b_log_rho_reset)
            b_log_rho_reset_centered = b_log_rho_reset + log_E
            b_log_rho_prior_centered = log_gate_mix(
                b_log_rho_prev_exact,
                b_log_rho_reset_centered,
                b_gate_q_raw[:, None],
            )
            # Keep branch ratios centered, but the public probability-
            # coordinate initial-prior derivative uses log(rho), not log(E*rho).
            b_log_rho_prior = b_log_rho_prior_centered - log_E
            b_reset_weight_q = tl.exp(
                b_log_g_q[:, None]
                + b_log_rho_reset_centered
                - b_log_rho_prior_centered
            )
            b_transition_weight_q = tl.exp(
                b_log_1mg_q[:, None]
                + b_log_rho_prev_exact
                - b_log_rho_prior_centered
            )
            b_reset_weight_q = tl.where(
                mask_ve, b_reset_weight_q, 0.0
            )
            b_transition_weight_q = tl.where(
                mask_ve, b_transition_weight_q, 0.0
            )

            b_d_log_reset_q = (
                b_d_log_rho_temporal * b_reset_weight_q
                + b_d_rho * b_g_q[:, None] * b_rho_reset
            )
            b_d_log_reset_q = tl.where(
                mask_ve, b_d_log_reset_q, 0.0
            )
            b_dpi_q = b_d_log_reset_q - b_rho_reset * tl.sum(
                tl.trans(b_d_log_reset_q), axis=0
            )[:, None]

            b_d_gate_q_e = (
                b_d_log_rho_temporal
                * (
                    b_reset_weight_q * b_1mg_q[:, None]
                    - b_transition_weight_q * b_g_q[:, None]
                )
                + b_d_rho
                * (b_rho_reset - b_rho_prev)
                * b_g_q[:, None]
                * b_1mg_q[:, None]
            )
            b_d_gate_q_e = tl.where(
                mask_ve, b_d_gate_q_e, 0.0
            )
            b_d_gate_q_raw = tl.sum(
                tl.trans(b_d_gate_q_e), axis=0
            )
            tl.store(
                p_d_gate_q,
                b_d_gate_q_raw.to(p_d_gate_q.dtype.element_ty),
                mask=mask_v,
            )

            if remaining > 0:
                b_d_log_rho_temporal = (
                    b_d_log_rho_temporal * b_transition_weight_q
                    + b_d_rho * b_1mg_q[:, None] * b_rho_prev
                )
            else:
                b_abs_q = tl.abs(b_d_log_rho_temporal)
                b_log_abs_q = tl.log(
                    tl.where(b_abs_q > 0.0, b_abs_q, 1.0)
                )
                b_abs_initial_q = tl.exp(
                    b_log_abs_q
                    + b_log_1mg_q[:, None]
                    - b_log_rho_prior
                )
                b_d_rho_temporal = tl.where(
                    b_d_log_rho_temporal == 0.0,
                    0.0,
                    tl.where(
                        b_d_log_rho_temporal < 0.0,
                        -b_abs_initial_q,
                        b_abs_initial_q,
                    ),
                )
                b_d_rho_temporal += (
                    b_d_rho * b_1mg_q[:, None]
                )
                b_d_rho_temporal = tl.where(
                    mask_ve, b_d_rho_temporal, 0.0
                )
        # ═══════════════════════════════════════════════════════
        # SECTION B: ACCUMULATE b_dh (using rho instead of pi_q_star)
        # ═══════════════════════════════════════════════════════

        # Accumulate dh
        b_do_weighted = b_do[:, None] * b_rho_val  # [BV, BE]
        b_dh += (b_q * scale)[None, None, :] * b_do_weighted[:, :, None]

        if DEBUG_INTERMEDIATES:
            # Capture [B,T,H,D_v,E,D_k] using the global token offset, including packed sequence starts.
            debug_dh_base = debug_dh + ((bos + remaining) * H + i_h) * D_v * E * D_k
            debug_dh_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
                e_range[None, :, None] * D_k + \
                (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
            tl.store(debug_dh_base + debug_dh_offset, b_dh.to(debug_dh.dtype.element_ty), mask=mask_vek)

        # ═══════════════════════════════════════════════════════
        # SECTION C: WEIGHT UPDATE GRADIENTS (SAME as non-temporal)
        # ═══════════════════════════════════════════════════════

        # ∂L/∂coef where coef = beta * r * u
        b_d_coef = dot_3d_axis2(b_dh, b_k)  # [BV, E]

        # Gradients through coef = beta * r * u
        b_dbeta = b_d_coef * b_r * b_u
        b_d_r = b_d_coef * b_beta * b_u
        b_d_u = b_d_coef * b_beta * b_r

        # After Section C: b_d_r = non-temporal part of B-16 (Claims 9+10)

        # ═══════════════════════════════════════════════════════
        # SECTION D: KV-SIDE TEMPORAL BACKWARD
        # ═══════════════════════════════════════════════════════

        # Add auxiliary loss gradient (load balancing loss w.r.t. r)
        if USE_AUX_LOSS_GRAD:
            b_d_r_aux = tl.load(p_d_r_seq, mask=mask_ve, other=0).to(tl.float32)
            b_d_r = b_d_r + b_d_r_aux

        # Final r is a probability output, so its VJP joins the local
        # probability-coordinate terms. Future recurrence dependence is on
        # log-r and must pass through log-softmax directly.
        if USE_FINAL_R_GRADIENT and remaining == T - 1:
            b_d_r += b_d_r_final
        b_d_x = softmax_backward_2d(b_r, b_d_r)
        b_sum_d_log_r_temporal = tl.sum(
            tl.trans(b_d_log_r_temporal), axis=0
        )[:, None]
        b_d_x += (
            b_d_log_r_temporal
            - b_r * b_sum_d_log_r_temporal
        )

        # The posterior contribution arrives as d_x = dL/dlog(r_prior).  In
        # exact reset-gate mode, differentiate the two log branches directly:
        #
        #   r_prior = (1-g) * r_transition + g * r_reset
        #   w_transition = (1-g) * r_transition / r_prior
        #   w_reset      = g * r_reset / r_prior
        #
        # Router and gate VJPs then use bounded branch weights rather than an
        # unscaled dL/dr_prior.  This matters when a tiny reset probability is
        # exactly offset by its likelihood: dL/dr_prior can exceed FP32 while
        # the gate/router derivatives remain ordinary finite values.  The
        # probability-space pi_bar auxiliary VJP is kept separate below.
        b_gate_kv_exact = tl.load(
            p_gate_kv,
            mask=mask_v,
            other=0,
        ).to(tl.float32)
        b_log_g_kv_exact, b_log_1mg_kv_exact = logsigmoid_pair(
            b_gate_kv_exact
        )
        b_g_kv_exact = tl.exp(b_log_g_kv_exact)
        b_1mg_kv_exact = tl.exp(b_log_1mg_kv_exact)
        b_log_r_reset_exact = log_softmax_2d(b_pi_k, mask_e)
        b_r_reset_exact = tl.exp(b_log_r_reset_exact)
        b_log_r_prior_exact = logaddexp(
            b_log_1mg_kv_exact[:, None] + b_log_r_prev_exact,
            b_log_g_kv_exact[:, None] + b_log_r_reset_exact,
        )
        b_reset_branch_weight = tl.exp(
            b_log_g_kv_exact[:, None]
            + b_log_r_reset_exact
            - b_log_r_prior_exact
        )
        b_transition_branch_weight = tl.exp(
            b_log_1mg_kv_exact[:, None]
            + b_log_r_prev_exact
            - b_log_r_prior_exact
        )
        b_reset_branch_weight = tl.where(
            mask_ve, b_reset_branch_weight, 0.0
        )
        b_transition_branch_weight = tl.where(
            mask_ve, b_transition_branch_weight, 0.0
        )

        # dL/dlog(r_reset) passes through log-softmax without ever forming
        # dL/dr_prior.  This is bounded by d_x because the branch weight is
        # in [0, 1].
        b_d_log_r_reset_posterior = (
            b_d_x * b_reset_branch_weight
        )
        b_d_log_r_reset_posterior = tl.where(
            mask_ve, b_d_log_r_reset_posterior, 0.0
        )

        b_d_log_r_transition_posterior = (
            b_d_x * b_transition_branch_weight
        )

        b_d_gate_kv_posterior_e = b_d_x * (
            b_reset_branch_weight * b_1mg_kv_exact[:, None]
            - b_transition_branch_weight * b_g_kv_exact[:, None]
        )
        b_d_gate_kv_posterior_e = tl.where(
            mask_ve, b_d_gate_kv_posterior_e, 0.0
        )
        b_d_gate_kv_posterior = tl.sum(
            tl.trans(b_d_gate_kv_posterior_e),
            axis=0,
        )

        if USE_PI_BAR_AUX_GRAD:
            b_d_pi_bar_aux = tl.load(
                p_d_pi_bar_seq,
                mask=mask_ve,
                other=0.0,
            ).to(tl.float32)
        else:
            b_d_pi_bar_aux = tl.zeros([BV, BE], dtype=tl.float32)

        # Gradient through -0.5*u^2
        b_d_u += b_d_x * (-b_u)
        if USE_U_AUX_GRAD:
            b_d_u += tl.load(
                p_d_u_seq,
                mask=mask_ve,
                other=0.0,
            ).to(tl.float32)

        # Posterior and direct pi_bar-output contributions are kept
        # separate until after each branch's Jacobian.  In particular,
        # never form ``d_pi_bar * g`` after d_pi_bar has overflowed:
        # dlog(r_reset) and dgate were already obtained from bounded
        # log-branch responsibilities above.
        b_r_transition = b_r_prev
        b_r_reset = b_r_reset_exact
        b_d_log_r_reset = (
            b_d_log_r_reset_posterior
            + b_d_pi_bar_aux
            * b_g_kv_exact[:, None]
            * b_r_reset
        )

        b_d_gate_kv_aux_e = (
            b_d_pi_bar_aux
            * (b_r_reset - b_r_transition)
            * b_g_kv_exact[:, None]
            * b_1mg_kv_exact[:, None]
        )
        b_d_gate_kv_aux_e = tl.where(
            mask_ve, b_d_gate_kv_aux_e, 0.0
        )
        b_d_gate_kv_raw = b_d_gate_kv_posterior + tl.sum(
            tl.trans(b_d_gate_kv_aux_e), axis=0
        )
        tl.store(
            p_d_gate_kv,
            b_d_gate_kv_raw.to(p_d_gate_kv.dtype.element_ty),
            mask=mask_v,
        )

        b_d_log_r_reset = tl.where(
            mask_ve, b_d_log_r_reset, 0.0
        )
        b_sum_d_log_r_reset = tl.sum(
            tl.trans(b_d_log_r_reset), axis=0
        )[:, None]
        b_dpi_k = (
            b_d_log_r_reset
            - b_r_reset * b_sum_d_log_r_reset
        )

        if remaining > 0:
            # Internal recurrence consumes log-r directly. The direct
            # pi_bar-output term contributes through exp(log-r).
            b_d_log_r_temporal = (
                b_d_log_r_transition_posterior
                + b_d_pi_bar_aux
                * b_1mg_kv_exact[:, None]
                * b_r_transition
            )
        else:
            # Only the caller-supplied initial prior is a probability
            # coordinate. Convert at this boundary in signed log space;
            # an actual exact-zero derivative may legitimately be inf.
            b_abs_d_x = tl.abs(b_d_x)
            b_log_abs_d_x = tl.log(
                tl.where(b_abs_d_x > 0.0, b_abs_d_x, 1.0)
            )
            b_abs_initial = tl.exp(
                b_log_abs_d_x
                + b_log_1mg_kv_exact[:, None]
                - b_log_r_prior_exact
            )
            b_d_r_temporal = tl.where(
                b_d_x == 0.0,
                0.0,
                tl.where(
                    b_d_x < 0.0,
                    -b_abs_initial,
                    b_abs_initial,
                ),
            )
            b_d_r_temporal += (
                b_d_pi_bar_aux * b_1mg_kv_exact[:, None]
            )
            b_d_r_temporal = tl.where(
                mask_ve, b_d_r_temporal, 0.0
            )

        # ═══════════════════════════════════════════════════════
        # SECTION E: REMAINING GRADIENTS (SAME as non-temporal)
        # ═══════════════════════════════════════════════════════

        # Gradient through u = v - pred
        # WORKAROUND: Use transpose + axis=0 to avoid Triton bug with axis=1
        b_d_u_t = tl.trans(b_d_u)  # [BV, BE] -> [BE, BV]
        b_dv = tl.sum(b_d_u_t, axis=0)  # [BE, BV] -> [BV]

        # Partial dk from state update: dk = sum_{v,e}(b_dh * coef)
        if REUSE_DK_COEF:
            b_coef_dk = b_beta * b_r * b_u
        else:
            # Reload coef components fresh from memory to preserve the original
            # large-kernel compiler workaround.
            b_beta_dk = tl.load(p_beta, mask=mask_ve, other=0).to(tl.float32)
            b_r_dk = tl.load(p_r, mask=mask_ve, other=0).to(tl.float32)
            b_u_dk = tl.load(p_u, mask=mask_ve, other=0).to(tl.float32)
            b_coef_dk = b_beta_dk * b_r_dk * b_u_dk
        b_dh_coef = b_dh * b_coef_dk[:, :, None]  # [BV, BE, BK]
        b_dh_coef = tl.where(mask_vek, b_dh_coef, 0.0)
        # Triton 3.2 mislowers the expert reduction of a sliced V/E layout
        # at these small tiles (including masked E3). Flatten V/E for the
        # total, and reuse it for key-only decay instead of a second sliced
        # reduction. See docs/temporal_reduction_diagnosis.md.
        if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
            b_dk = tl.sum(tl.reshape(b_dh_coef, (BV * BE, BK)), axis=0)
            if USE_LOG_DECAY:
                b_update_local = b_k * b_dk
        else:
            b_dk_2d = tl.sum(b_dh_coef, axis=0)  # [BV, BE, BK] -> [BE, BK]
            b_dk = tl.sum(b_dk_2d, axis=0)  # [BE, BK] -> [BK]
            if USE_LOG_DECAY:
                b_update_local_e = b_k[None, :] * b_dk_2d
                b_update_local = tl.sum(b_update_local_e, axis=0)
        if USE_LOG_DECAY:
            # Coordinatewise generalized-sigma local term from the write:
            # -k * sum_{v,e}(G * c), where c = beta * r * u.
            tl.atomic_add(p_dlog_decay, -b_update_local, mask=mask_k)

        # Store gradients computed in Phase 1
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), mask=mask_ve)
        tl.store(p_dpi_k, b_dpi_k.to(p_dpi_k.dtype.element_ty), mask=mask_ve)
        if not SPLIT_Q_TEMPORAL:
            tl.store(p_dpi_q, b_dpi_q.to(p_dpi_q.dtype.element_ty), mask=mask_ve)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        # Save b_d_u for Phase 2 grad_k correction
        tl.store(p_du_save, b_d_u.to(p_du_save.dtype.element_ty), mask=mask_ve)

        # Post-decay: dSprev = D * (G - du k^T).
        b_dh -= b_k[None, None, :] * b_d_u[:, :, None]
        if USE_LOG_DECAY:
            b_dh *= b_decay

        # Move pointers backward
        p_q -= H * D_k
        p_k -= H * D_k
        p_do -= H * D_v
        if not SPLIT_Q_TEMPORAL:
            p_pi_q -= H * D_v * E
        p_pi_k -= H * D_v * E
        p_beta -= H * D_v * E
        p_r -= H * D_v * E
        p_u -= H * D_v * E
        p_rho -= H * D_v * E
        if not SPLIT_Q_TEMPORAL:
            p_qS -= H * D_v * E
        p_dv -= H * D_v
        p_dbeta -= H * D_v * E
        p_dpi_k -= H * D_v * E
        if not SPLIT_Q_TEMPORAL:
            p_dpi_q -= H * D_v * E
        p_dk -= H * D_k
        p_du_save -= H * D_v * E
        if USE_AUX_LOSS_GRAD:
            p_d_r_seq -= H * D_v * E
            if not SPLIT_Q_TEMPORAL:
                p_d_rho_seq -= H * D_v * E
        if USE_U_AUX_GRAD:
            p_d_u_seq -= H * D_v * E
        if USE_PI_BAR_AUX_GRAD:
            p_d_pi_bar_seq -= H * D_v * E
        p_gate_kv -= H * D_v
        p_d_gate_kv -= H * D_v
        if not SPLIT_Q_TEMPORAL:
            p_gate_q -= H * D_v
            p_d_gate_q -= H * D_v
        if remaining > 0:
            p_r_prev -= H * D_v * E
            if not SPLIT_Q_TEMPORAL:
                p_rho_prev -= H * D_v * E
        if USE_LOG_DECAY:
            p_log_decay -= H * D_k
            p_dlog_decay -= H * D_k

    # Store initial state gradient (after phase 1, before barrier)
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * D_v * E * D_k + \
            (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
            e_range[None, :, None] * D_k + \
            (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
        tl.store(p_dh0, b_dh.to(dh0.dtype.element_ty), mask=mask_vek)

    # FIX: Store initial temporal state gradients (dr_prev_init, drho_prev_init)
    # These are the gradients that would flow to r_prev_{-1} and rho_prev_{-1}
    # after processing t=0 (which is the last iteration of Phase 1)
    if STORE_TEMPORAL_STATE_GRADIENT:
        dr_offset = i_nh * D_v * E + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
        tl.store(dr_prev_init + dr_offset, b_d_r_temporal.to(dr_prev_init.dtype.element_ty), mask=mask_ve)
        if not SPLIT_Q_TEMPORAL:
            tl.store(drho_prev_init + dr_offset, b_d_rho_temporal.to(drho_prev_init.dtype.element_ty), mask=mask_ve)

    if not SKIP_REPLAY:
        tl.debug_barrier()

        # ===== PHASE 2: Forward-in-time to compute dq and correct dk =====
        b_h = tl.zeros([BV, BE, BK], dtype=tl.float32)

        # Reset pointers to START of sequence
        p_q = q + (bos * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
        p_k = k + (bos * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
        p_do = do + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64)
        base_pi_start = (bos * H + i_h) * D_v * E + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
        p_beta = beta + base_pi_start
        p_r = r + base_pi_start
        p_u = u + base_pi_start
        p_rho = rho + base_pi_start

        p_dq = dq + ((i_v * all_bt + bos) * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
        p_dk = dk + ((i_v * all_bt + bos) * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
        # Pointer for loading saved du
        p_du_save = du_save + base_pi_start
        if USE_LOG_DECAY:
            log_decay_base_start = (bos * H + i_h) * D_k
            p_log_decay_fwd = log_decay + log_decay_base_start + k_range
            p_dlog_decay_fwd = dlog_decay + log_decay_base_start + k_range

        # Load initial state if provided
        if USE_INITIAL_STATE:
            p_h0 = h0 + i_nh * D_v * E * D_k + \
                (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
                e_range[None, :, None] * D_k + \
                (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
            b_h += tl.load(p_h0, mask=mask_vek, other=0.0).to(tl.float32)

        # Forward loop: reconstruct h and compute dq, correct dk
        elapsed = 0
        for _ in range(T):
            # Load inputs
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
            if USE_LOG_DECAY:
                b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)

            # Load saved intermediates
            b_beta = tl.load(p_beta, mask=mask_ve, other=0).to(tl.float32)
            b_r = tl.load(p_r, mask=mask_ve, other=0).to(tl.float32)
            b_u = tl.load(p_u, mask=mask_ve, other=0).to(tl.float32)
            b_rho_val = tl.load(p_rho, mask=mask_ve, other=0).to(tl.float32)

            # Load saved grad_u from Phase 1
            b_d_u = tl.load(p_du_save, mask=mask_ve, other=0).to(tl.float32)

            # Reconstruct the decayed state used by prediction.
            if USE_LOG_DECAY:
                b_log_decay = tl.load(
                    p_log_decay_fwd,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                b_h *= tl.exp(b_log_decay[None, None, :])

            # Correct dk using the state that formed the prediction.
            b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
            b_dk_weighted = b_d_u[:, :, None] * b_h  # [BV, E, BK]
            b_dk_weighted = tl.where(mask_vek, b_dk_weighted, 0.0)
            if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
                b_dk_correction = tl.sum(
                    tl.reshape(b_dk_weighted, (BV * BE, BK)), axis=0,
                )
            else:
                b_dk_intermediate = tl.sum(b_dk_weighted, axis=0)  # [E, BK]
                b_dk_correction = tl.sum(b_dk_intermediate, axis=0)  # [BK]
            b_dk = b_dk - b_dk_correction
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

            # Update state: h_after = decay(h_before) + coef * k.
            b_coef = b_beta * b_r * b_u
            b_h = state_update(b_h, b_coef, b_k)

            # dq computation using rho (instead of pi_q_star)
            b_d_qS = b_do[:, None] * b_rho_val  # [BV, E]

            # dq = sum_{v,e}(d_qS * h_after) * scale
            b_weighted_h = b_d_qS[:, :, None] * b_h  # [BV, E, BK]
            # FIX: Mask before sum to prevent NaN from unaligned dimensions (BE > E or BV > D_v)
            b_weighted_h = tl.where(mask_vek, b_weighted_h, 0.0)
            if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
                b_readout_total = tl.sum(
                    tl.reshape(b_weighted_h, (BV * BE, BK)), axis=0,
                )
                b_dq = b_readout_total * scale
            else:
                b_intermediate = tl.sum(b_weighted_h, axis=0)  # [E, BK]
                b_dq = tl.sum(b_intermediate, axis=0) * scale  # [BK]
            if USE_LOG_DECAY:
                # Output and prediction local terms belong to this timestep.
                if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
                    b_output_local = (b_q * scale) * b_readout_total
                    b_prediction_local = -b_k * b_dk_correction
                else:
                    b_output_local_e = (b_q * scale)[None, :] * b_intermediate
                    b_prediction_local_e = -b_k[None, :] * b_dk_intermediate
                    b_output_local = tl.sum(b_output_local_e, axis=0)
                    b_prediction_local = tl.sum(
                        b_prediction_local_e,
                        axis=0,
                    )
                tl.atomic_add(
                    p_dlog_decay_fwd,
                    b_output_local + b_prediction_local,
                    mask=mask_k,
                )

            # Store dq
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

            # Move pointers forward
            if USE_LOG_DECAY:
                p_q += H * D_k
            p_k += H * D_k
            p_do += H * D_v
            p_beta += H * D_v * E
            p_r += H * D_v * E
            p_u += H * D_v * E
            p_rho += H * D_v * E
            p_dq += H * D_k
            p_dk += H * D_k
            p_du_save += H * D_v * E
            if USE_LOG_DECAY:
                p_log_decay_fwd += H * D_k
                p_dlog_decay_fwd += H * D_k
            elapsed += 1  # noqa: SIM113 - Triton does not support enumerate
