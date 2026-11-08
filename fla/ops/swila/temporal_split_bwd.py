# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Split temporal query reverse and state replay kernels."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from fla.ops.swila.atoms import log_softmax_2d, state_update
from fla.ops.swila.atoms.activations import log_gate_mix


@triton.jit(do_not_specialize=['T'])
def fused_recurrent_swila_temporal_bwd_q_temporal_kernel(
    pi_q,           # [B, T, H, D_v, E]
    rho,            # [B, T, H, D_v, E]
    qS,             # [B, T, H, D_v, E]
    rho_prev_init,  # [B, H, D_v, E] initial rho_prev (optional)
    do,             # [B, T, H, D_v]
    d_rho_prev_final,  # [B, H, D_v, E] final rho-state gradient (optional)
    dpi_q,          # [B, T, H, D_v, E]
    drho_prev_init,  # [B, H, D_v, E] gradient w.r.t. initial rho_prev (optional)
    d_rho_seq,      # [B, T, H, D_v, E] gradient from aux loss w.r.t. rho (optional)
    gate_q,         # [B, T, H, D_v] raw reset gate logits Q-side (optional)
    d_gate_q,       # [B, T, H, D_v] gate gradient output Q-side
    cu_seqlens,     # [N+1] cumulative sequence lengths (optional, for variable-length)
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_v: tl.constexpr,
    E: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    USE_INITIAL_TEMPORAL_STATE: tl.constexpr,
    STORE_TEMPORAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_RHO_GRADIENT: tl.constexpr,
    USE_AUX_LOSS_GRAD: tl.constexpr,
    IS_VARLEN: tl.constexpr = False,
):
    """Q-side temporal reverse pass split out of the full temporal backward."""
    T = T.to(tl.int64)
    i_v, _, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    e_range = tl.arange(0, BE).to(tl.int64)
    mask_e = e_range < E
    mask_v = i_v * BV + tl.arange(0, BV).to(tl.int64) < D_v
    mask_ve = mask_v[:, None] & mask_e[None, :]

    # Exact forward carries log(E*rho). The public rho sequence is a
    # probability output and may therefore round a finite log carry to zero.
    # Recreate that centered log carry in dpi_q before the reverse pass; reverse
    # order makes this scratch reuse safe because slot t is overwritten only
    # after slot t-1 has been consumed.
    log_E = tl.log(tl.cast(E, tl.float32))
    rp_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    if USE_INITIAL_TEMPORAL_STATE:
        rhop_base = rho_prev_init + i_nh * D_v * E
        b_rho_initial = tl.load(
            rhop_base + rp_offset,
            mask=mask_ve,
            other=0.0,
        ).to(tl.float32)
        b_log_rho_carry = tl.log(b_rho_initial) + log_E
    else:
        b_log_rho_carry = tl.zeros([BV, BE], dtype=tl.float32)
        b_log_rho_carry = tl.where(
            mask_e[None, :], b_log_rho_carry, float('-inf')
        )

    base_pi_start = (bos * H + i_h) * D_v * E + \
        (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_pi_q_pre = pi_q + base_pi_start
    p_gate_q_pre = gate_q + (bos * H + i_h) * D_v + \
        i_v * BV + tl.arange(0, BV).to(tl.int64)
    p_log_rho_out = dpi_q + base_pi_start
    for _ in range(T):
        b_pi_q_pre = tl.load(
            p_pi_q_pre, mask=mask_ve, other=0.0
        ).to(tl.float32)
        b_gate_q_pre = tl.load(
            p_gate_q_pre, mask=mask_v, other=0.0
        ).to(tl.float32)
        b_log_rho_reset_pre = log_softmax_2d(b_pi_q_pre, mask_e) + log_E
        b_log_rho_carry = log_gate_mix(
            b_log_rho_carry,
            b_log_rho_reset_pre,
            b_gate_q_pre[:, None],
        )
        b_log_rho_carry = tl.where(
            mask_e[None, :], b_log_rho_carry, float('-inf')
        )
        tl.store(
            p_log_rho_out,
            b_log_rho_carry,
            mask=mask_ve,
        )
        p_pi_q_pre += H * D_v * E
        p_gate_q_pre += H * D_v
        p_log_rho_out += H * D_v * E

    base_pi_end = ((bos + T - 1) * H + i_h) * D_v * E + \
                  (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_do = do + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v
    p_pi_q = pi_q + base_pi_end
    p_qS = qS + base_pi_end
    p_dpi_q = dpi_q + base_pi_end

    if USE_AUX_LOSS_GRAD:
        p_d_rho_seq = d_rho_seq + base_pi_end

    p_gate_q = gate_q + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v
    p_d_gate_q = d_gate_q + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64) + (T - 1) * H * D_v

    remaining = T
    rp_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    b_d_rho_temporal = tl.zeros([BV, BE], dtype=tl.float32)
    b_d_log_rho_temporal = tl.zeros([BV, BE], dtype=tl.float32)
    b_d_rho_final = tl.zeros([BV, BE], dtype=tl.float32)
    if USE_FINAL_RHO_GRADIENT:
        final_offset = i_nh * D_v * E + rp_offset
        b_d_rho_final = tl.load(
            d_rho_prev_final + final_offset,
            mask=mask_ve,
            other=0.0,
        ).to(tl.float32)

    for _ in range(T):
        remaining -= 1

        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0).to(tl.float32)
        b_qS_val = tl.load(p_qS, mask=mask_ve, other=0).to(tl.float32)

        if remaining > 0:
            b_log_rho_prev_exact = tl.load(
                p_dpi_q - H * D_v * E,
                mask=mask_ve,
                other=float('-inf'),
            ).to(tl.float32)
            b_rho_prev = tl.exp(b_log_rho_prev_exact - log_E)
        else:
            if USE_INITIAL_TEMPORAL_STATE:
                rhop_base = rho_prev_init + i_nh * D_v * E
                b_rho_prev = tl.load(rhop_base + rp_offset, mask=mask_ve, other=0).to(tl.float32)
            else:
                b_rho_prev = tl.full([BV, BE], 1.0 / E, dtype=tl.float32)
                b_rho_prev = tl.where(mask_e[None, :], b_rho_prev, 0.0)
            if USE_INITIAL_TEMPORAL_STATE:
                b_log_rho_prev_exact = tl.log(b_rho_prev) + log_E
            else:
                b_log_rho_prev_exact = tl.zeros([BV, BE], dtype=tl.float32)
                b_log_rho_prev_exact = tl.where(
                    mask_e[None, :], b_log_rho_prev_exact, float('-inf')
                )

        b_d_rho = b_do[:, None] * b_qS_val
        if USE_AUX_LOSS_GRAD:
            b_d_rho_aux = tl.load(p_d_rho_seq, mask=mask_ve, other=0).to(tl.float32)
            b_d_rho = b_d_rho + b_d_rho_aux

        if USE_FINAL_RHO_GRADIENT and remaining == T - 1:
            b_d_rho += b_d_rho_final

        b_gate_q_raw = tl.load(p_gate_q, mask=mask_v, other=0).to(tl.float32)
        # Match the Q mixture's small softplus correction without
        # changing the historical KV logsigmoid helper.
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
        # Branch ratios cancel the shared offset. The initial-prior
        # probability VJP below needs the uncentered log denominator.
        b_log_rho_prior = b_log_rho_prior_centered - log_E
        b_reset_weight = tl.exp(
            b_log_g_q[:, None]
            + b_log_rho_reset_centered
            - b_log_rho_prior_centered
        )
        b_transition_weight = tl.exp(
            b_log_1mg_q[:, None]
            + b_log_rho_prev_exact
            - b_log_rho_prior_centered
        )
        b_reset_weight = tl.where(mask_ve, b_reset_weight, 0.0)
        b_transition_weight = tl.where(
            mask_ve, b_transition_weight, 0.0
        )

        # Keep probability-output and future log-carry VJPs separate.
        # The affine probability output has a finite initial-prior VJP
        # even when exp(log-rho) rounded that coordinate to zero.
        b_d_log_reset = (
            b_d_log_rho_temporal * b_reset_weight
            + b_d_rho * b_g_q[:, None] * b_rho_reset
        )
        b_d_log_reset = tl.where(mask_ve, b_d_log_reset, 0.0)
        b_dpi_q = b_d_log_reset - b_rho_reset * tl.sum(
            tl.trans(b_d_log_reset), axis=0
        )[:, None]

        b_d_gate_q_e = (
            b_d_log_rho_temporal
            * (
                b_reset_weight * b_1mg_q[:, None]
                - b_transition_weight * b_g_q[:, None]
            )
            + b_d_rho
            * (b_rho_reset - b_rho_prev)
            * b_g_q[:, None]
            * b_1mg_q[:, None]
        )
        b_d_gate_q_e = tl.where(mask_ve, b_d_gate_q_e, 0.0)
        b_d_gate_q_raw = tl.sum(tl.trans(b_d_gate_q_e), axis=0)
        tl.store(
            p_d_gate_q,
            b_d_gate_q_raw.to(p_d_gate_q.dtype.element_ty),
            mask=mask_v,
        )

        if remaining > 0:
            b_d_log_rho_temporal = (
                b_d_log_rho_temporal * b_transition_weight
                + b_d_rho * b_1mg_q[:, None] * b_rho_prev
            )
        else:
            b_abs = tl.abs(b_d_log_rho_temporal)
            b_log_abs = tl.log(tl.where(b_abs > 0.0, b_abs, 1.0))
            b_abs_initial = tl.exp(
                b_log_abs
                + b_log_1mg_q[:, None]
                - b_log_rho_prior
            )
            b_d_rho_temporal = tl.where(
                b_d_log_rho_temporal == 0.0,
                0.0,
                tl.where(
                    b_d_log_rho_temporal < 0.0,
                    -b_abs_initial,
                    b_abs_initial,
                ),
            )
            b_d_rho_temporal += b_d_rho * b_1mg_q[:, None]
            b_d_rho_temporal = tl.where(
                mask_ve, b_d_rho_temporal, 0.0
            )

        tl.store(p_dpi_q, b_dpi_q.to(p_dpi_q.dtype.element_ty), mask=mask_ve)

        p_do -= H * D_v
        p_pi_q -= H * D_v * E
        p_qS -= H * D_v * E
        p_dpi_q -= H * D_v * E
        if USE_AUX_LOSS_GRAD:
            p_d_rho_seq -= H * D_v * E
        p_gate_q -= H * D_v
        p_d_gate_q -= H * D_v

    if STORE_TEMPORAL_STATE_GRADIENT:
        drho_offset = i_nh * D_v * E + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
        tl.store(drho_prev_init + drho_offset, b_d_rho_temporal.to(drho_prev_init.dtype.element_ty), mask=mask_ve)


@triton.jit(do_not_specialize=["T"])
def fused_recurrent_swila_temporal_bwd_replay_kernel(
    q,
    k,
    beta,
    r,
    u,
    rho,
    log_decay,
    h0,
    do,
    dq,
    dk_prediction,
    du_save,
    dlog_decay,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_k: tl.constexpr,
    D_v: tl.constexpr,
    E: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Replay state forward after the resource-safe temporal reverse pass."""
    T = T.to(tl.int64)
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_bt = T
        T = eos - bos
    else:
        bos = i_n * T
        all_bt = B * T

    e_range = tl.arange(0, BE).to(tl.int64)
    k_range = i_k * BK + tl.arange(0, BK).to(tl.int64)
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    mask_e = e_range < E
    mask_k = k_range < D_k
    mask_v = v_range < D_v
    mask_ve = mask_v[:, None] & mask_e[None, :]
    mask_vek = mask_v[:, None, None] & mask_e[None, :, None] & mask_k[None, None, :]

    b_h = tl.zeros([BV, BE, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * D_v * E * D_k + \
            v_range[:, None, None] * E * D_k + \
            e_range[None, :, None] * D_k + k_range[None, None, :]
        b_h += tl.load(p_h0, mask=mask_vek, other=0.0).to(tl.float32)

    p_q = q + (bos * H + i_h) * D_k + k_range
    p_k = k + (bos * H + i_h) * D_k + k_range
    p_do = do + (bos * H + i_h) * D_v + v_range
    base = (bos * H + i_h) * D_v * E + v_range[:, None] * E + e_range[None, :]
    p_beta = beta + base
    p_r = r + base
    p_u = u + base
    p_rho = rho + base
    p_du = du_save + base
    p_dq = dq + ((i_v * all_bt + bos) * H + i_h) * D_k + k_range
    p_dk_prediction = dk_prediction + ((i_v * all_bt + bos) * H + i_h) * D_k + k_range

    if USE_LOG_DECAY:
        decay_base = (bos * H + i_h) * D_k
        p_log_decay = log_decay + decay_base + k_range
        p_dlog_decay = dlog_decay + decay_base + k_range

    elapsed = 0
    for _ in range(T):
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0.0).to(tl.float32)
        b_beta = tl.load(p_beta, mask=mask_ve, other=0.0).to(tl.float32)
        b_r = tl.load(p_r, mask=mask_ve, other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_ve, other=0.0).to(tl.float32)
        b_rho = tl.load(p_rho, mask=mask_ve, other=0.0).to(tl.float32)
        b_du = tl.load(p_du, mask=mask_ve, other=0.0).to(tl.float32)

        if USE_LOG_DECAY:
            b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
            b_log_decay = tl.load(p_log_decay, mask=mask_k, other=0.0).to(tl.float32)
            b_h *= tl.exp(b_log_decay[None, None, :])

        # The replay contribution is kept separate from the direct write term.
        b_dk_weighted = tl.where(mask_vek, b_du[:, :, None] * b_h, 0.0)
        if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
            # Match the qualified combined replay's small-key reduction.
            b_dk_correction = tl.sum(
                tl.reshape(b_dk_weighted, (BV * BE, BK)), axis=0,
            )
        else:
            b_dk_per_expert = tl.sum(b_dk_weighted, axis=0)
            b_dk_correction = tl.sum(b_dk_per_expert, axis=0)
        tl.store(p_dk_prediction, -b_dk_correction, mask=mask_k)

        b_h = state_update(b_h, b_beta * b_r * b_u, b_k)
        b_weighted_h = tl.where(
            mask_vek,
            (b_do[:, None] * b_rho)[:, :, None] * b_h,
            0.0,
        )
        if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
            b_readout_total = tl.sum(
                tl.reshape(b_weighted_h, (BV * BE, BK)), axis=0,
            )
            b_dq = b_readout_total * scale
        else:
            b_output_per_expert = tl.sum(b_weighted_h, axis=0)
            b_dq = tl.sum(b_output_per_expert, axis=0) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        if USE_LOG_DECAY:
            if (BV == 8) & ((BE == 2) | (BE == 4)) & (BK == 16):
                b_output_local = (b_q * scale) * b_readout_total
                b_prediction_local = -b_k * b_dk_correction
            else:
                b_output_local_per_expert = (
                    (b_q * scale)[None, :] * b_output_per_expert
                )
                b_prediction_local_per_expert = -b_k[None, :] * b_dk_per_expert
                b_output_local = tl.sum(b_output_local_per_expert, axis=0)
                b_prediction_local = tl.sum(b_prediction_local_per_expert, axis=0)
            tl.atomic_add(
                p_dlog_decay,
                b_output_local + b_prediction_local,
                mask=mask_k,
            )

        if USE_LOG_DECAY:
            p_q += H * D_k
        p_k += H * D_k
        p_do += H * D_v
        p_beta += H * D_v * E
        p_r += H * D_v * E
        p_u += H * D_v * E
        p_rho += H * D_v * E
        p_du += H * D_v * E
        p_dq += H * D_k
        p_dk_prediction += H * D_k
        if USE_LOG_DECAY:
            p_log_decay += H * D_k
            p_dlog_decay += H * D_k
        elapsed += 1  # noqa: SIM113 - Triton does not support enumerate
