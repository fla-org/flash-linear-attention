# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Exact-identity temporal SwiLA forward kernel and launch wrapper."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.swila.atoms import log_softmax_2d, logaddexp, logsigmoid_pair, prediction, readout, state_update, sum_2d_axis1
from fla.ops.swila.atoms.activations import log_gate_mix
from fla.ops.swila.autotune import prune_temporal_forward_configs
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [8, 16, 32, 64]
        for num_warps in ([1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8])
        for num_stages in [2, 3, 4]
    ],
    # Checkpoint prepass and recompute have different store policies.  Keep
    # them, temporal-state modes, and decay layouts in the autotune key so a
    # light prepass winner is never reused for the heavier recompute.
    key=[
        'D_k', 'D_v', 'E', 'B', 'T', 'H', 'IDENTITY_TRANSITIONS',
        'USE_INITIAL_STATE', 'STORE_FINAL_STATE', 'STORE_R', 'STORE_U',
        'STORE_PI_BAR', 'STORE_RHO', 'STORE_QS',
        'USE_INITIAL_TEMPORAL_STATE', 'USE_GATE', 'USE_LOG_DECAY',
        'IS_VARLEN',
    ],
    prune_configs_by={"early_config_prune": prune_temporal_forward_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_swila_temporal_fwd_kernel(
    q,       # [B, T, H, D_k]
    k,       # [B, T, H, D_k]
    v,       # [B, T, H, D_v]
    beta,    # [B, T, H, D_v, E]
    pi_q,    # [B, T, H, D_v, E]
    pi_k,    # [B, T, H, D_v, E]
    log_decay,  # [B, T, H, D_k] (optional)
    o,       # [NK, B, T, H, D_v]
    h0,      # [B, H, D_v, E, D_k] initial state (optional)
    ht,      # [B, H, D_v, E, D_k] final state (optional)
    # Intermediate outputs for backward pass
    r_out,        # [B, T, H, D_v, E] - responsibilities
    u_out,        # [B, T, H, D_v, E] - error signals
    pi_bar_out,   # [B, T, H, D_v, E] - predicted prior (temporal) / r_prior (gate mode)
    rho_out,      # [B, T, H, D_v, E] - Q filtered state (temporal)
    qS_out,       # [B, T, H, D_v, E] - per-expert scaled output (temporal)
    # Temporal state inputs (optional)
    r_prev_init,      # [B, H, D_v, E] initial r_prev (temporal)
    rho_prev_init,    # [B, H, D_v, E] initial rho_prev (temporal)
    # Temporal state outputs
    r_prev_final,     # [B, H, D_v, E] final r_prev (temporal)
    rho_prev_final,   # [B, H, D_v, E] final rho_prev (temporal)
    # Gate inputs for reset_gate temporal coupling (optional)
    gate_kv,          # [B, T, H, D_v] - raw gate logits KV-side (optional)
    gate_q,           # [B, T, H, D_v] - raw gate logits Q-side (optional)
    cu_seqlens,       # [N+1] cumulative sequence lengths (optional, for variable-length)
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_k: tl.constexpr,  # Key/query dimension
    D_v: tl.constexpr,  # Value dimension
    E: tl.constexpr,    # Number of experts (actual)
    BK: tl.constexpr,   # Block size for D_k
    BV: tl.constexpr,   # Block size for D_v
    BE: tl.constexpr,   # Block size for E (must be power of 2, >= E)
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_INTERMEDIATES: tl.constexpr,
    STORE_R: tl.constexpr,
    STORE_U: tl.constexpr,
    STORE_PI_BAR: tl.constexpr,
    STORE_RHO: tl.constexpr,
    STORE_QS: tl.constexpr,
    USE_INITIAL_TEMPORAL_STATE: tl.constexpr,
    USE_GATE: tl.constexpr,
    IDENTITY_TRANSITIONS: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    IS_VARLEN: tl.constexpr = False,
):
    """
    Fused recurrent SwiLA forward kernel with temporal (rSLDS) transitions.

    Grid: (NV, NK, N*H) where NV = cdiv(D_v, BV), NK = cdiv(D_k, BK)
    Each thread block handles a block of V dimensions across all experts.
    """
    # Retain these metadata keys for the established forward launch policy.
    tl.static_assert(IDENTITY_TRANSITIONS, "temporal SwiLA requires exact identity carry")
    tl.static_assert(USE_GATE, "temporal SwiLA requires reset gates")
    # Grid indices
    T = T.to(tl.int64)
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    # Beginning of sequence offset
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_bt = T
        T = eos - bos
    else:
        bos = i_n * T
        all_bt = B * T

    # Value and key ranges for this block
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    k_range = i_k * BK + tl.arange(0, BK).to(tl.int64)
    mask_v = v_range < D_v
    mask_k = k_range < D_k

    # Initialize pointers for q, k, v (shape: [B, T, H, D])
    # Pointer to first element of this batch/head at t=0
    p_q = q + (bos * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
    p_k = k + (bos * H + i_h) * D_k + i_k * BK + tl.arange(0, BK).to(tl.int64)
    p_v = v + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64)

    # Pointers for beta, pi_q, pi_k (shape: [B, T, H, D_v, E])
    # Base offset for this batch/head/v_block
    # Use BE for arange but E for actual memory layout
    e_range = tl.arange(0, BE).to(tl.int64)
    base_pi = (bos * H + i_h) * D_v * E + (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    p_beta = beta + base_pi
    p_pi_q = pi_q + base_pi
    p_pi_k = pi_k + base_pi
    if USE_LOG_DECAY:
        p_log_decay = log_decay + (bos * H + i_h) * D_k + k_range

    # Output pointer (shape: [NK, B, T, H, D_v])
    p_o = o + ((i_k * all_bt + bos) * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64)

    # Masks for pi/beta loading [BV, BE] - check actual E bounds
    mask_e = e_range < E
    mask_ve = mask_v[:, None] & mask_e[None, :]

    # Pointers for intermediate outputs (r, u, pi_bar, rho, qS) if storing
    # Only first K-block writes intermediates to avoid duplication
    if STORE_R:
        p_r = r_out + base_pi
    if STORE_U:
        p_u = u_out + base_pi
    if STORE_PI_BAR:
        p_pi_bar = pi_bar_out + base_pi
    if STORE_RHO:
        p_rho = rho_out + base_pi
    if STORE_QS:
        p_qS = qS_out + base_pi

    # Initialize hidden states for all experts: [BV, BE, BK]
    # State shape in memory: [B, H, D_v, E, D_k]
    # We use 3D indexing with proper offset computation

    # Masks for 3D state loading/storing [BV, BE, BK]
    mask_vek = mask_v[:, None, None] & mask_e[None, :, None] & mask_k[None, None, :]

    # Initialize hidden states (use BE for block size)
    b_h = tl.zeros([BV, BE, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        # h0 shape: [B, H, D_v, E, D_k]
        # Offset: nh * D_v * E * D_k + v * E * D_k + e * D_k + k
        # Using 3D tensor indexing: [BV, BE, BK]
        h0_base = h0 + i_nh * D_v * E * D_k
        h0_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
            e_range[None, :, None] * D_k + \
            (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
        b_h = tl.load(h0_base + h0_offset, mask=mask_vek, other=0.0).to(tl.float32)

    # KV carries log(r); Q carries log(E*rho), a fixed offset that improves
    # FP32 precision near a uniform prior without normalizing supplied mass.
    log_E = tl.log(tl.cast(E, tl.float32))
    # r_prev_init/rho_prev_init shape: [B, H, D_v, E]
    rp_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None]) * E + e_range[None, :]
    if USE_INITIAL_TEMPORAL_STATE:
        rp_base = r_prev_init + i_nh * D_v * E
        b_r_prev_prob = tl.load(rp_base + rp_offset, mask=mask_ve, other=0).to(tl.float32)
        rhop_base = rho_prev_init + i_nh * D_v * E
        b_rho_prev_prob = tl.load(rhop_base + rp_offset, mask=mask_ve, other=0).to(tl.float32)
        # Exact mode carries the supplied probability directly. Zeros are
        # valid here: the finite reset branch makes the gate mixture
        # positive, and the custom backward differentiates the equivalent
        # probability-space equation without passing through log(0).
        b_log_r_prev = tl.log(b_r_prev_prob)
        b_log_rho_prev = tl.log(b_rho_prev_prob) + log_E
    else:
        # Uniform KV log carry is -log(E); centered Q log carry is zero.
        log_inv_E = -tl.log(tl.cast(E, tl.float32))
        b_log_r_prev = tl.full([BV, BE], log_inv_E, dtype=tl.float32)
        b_log_rho_prev = tl.zeros([BV, BE], dtype=tl.float32)
        # Set invalid elements to -inf
        b_log_r_prev = tl.where(mask_e[None, :], b_log_r_prev, float('-inf'))
        b_log_rho_prev = tl.where(mask_e[None, :], b_log_rho_prev, float('-inf'))

    # Gate pointers and precomputed transition matrices for reset_gate mode
    # gate_kv/gate_q shape: [B, T, H, D_v], stride: T*H*D_v per batch
    p_gate_kv = gate_kv + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64)
    p_gate_q = gate_q + (bos * H + i_h) * D_v + i_v * BV + tl.arange(0, BV).to(tl.int64)

    # Main recurrence loop
    for _ in range(0, T):
        # Load inputs for this timestep
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # [BK]
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # [BV]
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale  # [BK]

        # Load beta, pi_k, pi_q for this timestep [BV, E]
        b_beta = tl.load(p_beta, mask=mask_ve, other=0).to(tl.float32)  # [BV, E]
        b_pi_k = tl.load(p_pi_k, mask=mask_ve, other=0).to(tl.float32)  # [BV, E]
        b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0).to(tl.float32)  # [BV, E]

        # Prediction and routing use the state after key-wise decay.
        if USE_LOG_DECAY:
            b_log_decay = tl.load(p_log_decay, mask=mask_k, other=0.0).to(tl.float32)
            b_h *= tl.exp(b_log_decay[None, None, :])

        # Step 1: Compute predictions and errors for all experts
        b_pred = prediction(b_h, b_k)  # [BV, E]

        # Error: u_e = v - pred_e
        b_u = b_v[:, None] - b_pred  # [BV, E]

        b_u_sq = b_u * b_u  # [BV, E]

        # === TEMPORAL KV SIDE ===
        # --- RESET GATE MODE (KV-side) — LOG-SPACE ---
        # Load gate raw logit: [BV]
        b_gate_kv = tl.load(p_gate_kv, mask=mask_v, other=0).to(tl.float32)  # [BV]

        # Stable logsigmoid pair
        b_log_g_kv, b_log_1mg_kv = logsigmoid_pair(b_gate_kv)

        b_log_r_transition = b_log_r_prev

        # 2. Reset path in log-space: log_r_reset = log_softmax(pi_k)
        b_log_r_reset = log_softmax_2d(b_pi_k, mask_e)  # [BV, BE]

        # 3. Gate mix in log-space via logaddexp
        b_a = b_log_1mg_kv[:, None] + b_log_r_transition  # [BV, BE]
        b_b = b_log_g_kv[:, None] + b_log_r_reset  # [BV, BE]
        b_log_r_prior = logaddexp(b_a, b_b)
        b_log_r_prior = tl.where(mask_e[None, :], b_log_r_prior, float('-inf'))

        # Store pi_bar in probability space (needed by backward kernel)
        b_pi_bar = tl.exp(b_log_r_prior)
        b_pi_bar = tl.where(mask_ve, b_pi_bar, 0.0)

        # Bayesian update (already in log-space)
        b_log_posterior = b_log_r_prior - 0.5 * b_u_sq
        # MASK: Set invalid expert indices to -inf before softmax
        b_log_posterior = tl.where(mask_e[None, :], b_log_posterior, float('-inf'))
        # Softmax to get r
        b_log_posterior_t = tl.trans(b_log_posterior)
        b_logits_max_t = tl.max(b_log_posterior_t, axis=0)
        b_logits_max_safe = tl.where(b_logits_max_t == float('-inf'), 0.0, b_logits_max_t)
        b_logits_max = b_logits_max_safe[:, None]
        b_exp = tl.exp(b_log_posterior - b_logits_max)
        b_exp = tl.where(mask_ve, b_exp, 0.0)
        b_exp_t = tl.trans(b_exp)
        b_sum_exp_t = tl.sum(b_exp_t, axis=0)
        b_sum_exp = tl.maximum(b_sum_exp_t[:, None], 1e-8)
        b_r = b_exp / b_sum_exp

        # Log carry (direct — no exp→log round-trip)
        b_log_r_prev = (b_log_posterior - b_logits_max - tl.log(b_sum_exp)).to(tl.float32)

        # Step 3: Apply the update to the decayed state.
        b_update_coef = b_beta * b_r * b_u  # [BV, E]
        b_h = state_update(b_h, b_update_coef, b_k).to(tl.float32)  # [BV, E, BK]

        # === TEMPORAL Q SIDE ===
        # Compute qS first (needed for output in both modes)
        b_qS = readout(b_h, b_q)  # [BV, E]

        b_qS_normed = b_qS

        # --- RESET GATE MODE (Q-side) — LOG-SPACE ---
        # Load gate raw logit: [BV]
        b_gate_q = tl.load(p_gate_q, mask=mask_v, other=0).to(tl.float32)  # [BV]

        b_log_rho_transition = b_log_rho_prev

        # Reset and transition share the fixed log(E) carry offset.
        b_log_rho_reset = log_softmax_2d(b_pi_q, mask_e) + log_E
        b_log_rho = log_gate_mix(
            b_log_rho_transition, b_log_rho_reset, b_gate_q[:, None]
        )
        b_log_rho = tl.where(mask_e[None, :], b_log_rho, float('-inf'))

        # Convert to probability for output computation
        b_rho = tl.exp(b_log_rho - log_E)
        b_rho = tl.where(mask_ve, b_rho, 0.0)

        # Output (use normalized qS for output computation)
        b_o = sum_2d_axis1(b_rho * b_qS_normed)  # [BV]

        # Log carry (direct — no exp→log round-trip)
        b_log_rho_prev = b_log_rho.to(tl.float32)

        # Store intermediates for backward pass
        if STORE_R:
            tl.store(p_r, b_r.to(p_r.dtype.element_ty), mask=mask_ve)
        if STORE_U:
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=mask_ve)
        if STORE_PI_BAR:
            tl.store(p_pi_bar, b_pi_bar.to(p_pi_bar.dtype.element_ty), mask=mask_ve)
        if STORE_RHO:
            tl.store(p_rho, b_rho.to(p_rho.dtype.element_ty), mask=mask_ve)
        if STORE_QS:
            tl.store(p_qS, b_qS.to(p_qS.dtype.element_ty), mask=mask_ve)

        # Store output
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Advance pointers
        p_q += H * D_k
        p_k += H * D_k
        p_v += H * D_v
        p_o += H * D_v
        p_beta += H * D_v * E
        p_pi_q += H * D_v * E
        p_pi_k += H * D_v * E
        if USE_LOG_DECAY:
            p_log_decay += H * D_k
        if STORE_R:
            p_r += H * D_v * E
        if STORE_U:
            p_u += H * D_v * E
        if STORE_PI_BAR:
            p_pi_bar += H * D_v * E
        if STORE_RHO:
            p_rho += H * D_v * E
        if STORE_QS:
            p_qS += H * D_v * E
        p_gate_kv += H * D_v
        p_gate_q += H * D_v

    # Store final state
    if STORE_FINAL_STATE:
        # ht shape: [B, H, D_v, E, D_k]
        # Using 3D tensor indexing: [BV, BE, BK]
        ht_base = ht + i_nh * D_v * E * D_k
        ht_offset = (i_v * BV + tl.arange(0, BV).to(tl.int64)[:, None, None]) * E * D_k + \
            e_range[None, :, None] * D_k + \
            (i_k * BK + tl.arange(0, BK).to(tl.int64)[None, None, :])
        tl.store(ht_base + ht_offset, b_h.to(ht.dtype.element_ty), mask=mask_vek)

        # Store final temporal states (convert from log-space to probability)
        rpf_base = r_prev_final + i_nh * D_v * E
        tl.store(rpf_base + rp_offset, tl.exp(b_log_r_prev).to(r_prev_final.dtype.element_ty), mask=mask_ve)
        rhopf_base = rho_prev_final + i_nh * D_v * E
        tl.store(rhopf_base + rp_offset, tl.exp(b_log_rho_prev - log_E).to(rho_prev_final.dtype.element_ty), mask=mask_ve)


def fused_recurrent_swila_temporal_fwd(
    q: torch.Tensor,            # [B, T, H, D_k]
    k: torch.Tensor,            # [B, T, H, D_k]
    v: torch.Tensor,            # [B, T, H, D_v]
    beta: torch.Tensor,         # [B, T, H, D_v, E]
    pi_q: torch.Tensor,         # [B, T, H, D_v, E]
    pi_k: torch.Tensor,         # [B, T, H, D_v, E]
    scale: float,
    log_decay: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, D_v, E, D_k]
    output_final_state: bool = False,
    output_intermediates: bool = False,
    r_prev_init: torch.Tensor | None = None,    # [B, H, D_v, E]
    rho_prev_init: torch.Tensor | None = None,  # [B, H, D_v, E]
    gate_kv: torch.Tensor | None = None,        # [B, T, H, D_v] reset gate KV-side
    gate_q: torch.Tensor | None = None,         # [B, T, H, D_v] reset gate Q-side
    cu_seqlens: torch.Tensor | None = None,     # [N+1] cumulative sequence lengths
    checkpoint_prepass: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Forward pass for fused recurrent SwiLA with temporal transitions.

    ``log_decay`` follows the layer contract: ``[B,T,H,D_k]``, shared over
    values and experts and applied before prediction and routing.
    """
    B, T, H, D_k_dim = q.shape
    D_v_dim = v.shape[-1]
    E_dim = beta.shape[-1]  # Number of experts
    assert 1 <= E_dim <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={E_dim}"
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Compute block sizes (BV autotuned; BE/BK fixed)
    BK = max(triton.next_power_of_2(D_k_dim), 16)
    BE = triton.next_power_of_2(E_dim)
    NK = triton.cdiv(D_k_dim, BK)

    assert NK == 1, f"NK > 1 not supported yet (D_k={D_k_dim}, BK={BK}, NK={NK})"
    if gate_kv is None or gate_q is None:
        raise ValueError("temporal SwiLA requires reset gate inputs")

    # Allocate output tensor
    o = q.new_empty(NK, B, T, H, D_v_dim)

    # Allocate final state if needed (N sequences, not B)
    if output_final_state:
        final_state = q.new_empty(N, H, D_v_dim, E_dim, D_k_dim, dtype=torch.float32)
        r_prev_final = q.new_empty(N, H, D_v_dim, E_dim, dtype=torch.float32)
        rho_prev_final = q.new_empty(N, H, D_v_dim, E_dim, dtype=torch.float32)
    else:
        final_state = None
        r_prev_final = None
        rho_prev_final = None

    # A checkpoint prepass needs r/rho to form the MI output, but its backward
    # graph is discarded and recomputed. Avoid writing the three E-wide
    # backward-only sequences until recomputation.
    store_responsibilities = output_intermediates
    store_backward_intermediates = output_intermediates and not checkpoint_prepass
    if store_responsibilities:
        r_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
        rho_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
    else:
        r_out = None
        rho_out = None
    if store_backward_intermediates:
        u_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
        pi_bar_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
        qS_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
    else:
        u_out = None
        pi_bar_out = None
        qS_out = None

    is_varlen = cu_seqlens is not None
    use_log_decay = log_decay is not None

    # Launch kernel (BV, num_warps, num_stages all autotuned).
    def grid(meta):
        return (triton.cdiv(D_v_dim, meta['BV']), NK, N * H)
    fused_recurrent_swila_temporal_fwd_kernel[grid](
        q, k, v,
        beta, pi_q, pi_k,
        log_decay,
        o,
        initial_state,
        final_state,
        r_out,
        u_out,
        pi_bar_out,
        rho_out,
        qS_out,
        r_prev_init,
        rho_prev_init,
        r_prev_final,
        rho_prev_final,
        gate_kv,
        gate_q,
        cu_seqlens,
        scale,
        T,
        B=B,
        H=H,
        D_k=D_k_dim,
        D_v=D_v_dim,
        E=E_dim,
        BK=BK,
        BE=BE,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        STORE_INTERMEDIATES=(
            store_responsibilities or store_backward_intermediates
        ),
        STORE_R=store_responsibilities,
        STORE_U=store_backward_intermediates,
        STORE_PI_BAR=store_backward_intermediates,
        STORE_RHO=store_responsibilities,
        STORE_QS=store_backward_intermediates,
        USE_INITIAL_TEMPORAL_STATE=r_prev_init is not None,
        USE_GATE=True,
        IDENTITY_TRANSITIONS=True,
        USE_LOG_DECAY=use_log_decay,
        IS_VARLEN=is_varlen,
    )

    # Remove NK dimension (it's always 1)
    o = o.squeeze(0)

    # Package intermediates
    if output_intermediates:
        intermediates = (r_out, u_out, pi_bar_out, rho_out, qS_out)
    else:
        intermediates = None

    return o, final_state, intermediates, r_prev_final, rho_prev_final
