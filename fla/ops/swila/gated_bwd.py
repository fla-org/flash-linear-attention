# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Gated SwiLA backward Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.swila.atoms import dot_3d_axis2, readout, softmax_2d, softmax_backward_2d, state_update
from fla.ops.swila.autotune import prune_gated_backward_configs
from fla.ops.swila.dispatch import (
    _is_nvidia_sm90_tensor,
    _use_dense_d64e4_key_bwd_fastpath,
)
from fla.ops.swila.gated_split_bwd import (
    _DENSE_D64E4_BWD_MIN_VALUE_TILE,
    _fused_recurrent_gated_swila_bwd_scheduled,
    _fused_recurrent_gated_swila_dense_d64e4_bwd_replay_kernel,
    _fused_recurrent_gated_swila_dense_d64e4_bwd_reverse_kernel,
    _reverse_cumsum_log_decay_kernel,
)
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages,
                      pre_hook=lambda args: (
                          args['dq'].zero_(),
                          args['dk'].zero_(),
                          args['dlog_decay'].zero_(),
        ))
        for BV in [8, 16, 32, 64]
        for num_warps in ([1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8])
        for num_stages in [2, 3, 4]
    ],
    key=['D_k', 'D_v', 'E'],
    prune_configs_by={"early_config_prune": prune_gated_backward_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_swila_bwd_kernel(
    # Inputs (saved from forward)
    q,       # [B, T, H, D_k]
    k,       # [B, T, H, D_k]
    v,       # [B, T, H, D_v]
    beta,    # [B, T, H, D_v, E]
    pi_q,    # [B, T, H, D_v, E]
    pi_k,    # [B, T, H, D_v, E]
    log_decay,  # [B, T, H, D_k]
    # Saved intermediates from forward
    r,       # [B, T, H, D_v, E] - responsibilities
    u,       # [B, T, H, D_v, E] - error signals
    h0,      # [N, H, D_v, E, D_k] initial state (optional)
    # Gradient inputs
    do,      # [B, T, H, D_v]
    dht,     # [N, H, D_v, E, D_k] final state gradient (optional)
    ht,      # [N, H, D_v, E, D_k] final state paired with dht (optional)
    d_r_seq,  # [B, T, H, D_v, E] auxiliary loss gradient w.r.t. r (optional)
    # Gradient outputs
    dq,      # [NV, B, T, H, D_k]
    dk,      # [NV, B, T, H, D_k]
    dv,      # [B, T, H, D_v]
    dbeta,   # [B, T, H, D_v, E]
    dpi_q,   # [B, T, H, D_v, E]
    dpi_k,   # [B, T, H, D_v, E]
    dlog_decay,  # float32, same shape as log_decay; local terms then reverse scan
    du_save,  # [B, T, H, D_v, E] - stores grad_u from Phase 1 for Phase 2 correction
    dh0,     # [N, H, D_v, E, D_k] initial state gradient (optional)
    cu_seqlens,  # [N+1] cumulative sequence lengths (optional, for variable-length)
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
    BE: tl.constexpr,   # Block size for experts (power of 2)
    NK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_AUX_LOSS_GRAD: tl.constexpr,
    IS_VARLEN: tl.constexpr = False,
):
    """
    Fused recurrent gated SwiLA backward kernel using saved intermediates (r, u).

    Phase 1: Backward-in-time (t = T-1 to 0)
        - Load saved r, u
        - Compute gradients for v, beta, pi_k, partial k
        - Accumulate grad_W (b_dh) going backward
        - Atomically accumulate the coordinatewise state-update contribution
          into the compact dlog_decay buffer
        - Store dh0 at end of phase

    Phase 2: Forward-in-time (t = 0 to T-1)
        - Reconstruct W (state)
        - Compute grad_q, grad_pi_q
        - Correct grad_k using W_before
        - Atomically add readout and prediction contributions to dlog_decay
    """
    # Grid indices
    T = T.to(tl.int64)
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    # Sequence boundaries (variable-length or fixed-length)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_bt = T
        T = eos - bos
    else:
        bos = i_n * T
        all_bt = B * T

    # Ranges and masks
    k_range = i_k * BK + tl.arange(0, BK).to(tl.int64)
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    e_range = tl.arange(0, BE).to(tl.int64)
    mask_k = k_range < D_k
    mask_v = v_range < D_v
    mask_e = e_range < E
    mask_ve = mask_v[:, None] & mask_e[None, :]
    mask_vek = mask_v[:, None, None] & mask_e[None, :, None] & mask_k[None, None, :]

    # Clamped indices for safe pointer arithmetic (prevents out-of-bounds access)
    v_idx = tl.minimum(v_range, D_v - 1)
    k_idx = tl.minimum(k_range, D_k - 1)
    e_idx = tl.minimum(e_range, E - 1)

    # ===== PHASE 1: Backward-in-time =====
    # Initialize pointers at END of sequence (t = T-1)
    p_q = q + (bos * H + i_h) * D_k + k_idx + (T - 1) * H * D_k
    p_k = k + (bos * H + i_h) * D_k + k_idx + (T - 1) * H * D_k
    p_do = do + (bos * H + i_h) * D_v + v_idx + (T - 1) * H * D_v
    base_pi_end = ((bos + T - 1) * H + i_h) * D_v * E + v_idx[:, None] * E + e_idx[None, :]
    p_pi_q = pi_q + base_pi_end
    p_beta = beta + base_pi_end
    p_r = r + base_pi_end
    p_u = u + base_pi_end
    if USE_AUX_LOSS_GRAD:
        p_d_r_seq = d_r_seq + base_pi_end
    decay_base_end = ((bos + T - 1) * H + i_h) * D_k + k_idx
    p_log_decay_bwd = log_decay + decay_base_end
    p_dlog_decay_bwd = dlog_decay + decay_base_end

    # Output pointers at END (dv, dbeta, dpi_k are direct-write: [B, T, H, ...])
    p_dv = dv + ((bos + T - 1) * H + i_h) * D_v + v_idx
    dve_base_end = ((bos + T - 1) * H + i_h) * D_v * E + v_idx[:, None] * E + e_idx[None, :]
    p_dbeta = dbeta + dve_base_end
    p_dpi_k = dpi_k + dve_base_end
    p_dk = dk + ((i_v * all_bt + bos + T - 1) * H + i_h) * D_k + k_idx
    # Pointer for saving du (for Phase 2 correction)
    p_du_save = du_save + base_pi_end

    # Initialize accumulated state gradient (grad_W in Python code)
    b_dh = tl.zeros([BV, BE, BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * D_v * E * D_k + \
            v_idx[:, None, None] * E * D_k + \
            e_idx[None, :, None] * D_k + \
            k_idx[None, None, :]
        b_dht = tl.load(p_dht, mask=mask_vek, other=0.0).to(tl.float32)
        b_dh += b_dht

        # The generalized-sigma recurrence is seeded after the final token by
        # <dht, ht>, reduced over the value axis (and experts for ``key``).
        p_ht = ht + i_nh * D_v * E * D_k + \
            v_idx[:, None, None] * E * D_k + \
            e_idx[None, :, None] * D_k + \
            k_idx[None, None, :]
        b_ht = tl.load(p_ht, mask=mask_vek, other=0.0).to(tl.float32)
        b_tail_e = tl.sum(b_dht * b_ht, axis=0)
        b_tail = tl.sum(b_tail_e, axis=0)
        tl.atomic_add(p_dlog_decay_bwd, b_tail, mask=mask_k)

    # Backward loop: compute gradients using saved r, u
    for _ in range(T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0).to(tl.float32)
        b_beta = tl.load(p_beta, mask=mask_ve, other=0).to(tl.float32)

        # Load saved intermediates
        b_r = tl.load(p_r, mask=mask_ve, other=0).to(tl.float32)          # [BV, E]
        b_u = tl.load(p_u, mask=mask_ve, other=0).to(tl.float32)          # [BV, E]
        b_log_decay = tl.load(
            p_log_decay_bwd,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        # Output gradient w.r.t. state: weight do by rho = softmax(pi_q)
        b_pi_q_star = softmax_2d(b_pi_q, mask_e)  # [BV, BE]
        b_do_weighted = b_do[:, None] * b_pi_q_star  # [BV, BE]
        b_dh += (b_q * scale)[None, None, :] * b_do_weighted[:, :, None]

        # dL/d_coef where coef = beta * r * u
        b_d_coef = dot_3d_axis2(b_dh, b_k)  # [BV, BE]

        # Gradients through coef = beta * r * u
        b_dbeta = b_d_coef * b_r * b_u
        b_d_r = b_d_coef * b_beta * b_u
        b_d_u = b_d_coef * b_beta * b_r

        # Add auxiliary loss gradient w.r.t. r
        if USE_AUX_LOSS_GRAD:
            b_d_r_aux = tl.load(p_d_r_seq, mask=mask_ve, other=0).to(tl.float32)
            b_d_r = b_d_r + b_d_r_aux

        # Gradient through routing logits.
        b_d_logits = softmax_backward_2d(b_r, b_d_r)
        b_dpi_k = b_d_logits
        b_d_u += b_d_logits * (-b_u)

        # Gradient through u = v - pred
        # WORKAROUND: tl.sum(2D, axis=1) may have Triton bug at small block sizes.
        # Use transpose + axis=0 to be safe.
        b_d_u_t = tl.trans(b_d_u)  # [BE, BV]
        b_dv = tl.sum(b_d_u_t, axis=0)  # [BV]

        # Partial dk from state update: dk = sum_{v,e}(b_dh * coef)
        b_coef = b_beta * b_r * b_u
        b_dh_coef = b_dh * b_coef[:, :, None]  # [BV, BE, BK]
        b_dk_2d = tl.sum(b_dh_coef, axis=0)    # [BE, BK]
        b_dk = tl.sum(b_dk_2d, axis=0)          # [BK]

        # Store gradients computed in Phase 1
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), mask=mask_ve)
        tl.store(p_dpi_k, b_dpi_k.to(p_dpi_k.dtype.element_ty), mask=mask_ve)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        # Save b_d_u for Phase 2 grad_k correction
        tl.store(p_du_save, b_d_u.to(p_du_save.dtype.element_ty), mask=mask_ve)

        # Coordinatewise update contribution:
        #   -k_j * sum_v(G_t[v,e,j] * c_t[v,e]).
        # Readout and prediction terms are added during the forward replay.
        b_update_local = -b_dh_coef * b_k[None, None, :]
        b_update_local = tl.sum(b_update_local, axis=0)  # [BE, BK]
        b_update_local = tl.sum(b_update_local, axis=0)
        tl.atomic_add(p_dlog_decay_bwd, b_update_local, mask=mask_k)

        # Apply decay after the prediction contribution: D * (G - du k^T).
        b_dh -= b_k[None, None, :] * b_d_u[:, :, None]
        b_dh *= tl.exp(b_log_decay[None, None, :])

        # Move pointers backward
        p_q -= H * D_k
        p_k -= H * D_k
        p_do -= H * D_v
        p_pi_q -= H * D_v * E
        p_beta -= H * D_v * E
        p_r -= H * D_v * E
        p_u -= H * D_v * E
        if USE_AUX_LOSS_GRAD:
            p_d_r_seq -= H * D_v * E
        p_log_decay_bwd -= H * D_k
        p_dlog_decay_bwd -= H * D_k
        p_dv -= H * D_v
        p_dbeta -= H * D_v * E
        p_dpi_k -= H * D_v * E
        p_dk -= H * D_k
        p_du_save -= H * D_v * E

    # Store initial state gradient (after phase 1, before barrier)
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * D_v * E * D_k + \
            v_idx[:, None, None] * E * D_k + \
            e_idx[None, :, None] * D_k + \
            k_idx[None, None, :]
        tl.store(p_dh0, b_dh.to(dh0.dtype.element_ty), mask=mask_vek)

    tl.debug_barrier()

    # ===== PHASE 2: Forward-in-time to compute dq, dpi_q, and correct dk =====
    b_h = tl.zeros([BV, BE, BK], dtype=tl.float32)

    # Reset pointers to START of sequence
    p_q = q + (bos * H + i_h) * D_k + k_idx
    p_k = k + (bos * H + i_h) * D_k + k_idx
    p_do = do + (bos * H + i_h) * D_v + v_idx
    base_pi_start = (bos * H + i_h) * D_v * E + v_idx[:, None] * E + e_idx[None, :]
    p_beta = beta + base_pi_start
    p_pi_q = pi_q + base_pi_start
    p_r = r + base_pi_start
    p_u = u + base_pi_start

    p_dq = dq + ((i_v * all_bt + bos) * H + i_h) * D_k + k_idx
    p_dk = dk + ((i_v * all_bt + bos) * H + i_h) * D_k + k_idx
    p_dpi_q = dpi_q + (bos * H + i_h) * D_v * E + v_idx[:, None] * E + e_idx[None, :]
    # Pointer for loading saved du
    p_du_save = du_save + base_pi_start

    decay_base_start = (bos * H + i_h) * D_k + k_idx
    p_log_decay_fwd = log_decay + decay_base_start
    p_dlog_decay_fwd = dlog_decay + decay_base_start

    # Load initial state if provided
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * D_v * E * D_k + \
            v_idx[:, None, None] * E * D_k + \
            e_idx[None, :, None] * D_k + \
            k_idx[None, None, :]
        b_h += tl.load(p_h0, mask=mask_vek, other=0.0).to(tl.float32)

    # Forward loop: reconstruct h and compute dq, dpi_q, correct dk
    for i_t in range(T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_pi_q = tl.load(p_pi_q, mask=mask_ve, other=0).to(tl.float32)

        # Load saved intermediates
        b_beta = tl.load(p_beta, mask=mask_ve, other=0).to(tl.float32)
        b_r = tl.load(p_r, mask=mask_ve, other=0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_ve, other=0).to(tl.float32)

        # Load saved grad_u from Phase 1
        b_d_u = tl.load(p_du_save, mask=mask_ve, other=0).to(tl.float32)

        # Reconstruct the decayed state used by prediction.
        b_log_decay = tl.load(
            p_log_decay_fwd,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_h *= tl.exp(b_log_decay[None, None, :])

        # Add the coordinatewise prediction contribution at this timestep.
        b_pred_local = -b_d_u[:, :, None] * b_h * b_k[None, None, :]
        b_pred_local = tl.sum(b_pred_local, axis=0)  # [BE, BK]

        # Correct dk using the same state used by prediction.
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        b_dk_weighted = b_d_u[:, :, None] * b_h  # [BV, BE, BK]
        b_dk_intermediate = tl.sum(b_dk_weighted, axis=0)  # [BE, BK]
        b_dk_correction = tl.sum(b_dk_intermediate, axis=0)  # [BK]
        b_dk = b_dk - b_dk_correction

        # Update state: h_t = decay * h_{t-1} + coef * k
        b_coef = b_beta * b_r * b_u
        b_h = state_update(b_h, b_coef, b_k)

        # Output gradients (using h_after)
        b_pi_q_star = softmax_2d(b_pi_q, mask_e)  # [BV, BE]
        b_d_qS = b_do[:, None] * b_pi_q_star  # [BV, BE]

        # dq = sum_{v,e}(d_qS * h_after) * scale
        b_weighted_h = b_d_qS[:, :, None] * b_h  # [BV, BE, BK]
        b_intermediate = tl.sum(b_weighted_h, axis=0)  # [BE, BK]
        b_dq = tl.sum(b_intermediate, axis=0) * scale  # [BK]

        # Add the coordinatewise readout contribution
        # sum_v(do[v] * rho[v,e] * q_scaled[j] * S_t[v,e,j]).
        b_output_local = (
            b_d_qS[:, :, None]
            * b_h
            * (b_q * scale)[None, None, :]
        )
        b_output_local = tl.sum(b_output_local, axis=0)  # [BE, BK]
        b_replay_local = b_pred_local + b_output_local

        # dpi_q: softmax backward for softmax(pi_q)
        b_qS = readout(b_h, b_q * scale)  # [BV, BE]
        b_dpi_q_star = b_do[:, None] * b_qS
        b_dpi_q = softmax_backward_2d(b_pi_q_star, b_dpi_q_star)

        # Store gradients
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.store(p_dpi_q, b_dpi_q.to(p_dpi_q.dtype.element_ty), mask=mask_ve)
        b_replay_local = tl.sum(b_replay_local, axis=0)
        tl.atomic_add(p_dlog_decay_fwd, b_replay_local, mask=mask_k)

        # Move pointers forward
        p_q += H * D_k
        p_k += H * D_k
        p_do += H * D_v
        p_pi_q += H * D_v * E
        p_beta += H * D_v * E
        p_r += H * D_v * E
        p_u += H * D_v * E
        p_dq += H * D_k
        p_dk += H * D_k
        p_dpi_q += H * D_v * E
        p_du_save += H * D_v * E
        p_log_decay_fwd += H * D_k
        p_dlog_decay_fwd += H * D_k


def fused_recurrent_gated_swila_bwd(
    q: torch.Tensor,            # [B, T, H, D_k]
    k: torch.Tensor,            # [B, T, H, D_k]
    v: torch.Tensor,            # [B, T, H, D_v]
    beta: torch.Tensor,         # [B, T, H, D_v, E]
    pi_q: torch.Tensor,         # [B, T, H, D_v, E]
    pi_k: torch.Tensor,         # [B, T, H, D_v, E]
    # Saved intermediates from forward
    r: torch.Tensor,            # [B, T, H, D_v, E]
    u: torch.Tensor,            # [B, T, H, D_v, E]
    initial_state: torch.Tensor | None,  # [N, H, D_v, E, D_k]
    do: torch.Tensor,           # [B, T, H, D_v]
    dht: torch.Tensor | None,  # [N, H, D_v, E, D_k]
    log_decay: torch.Tensor = None,
    final_state: torch.Tensor | None = None,  # [N, H, D_v, E, D_k]
    d_r_seq: torch.Tensor | None = None,  # [B, T, H, D_v, E] auxiliary loss gradient w.r.t. r
    scale: float = 1.0,
    cu_seqlens: torch.LongTensor | None = None,
    *,
    dense_split_schedule: tuple[tuple[int, int, int], ...] | str | None = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Backward pass for fused recurrent gated SwiLA.

    Returns:
        dq, dk, dv, dbeta, dpi_q, dpi_k, dh0, dlog_decay
    """
    assert 1 <= beta.shape[-1] <= 8, f"fused_recurrent SwiLA requires 1 <= num_experts <= 8, got E={beta.shape[-1]}"
    # An explicit None retains the native autotuned control. Omission selects
    # automatically, but only within the measured speed/memory envelope. Keep
    # the existing predicate authoritative so forced-generic tests stay generic.
    if isinstance(dense_split_schedule, str):
        if dense_split_schedule != "auto":
            raise ValueError("dense_split_schedule must be 'auto', None, or a declared pair")
        dense_split_schedule = None
        if (
            tuple(q.shape) == (1, 4096, 16, 64)
            and log_decay is not None
            and _use_dense_d64e4_key_bwd_fastpath(q, k, v, beta, pi_q, pi_k, log_decay, cu_seqlens)
            and _is_nvidia_sm90_tensor(q)
            and "H100" in torch.cuda.get_device_name(q.device)
            and all(tensor is None or tensor.device == q.device
                    for tensor in (k, v, beta, pi_q, pi_k, log_decay, r, u, do,
                                   initial_state, final_state, dht, d_r_seq))
            and all(tensor is None or (tensor.dtype == torch.float32 and tensor.is_contiguous())
                    for tensor in (r, u, final_state, dht, d_r_seq))
            and do.dtype == torch.bfloat16 and do.is_contiguous()
            and (initial_state is None or (initial_state.dtype in (torch.float16, torch.bfloat16, torch.float32)
                                           and initial_state.is_contiguous()))
        ):
            dense_split_schedule = ((8, 1, 2), (8, 1, 1))
    if dense_split_schedule is not None:
        return _fused_recurrent_gated_swila_bwd_scheduled(
            q, k, v, beta, pi_q, pi_k, r, u, initial_state, do, dht,
            log_decay, final_state, d_r_seq, scale, cu_seqlens, dense_split_schedule,
        )
    B, T, H, D_k_dim = q.shape
    D_v_dim = v.shape[-1]
    E_dim = beta.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Compute block sizes
    BK = max(triton.next_power_of_2(D_k_dim), 16)
    BE = triton.next_power_of_2(E_dim)
    NK = triton.cdiv(D_k_dim, BK)

    use_dense_d64e4_fastpath = _use_dense_d64e4_key_bwd_fastpath(
        q,
        k,
        v,
        beta,
        pi_q,
        pi_k,
        log_decay,
        cu_seqlens,
    )
    # Allocate only enough partials for the smallest tile in the selected
    # autotune table. The generic table still starts at BV=8.
    min_value_tile = (
        _DENSE_D64E4_BWD_MIN_VALUE_TILE
        if use_dense_d64e4_fastpath
        else 8
    )
    NV_max = triton.cdiv(D_v_dim, min_value_tile)

    assert NK == 1, f"NK > 1 not supported yet (D_k={D_k_dim}, BK={BK}, NK={NK})"

    # Allocate gradient tensors
    # dq and dk overlap in D_k across NV blocks -> need reduction.
    # Allocated with NV_max; pre_hook zeros before each autotune benchmark.
    dq = q.new_zeros(NV_max, B, T, H, D_k_dim, dtype=torch.float32)
    dk = q.new_zeros(
        NV_max,
        B,
        T,
        H,
        D_k_dim,
        dtype=torch.float32,
    )
    # Reverse and replay tune independently in the split path, so prediction
    # corrections use their own right-sized partials rather than assuming both
    # kernels select the same BV.
    dk_prediction = (
        q.new_zeros(NV_max, B, T, H, D_k_dim, dtype=torch.float32)
        if use_dense_d64e4_fastpath
        else None
    )
    # dv: NV blocks write to disjoint D_v ranges -> direct-write, no reduction needed.
    dv = v.new_empty(B, T, H, D_v_dim)
    # dbeta, dpi_q, dpi_k: NV blocks write to disjoint D_v ranges -> direct-write.
    dbeta = beta.new_empty(B, T, H, D_v_dim, E_dim)
    dpi_q = pi_q.new_empty(B, T, H, D_v_dim, E_dim)
    dpi_k = pi_k.new_empty(B, T, H, D_v_dim, E_dim)

    # Temporary storage for grad_u (used for grad_k correction in Phase 2)
    du_save = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)

    # All value blocks atomically reduce their local generalized-sigma terms
    # into one compact float32 buffer.  Its size is independent of NV and D_v.
    dlog_decay = torch.zeros_like(log_decay, dtype=torch.float32)
    dlog_decay_replay = (
        torch.zeros_like(log_decay, dtype=torch.float32)
        if use_dense_d64e4_fastpath
        else None
    )

    # Initial state gradient (N sequences, not B)
    if initial_state is not None:
        dh0 = q.new_empty(N, H, D_v_dim, E_dim, D_k_dim, dtype=torch.float32)
    else:
        dh0 = None

    is_varlen = cu_seqlens is not None
    if dht is not None and final_state is None:
        raise RuntimeError("final_state is required when dht is provided")

    if use_dense_d64e4_fastpath:
        def split_grid(meta):
            return (
                triton.cdiv(D_v_dim, meta["BV"]),
                N * H,
            )
        _fused_recurrent_gated_swila_dense_d64e4_bwd_reverse_kernel[
            split_grid
        ](
            q,
            k,
            beta,
            pi_q,
            log_decay,
            r,
            u,
            initial_state,
            do,
            dht,
            final_state,
            d_r_seq,
            dk,
            dv,
            dbeta,
            dpi_k,
            dlog_decay,
            du_save,
            dh0,
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
            USE_FINAL_STATE_GRADIENT=dht is not None,
            USE_AUX_LOSS_GRAD=d_r_seq is not None,
        )
        _fused_recurrent_gated_swila_dense_d64e4_bwd_replay_kernel[
            split_grid
        ](
            q,
            k,
            beta,
            pi_q,
            log_decay,
            r,
            u,
            initial_state,
            do,
            du_save,
            dq,
            dk_prediction,
            dpi_q,
            dlog_decay_replay,
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
        )
        # Autotune pre-hooks must be able to clear replay atomics without
        # erasing the reverse contribution, hence the separate compact buffer.
        dlog_decay.add_(dlog_decay_replay)
    else:
        # Generic fused reverse/replay fallback.
        def grid(meta):
            return (
                triton.cdiv(D_v_dim, meta["BV"]),
                NK,
                N * H,
            )
        fused_recurrent_gated_swila_bwd_kernel[grid](
            q, k, v, beta, pi_q, pi_k,
            log_decay,
            r, u,
            initial_state,
            do, dht, final_state,
            d_r_seq,
            dq, dk, dv, dbeta, dpi_q, dpi_k,
            dlog_decay, du_save, dh0,
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
            NK=NK,
            USE_INITIAL_STATE=initial_state is not None,
            USE_FINAL_STATE_GRADIENT=dht is not None,
            USE_AUX_LOSS_GRAD=d_r_seq is not None,
            IS_VARLEN=is_varlen,
        )

    _reverse_cumsum_log_decay_kernel[(NK, N * H)](
        dlog_decay,
        cu_seqlens,
        T,
        H=H,
        D_k=D_k_dim,
        E=E_dim,
        BK=BK,
        BE=BE,
        IS_VARLEN=is_varlen,
        num_warps=4,
    )

    # Reduce dq, dk across NV blocks (they overlap in D_k)
    dq = dq.sum(0).to(q.dtype)
    if use_dense_d64e4_fastpath:
        dk = dk.sum(0) + dk_prediction.sum(0)
    else:
        dk = dk.sum(0)
    dk = dk.to(k.dtype)
    dlog_decay = dlog_decay.to(log_decay)

    return dq, dk, dv, dbeta, dpi_q, dpi_k, dh0, dlog_decay
