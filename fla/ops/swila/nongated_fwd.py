# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Nongated SwiLA forward Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.swila.atoms import prediction, readout, softmax_2d, state_update, sum_2d_axis1
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [2, 4, 8, 16, 32]
        for num_warps in ([1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8])
        for num_stages in [2, 3, 4]
    ],
    key=['D_k', 'D_v', 'E'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_swila_fwd_kernel(
    q,       # [B, T, H, D_k]
    k,       # [B, T, H, D_k]
    v,       # [B, T, H, D_v]
    beta,    # [B, T, H, D_v, E]
    pi_q,    # [B, T, H, D_v, E]
    pi_k,    # [B, T, H, D_v, E]
    o,       # [NK, B, T, H, D_v]
    h0,      # [N, H, D_v, E, D_k] initial state (optional)
    ht,      # [N, H, D_v, E, D_k] final state (optional)
    # Intermediate outputs for backward pass (optional)
    r_out,      # [B, T, H, D_v, E] - responsibilities
    u_out,      # [B, T, H, D_v, E] - error signals
    cu_seqlens,  # [N+1] cumulative sequence lengths (optional, for variable-length)
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    D_k: tl.constexpr,  # Key/query dimension
    D_v: tl.constexpr,  # Value dimension
    E: tl.constexpr,    # Number of experts
    BK: tl.constexpr,   # Block size for D_k
    BV: tl.constexpr,   # Block size for D_v
    BE: tl.constexpr,   # Block size for experts (power of 2)
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_INTERMEDIATES: tl.constexpr,
    IS_VARLEN: tl.constexpr = False,
):
    """
    Fused recurrent SwiLA forward kernel supporting multiple experts.

    Grid: (NV, NK, N*H) where NV = cdiv(D_v, BV), NK = cdiv(D_k, BK), N = num sequences
    Each thread block handles a block of V dimensions across all experts.
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

    # Value, key, and expert ranges for this block
    v_range = i_v * BV + tl.arange(0, BV).to(tl.int64)
    k_range = i_k * BK + tl.arange(0, BK).to(tl.int64)
    e_range = tl.arange(0, BE).to(tl.int64)
    mask_v = v_range < D_v
    mask_k = k_range < D_k
    mask_e = e_range < E

    # Clamped indices for safe pointer arithmetic (prevents out-of-bounds access)
    v_idx = tl.minimum(v_range, D_v - 1)
    k_idx = tl.minimum(k_range, D_k - 1)
    e_idx = tl.minimum(e_range, E - 1)

    # Initialize pointers for q, k, v (shape: [B, T, H, D])
    # Pointer to first element of this batch/head at t=0
    p_q = q + (bos * H + i_h) * D_k + k_idx
    p_k = k + (bos * H + i_h) * D_k + k_idx
    p_v = v + (bos * H + i_h) * D_v + v_idx

    # Pointers for beta, pi_q, pi_k (shape: [B, T, H, D_v, E])
    # Base offset for this batch/head/v_block
    base_pi = (bos * H + i_h) * D_v * E + v_idx[:, None] * E + e_idx[None, :]
    p_beta = beta + base_pi
    p_pi_q = pi_q + base_pi
    p_pi_k = pi_k + base_pi

    # Output pointer (shape: [NK, B, T, H, D_v])
    p_o = o + ((i_k * all_bt + bos) * H + i_h) * D_v + v_idx

    # Masks for pi/beta loading [BV, BE]
    mask_ve = mask_v[:, None] & mask_e[None, :]

    # Pointers for intermediate outputs (r, u) if storing
    # Only first K-block writes intermediates to avoid duplication
    if STORE_INTERMEDIATES:
        p_r = r_out + base_pi
    if STORE_INTERMEDIATES:
        p_u = u_out + base_pi

    # Initialize hidden states for all experts: [BV, BE, BK]
    # State shape in memory: [B, H, D_v, E, D_k]
    # We use 3D indexing with proper offset computation

    # Masks for 3D state loading/storing [BV, BE, BK]
    mask_vek = mask_v[:, None, None] & mask_e[None, :, None] & mask_k[None, None, :]

    # Initialize hidden states (always zeros first, then add initial state if provided)
    # This pattern matches FLA's approach and ensures consistent tensor allocation
    b_h = tl.zeros([BV, BE, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        # h0 shape: [B, H, D_v, E, D_k]
        # Offset: nh * D_v * E * D_k + v * E * D_k + e * D_k + k
        # Using 3D tensor indexing: [BV, BE, BK]
        h0_base = h0 + i_nh * D_v * E * D_k
        h0_offset = v_idx[:, None, None] * E * D_k + \
            e_idx[None, :, None] * D_k + \
            k_idx[None, None, :]
        b_h += tl.load(h0_base + h0_offset, mask=mask_vek, other=0.0).to(tl.float32)

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

        # Step 1: Compute predictions and errors for all experts
        # pred_e = k @ h_e = sum_k(k[k] * h_e[v, k]) -> [BV, E]
        b_pred = prediction(b_h, b_k)  # [BV, E]

        # Error: u_e = v - pred_e
        b_u = b_v[:, None] - b_pred  # [BV, E]

        b_u_sq = b_u * b_u  # [BV, E]

        # Step 2: Compute responsibilities r = softmax(pi_k - 0.5 * u^2) over experts
        b_r = softmax_2d(b_pi_k - 0.5 * b_u_sq, mask_e)  # [BV, BE]

        # Store intermediates for backward pass
        if STORE_INTERMEDIATES:
            tl.store(p_r, b_r.to(p_r.dtype.element_ty), mask=mask_ve)
        if STORE_INTERMEDIATES:
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=mask_ve)

        # Step 3: State update for all experts
        # delta_h_e = beta_e * r_e * u_e * k
        b_update_coef = b_beta * b_r * b_u  # [BV, BE]
        b_h = state_update(b_h, b_update_coef, b_k)  # [BV, BE, BK]

        # Step 4: Compute output
        # qS_e = q @ h_e = sum_k(q[k] * h_e[v, k]) -> [BV, BE]
        b_qS = readout(b_h, b_q)  # [BV, BE]

        # Use rho = softmax(pi_q) for output weighting: o = sum_e(rho_e * qS_e)
        b_pi_q_star = softmax_2d(b_pi_q, mask_e)  # [BV, BE]
        b_o = sum_2d_axis1(b_pi_q_star * b_qS)  # [BV]

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
        if STORE_INTERMEDIATES:
            p_r += H * D_v * E
        if STORE_INTERMEDIATES:
            p_u += H * D_v * E

    # Store final state
    if STORE_FINAL_STATE:
        # ht shape: [B, H, D_v, E, D_k]
        # Using 3D tensor indexing: [BV, BE, BK]
        ht_base = ht + i_nh * D_v * E * D_k
        ht_offset = v_idx[:, None, None] * E * D_k + \
            e_idx[None, :, None] * D_k + \
            k_idx[None, None, :]
        tl.store(ht_base + ht_offset, b_h.to(ht.dtype.element_ty), mask=mask_vek)


def fused_recurrent_swila_fwd(
    q: torch.Tensor,            # [B, T, H, D_k]
    k: torch.Tensor,            # [B, T, H, D_k]
    v: torch.Tensor,            # [B, T, H, D_v]
    beta: torch.Tensor,         # [B, T, H, D_v, E]
    pi_q: torch.Tensor,         # [B, T, H, D_v, E]
    pi_k: torch.Tensor,         # [B, T, H, D_v, E]
    scale: float,
    initial_state: torch.Tensor | None = None,  # [N, H, D_v, E, D_k]
    output_final_state: bool = False,
    output_intermediates: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
    """
    Forward pass for fused recurrent SwiLA with multiple experts.

    Args:
        q: Queries [B, T, H, D_k]
        k: Keys [B, T, H, D_k]
        v: Values [B, T, H, D_v]
        beta: Per-dimension learning rate [B, T, H, D_v, E]
        pi_q: Query-side log-prior responsibilities [B, T, H, D_v, E]
        pi_k: Key-side log-prior responsibilities [B, T, H, D_v, E]
        scale: Scaling factor for queries (typically 1/sqrt(D_k))
        initial_state: Initial hidden state [N, H, D_v, E, D_k], optional
        output_final_state: Whether to return final hidden state
        output_intermediates: Whether to return intermediates (r, u) for backward
        cu_seqlens: Cumulative sequence lengths [N+1] for variable-length inputs.
            When provided, B must be 1 and inputs are packed.

    Returns:
        o: Output [B, T, H, D_v]
        final_state: Final hidden state [N, H, D_v, E, D_k] if output_final_state else None
        intermediates: Tuple of (r, u) each [B, T, H, D_v, E] if output_intermediates else None
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

    # Allocate output tensor (NK=1 always, so no NK dimension needed)
    o = q.new_empty(B, T, H, D_v_dim)

    # Allocate final state if needed (N sequences, not B)
    if output_final_state:
        final_state = q.new_empty(N, H, D_v_dim, E_dim, D_k_dim, dtype=torch.float32)
    else:
        final_state = None

    # Allocate intermediates for backward if needed
    if output_intermediates:
        r_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
    else:
        r_out = None
    if output_intermediates:
        u_out = q.new_empty(B, T, H, D_v_dim, E_dim, dtype=torch.float32)
    else:
        u_out = None

    is_varlen = cu_seqlens is not None

    # Launch kernel (BV, num_warps, num_stages all autotuned)
    def grid(meta):
        return (triton.cdiv(D_v_dim, meta['BV']), NK, N * H)
    fused_recurrent_swila_fwd_kernel[grid](
        q, k, v,
        beta, pi_q, pi_k,
        o,
        initial_state,
        final_state,
        r_out,
        u_out,
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
        STORE_FINAL_STATE=final_state is not None,
        STORE_INTERMEDIATES=u_out is not None,
        IS_VARLEN=is_varlen,
    )

    # Package intermediates
    if output_intermediates:
        intermediates = (r_out, u_out)
    else:
        intermediates = None

    return o, final_state, intermediates
