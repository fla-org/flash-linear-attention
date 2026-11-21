# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


"""Input-dependent SwiLA router and temporal reset-gate computations.

These helpers own no parameters. Their projection, cast and random-draw order
matches the layer's original forward path.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def compute_router_logits(layer, pi_q_conv_out, pi_k_conv_out):
    """Return raw Q/K logits, per-head inputs, and optional K/Q noise scales."""
    # Per-head router inputs also feed the noise and temporal gates.
    if layer.pi_proj_rank > 0:
        pi_q_input = pi_q_conv_out
        pi_k_input = pi_k_conv_out
        pi_q_input_head = rearrange(pi_q_input, '... (h d) -> ... h d', d=layer.head_pi_dim)
        pi_k_input_head = rearrange(pi_k_input, '... (h d) -> ... h d', d=layer.head_pi_dim)
    else:
        pi_q_input = rearrange(pi_q_conv_out, '... (h d) -> ... h d', d=layer.head_pi_dim)
        pi_k_input = rearrange(pi_k_conv_out, '... (h d) -> ... h d', d=layer.head_pi_dim)
        pi_q_input_head = pi_q_input
        pi_k_input_head = pi_k_input

    # Compute input-dependent noise only during training.
    # Noise always uses per-head input (pi_k_input_head: [B, T, H, head_pi_dim])
    # Formula: noise = randn() * softplus(W_noise(pi_input)), where the
    # Linear includes the inverse-softplus initialization bias.
    noise_std_k = None
    noise_std_q = None
    if layer.inject_noise and layer.training:
        noise_std_k = F.softplus(layer.W_noise_k(pi_k_input_head)).view(
            *pi_k_input_head.shape[:-1], layer.head_v_dim, layer.num_experts
        )
        noise_std_q = F.softplus(layer.W_noise_q(pi_q_input_head)).view(
            *pi_q_input_head.shape[:-1], layer.head_v_dim, layer.num_experts
        )
        # Preserve the K-then-Q noise draw order.
        noise_k = torch.randn_like(noise_std_k) * noise_std_k
        noise_q = torch.randn_like(noise_std_q) * noise_std_q
    else:
        noise_k = None
        noise_q = None

    # Pass raw logits to the recurrence, which applies softmax internally.
    if layer.pi_proj_rank > 0:
        pi_q_raw = layer.pi_q_proj(pi_q_input).view(
            *pi_q_input.shape[:-1], layer.num_heads, layer.head_v_dim, layer.num_experts)
        pi_k_raw = layer.pi_k_proj(pi_k_input).view(
            *pi_k_input.shape[:-1], layer.num_heads, layer.head_v_dim, layer.num_experts)
    else:
        pi_q_raw = layer.pi_q_proj(pi_q_input).view(
            *pi_q_input.shape[:-1], layer.head_v_dim, layer.num_experts)
        pi_k_raw = layer.pi_k_proj(pi_k_input).view(
            *pi_k_input.shape[:-1], layer.head_v_dim, layer.num_experts)
    pi_q = pi_q_raw
    pi_k = pi_k_raw

    # Add pre-computed noise to pi_k and pi_q for load balancing (only during training)
    if noise_k is not None:
        pi_k = pi_k + noise_k
    if noise_q is not None:
        pi_q = pi_q + noise_q

    return pi_q, pi_k, pi_q_input_head, pi_k_input_head, noise_std_k, noise_std_q


def compute_temporal_gates(layer, pi_q_input_head, pi_k_input_head):
    """Return raw KV/Q reset logits and optional noise scales, without sigmoid."""
    # Compute gate values for temporal reset gates
    # We pass raw logits (not sigmoid) to kernels for numerically stable logsigmoid
    gate_noise_std_kv = None
    gate_noise_std_q = None
    if layer.temporal:
        # Gate inputs: always per-head [B, T, H, head_pi_dim]
        gate_kv_input = pi_k_input_head
        gate_kv_raw = layer.gate_kv_proj(gate_kv_input)  # [B, T, H, gate_out_dim]
        gate_q_input = pi_q_input_head
        gate_q_raw = layer.gate_q_proj(gate_q_input)     # [B, T, H, gate_out_dim]

        # Add learned noise to gate logits (same pattern as pi noise injection)
        if layer.inject_gate_noise and layer.training:
            gate_noise_std_kv = F.softplus(layer.W_noise_gate_kv(gate_kv_input))
            gate_kv_raw = gate_kv_raw + torch.randn_like(gate_noise_std_kv) * gate_noise_std_kv
            gate_noise_std_q = F.softplus(layer.W_noise_gate_q(gate_q_input))
            gate_q_raw = gate_q_raw + torch.randn_like(gate_noise_std_q) * gate_noise_std_q

        # gate_out_dim = D_v -> reshape to [B, T, H, D_v, 1]
        gate_kv_raw = gate_kv_raw.unsqueeze(-1)
        gate_q_raw = gate_q_raw.unsqueeze(-1)

        # Pass raw logits to kernels (they compute logsigmoid internally)
        gate_kv = gate_kv_raw
        gate_q = gate_q_raw
    else:
        gate_kv = None
        gate_q = None

    return gate_kv, gate_q, gate_noise_std_kv, gate_noise_std_q
