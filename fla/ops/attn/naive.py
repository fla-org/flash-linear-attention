# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F


def naive_parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
):
    """
    Reference PyTorch implementation of parallel attention that returns both output and max_logits.

    Args:
        q: [B, T, HQ, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        scale: float, optional. If None, defaults to 1 / sqrt(D)
        causal: bool, default True

    Returns:
        output: [B, T, HQ, D]
        max_logits: [B, T, HQ]
    """
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H

    if scale is None:
        scale = D ** -0.5

    # Reshape q to separate heads and groups
    q = q.reshape(B, T, H, G, D)  # [B, T, H, G, D]

    # Repeat k and v to match groups: [B, T, H, D] -> [B, T, H, G, D]
    k = k.unsqueeze(3).expand(B, T, H, G, D)  # Expand along group dimension
    v = v.unsqueeze(3).expand(B, T, H, G, D)

    # Reshape to treat each (B, H, G,) as a separate head
    q_flat = q.reshape(B * H * G, T, D)  # [B*H*G, T, D]
    k_flat = k.reshape(B * H * G, T, D)  # [B*H*G, T, D]
    v_flat = v.reshape(B * H * G, T, D)  # [B*H*G, T, D]

    # Compute attention scores: [B*H*G, T, T]
    scores = torch.bmm(q_flat, k_flat.transpose(1, 2)) * scale

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

    # Compute max_logits (max over key dimension): [B*H*G, T]
    max_logits_flat = scores.max(dim=-1).values
    max_logits = max_logits_flat.reshape(B, T, HQ)  # [B, T, HQ]

    # Compute attention weights and output
    attn_weights = F.softmax(scores, dim=-1)  # [B*H*G, T, T]
    output_flat = torch.bmm(attn_weights, v_flat)  # [B*H*G, T, D]
    output = output_flat.reshape(B, T, HQ, D)  # [B, T, HQ, D]

    return output, max_logits
