# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    causal: bool = True,
):
    """
    Reference PyTorch implementation of parallel attention that returns both output and max_logits.

    Args:
        q: [B, T, HQ, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        scale: float, optional. If None, defaults to 1 / sqrt(D)
        window_size: int, optional. If provided, each query at position i only attends to
            keys in [i - window_size + 1, i]. If None, full causal attention is used.
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

    # reshape q to separate groups: [B, T, HQ, D] -> [B, T, H, G, D]
    q = q.reshape(B, T, H, G, D)

    # compute attention scores via einsum: [B, H, G, T, T]
    # k is [B, T, H, D] — no group dim, so each group shares the same k
    scores = torch.einsum('bqhgd,bkhd->bhgqk', q, k) * scale

    # apply causal mask
    if causal:
        row_idx = torch.arange(T, device=q.device).unsqueeze(1)
        col_idx = torch.arange(T, device=q.device).unsqueeze(0)
        mask = col_idx > row_idx
        if window_size is not None:
            mask = mask | (row_idx - col_idx >= window_size)
        scores = scores.masked_fill(mask, float('-inf'))

    # max_logits: [B, H, G, T] -> [B, T, HQ]
    max_logits = scores.max(dim=-1).values.permute(0, 3, 1, 2).reshape(B, T, HQ)

    # compute output via einsum: [B, H, G, T, T] x [B, T, H, D] -> [B, T, H, G, D]
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhgqk,bkhd->bqhgd', attn_weights, v).reshape(B, T, HQ, D)

    return output, max_logits
