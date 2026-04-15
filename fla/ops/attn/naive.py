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
    *,
    g: torch.Tensor | None = None,
    g_scale: float = 1.0,
    query_indices: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
):
    """
    Reference PyTorch implementation of parallel attention that returns both output and max_logits.

    Args:
        q: [B, TQ, HQ, D]
        k: [B, TK, H, D]
        v: [B, TK, H, D]
        scale: float, optional. If None, defaults to 1 / sqrt(D)
        window_size: int, optional. If provided, each query at position i only attends to
            keys in [i - window_size + 1, i]. If None, full causal attention is used.
        causal: bool, default True
        g: [B, TK, HQ], optional per-query-head gating logits
        g_scale: float, optional scaling applied to the gating bias term
        query_indices: [TQ] or [B, TQ], optional absolute query positions relative to keys
        sinks: [HQ], optional per-query-head sink logits

    Returns:
        output: [B, TQ, HQ, D]
        max_logits: [B, TQ, HQ]
    """
    B, TQ, HQ, D = q.shape
    TK = k.shape[1]
    H = k.shape[2]
    G = HQ // H

    if scale is None:
        scale = D ** -0.5

    # reshape q to separate groups: [B, TQ, HQ, D] -> [B, TQ, H, G, D]
    q = q.reshape(B, TQ, H, G, D)

    # compute attention scores via einsum: [B, H, G, TQ, TK]
    # k is [B, T, H, D] — no group dim, so each group shares the same k
    scores = torch.einsum('bqhgd,bkhd->bhgqk', q, k) * scale

    if query_indices is None:
        query_positions = torch.arange(TQ, device=q.device).unsqueeze(0).expand(B, TQ)
    else:
        if query_indices.ndim == 1:
            query_positions = query_indices.unsqueeze(0).expand(B, TQ)
        else:
            query_positions = query_indices
        query_positions = query_positions.to(device=q.device, dtype=torch.long)
        assert query_positions.shape == (B, TQ), "query_indices must have shape [TQ] or [B, TQ]"

    # apply causal mask
    if causal:
        row_idx = query_positions[:, :, None]
        col_idx = torch.arange(TK, device=q.device)[None, None, :]
        mask = col_idx > row_idx
        if window_size is not None:
            mask = mask | (row_idx - col_idx >= window_size)
        scores = scores.masked_fill(mask[:, None, None], float('-inf'))

    if g is not None:
        assert g.shape == (B, TK, HQ), "g must have shape [B, TK, HQ]"
        g_cumsum = g.float().cumsum(1).reshape(B, TK, H, G).permute(0, 2, 3, 1)
        g_query_idx = query_positions[:, None, None, :].expand(B, H, G, TQ)
        g_q = torch.gather(g_cumsum, dim=-1, index=g_query_idx)
        scores = scores + (g_q[..., None] - g_cumsum[..., None, :]) * g_scale

    # max_logits: [B, H, G, TQ] -> [B, TQ, HQ]
    max_logits = scores.max(dim=-1).values
    if sinks is not None:
        assert sinks.shape == (HQ,), "sinks must have shape [HQ]"
        sink_logits = sinks.reshape(H, G)[None, :, :, None]
        max_logits = torch.maximum(max_logits, sink_logits)

    if sinks is None:
        # compute output via einsum: [B, H, G, TQ, TK] x [B, TK, H, D] -> [B, TQ, H, G, D]
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhgqk,bkhd->bqhgd', attn_weights, v).reshape(B, TQ, HQ, D)
    else:
        probs_unnorm = torch.exp(scores - max_logits[..., None])
        sink_unnorm = torch.exp(sink_logits - max_logits)
        denom = probs_unnorm.sum(dim=-1) + sink_unnorm
        output = torch.einsum('bhgqk,bkhd->bqhgd', probs_unnorm, v)
        output = (output / denom.permute(0, 3, 1, 2)[..., None]).reshape(B, TQ, HQ, D)

    return output, max_logits.permute(0, 3, 1, 2).reshape(B, TQ, HQ)
