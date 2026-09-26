# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch


def tril_softmax(scores: torch.Tensor, strict: bool = True) -> torch.Tensor:
    """
    Row-wise causal softmax over strictly lower-triangular (j < i) positions.

    Args:
        scores: [B, H, T, T] raw attention scores (q @ k^T).
        strict: if True, mask out diagonal as well (strictly causal). Otherwise include diagonal.

    Returns:
        probs: [B, H, T, T] with probabilities on j < i (or j <= i if strict=False), zeros elsewhere.
    """
    T = scores.size(-1)
    device = scores.device
    i = torch.arange(T, device=device).view(1, 1, T, 1)
    j = torch.arange(T, device=device).view(1, 1, 1, T)
    if strict:
        mask = (j < i)
    else:
        mask = (j <= i)

    masked = scores.masked_fill(~mask, float('-inf'))
    max_per_row = masked.max(dim=-1, keepdim=True).values
    exp = (masked - max_per_row).exp()
    exp = exp.masked_fill(~mask, 0.0)
    denom = exp.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    probs = exp / denom
    return probs


def naive_deltaformer_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Naive reference implementation of DeltaFormer attention for sequence-first format.

    Args:
        q: [B, T, H, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        beta: [B, T, H] or None (defaults to ones)

    Returns:
        o: [B, T, H, D]
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q,k,v must be [B,T,H,D]"
    B, T, H, D = q.shape
    assert k.shape == (B, T, H, D) and v.shape == (B, T, H, D)
    orig_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    if beta is None:
        betaf = torch.ones((B, T, H), dtype=torch.float32, device=q.device)
    else:
        assert beta.shape == (B, T, H)
        betaf = beta.float()

    qk_scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('bthd,bshd->bhts', qf, kf) * qk_scale
    probs = tril_softmax(scores, strict=True)  # [B,H,T,T] float32

    u_list = []
    for t in range(T):
        if t == 0:
            u_t = vf[:, t]
        else:
            w = probs[:, :, t, :t].transpose(1, 2)  # [B,t,H]
            u_prev = torch.stack(u_list, dim=1)  # [B,t,H,D]
            weighted_sum = (w.unsqueeze(-1) * u_prev).sum(dim=1)  # [B,H,D]
            u_t = vf[:, t] - betaf[:, t].unsqueeze(-1) * weighted_sum
        u_list.append(u_t)
    u = torch.stack(u_list, dim=1)  # [B,T,H,D]

    scores = torch.einsum('bthd,bshd->bhts', q, k) * qk_scale
    causal_mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)  # [B,H,T,T]
    o = torch.einsum('bhts,bshd->bthd', attn_weights, u.to(orig_dtype))
    return o.to(orig_dtype)


__all__ = [
    'naive_deltaformer_attn',
    'tril_softmax',
]
