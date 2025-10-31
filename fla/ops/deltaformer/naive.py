# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

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


def naive_causal_attention_bhtd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    B, H, T, D = q.shape
    qk_scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1, -2)) * qk_scale  # [B, H, T, T]
    causal_mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
    o = torch.matmul(attn_weights, v)  # [B, H, T, D]

    return o


def naive_deltaformer_attn_head_first(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Naive reference implementation of DeltaFormer attention for head-first format.

    Two-stage process:
    1. Computes u[i] = v[i] - beta[i] * sum_{j<i} softmax(q[i] @ k[:i]^T) @ u[:i]
    2. Applies causal attention: o = causal_attn(q, k, u)

    Args:
        q: [B, H, T, D]
        k: [B, H, T, D]
        v: [B, H, T, D]
        beta: [B, H, T] or None (defaults to ones)

    Returns:
        o: [B, H, T, D]
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q,k,v must be [B,H,T,D]"
    B, H, T, D = q.shape
    assert k.shape == (B, H, T, D) and v.shape == (B, H, T, D)
    orig_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    if beta is None:
        betaf = torch.ones((B, H, T), dtype=torch.float32, device=q.device)
    else:
        assert beta.shape == (B, H, T)
        betaf = beta.float()

    qk_scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * qk_scale
    probs = tril_softmax(scores, strict=True)  # [B,H,T,T] float32

    u_list = []
    for t in range(T):
        if t == 0:
            u_t = vf[:, :, t, :]
        else:
            w = probs[:, :, t, :t]  # [B,H,t]
            u_prev = torch.stack(u_list, dim=-2)  # [B,H,t,D]
            weighted_sum = (w.unsqueeze(-1) * u_prev).sum(dim=-2)  # [B,H,D]
            u_t = vf[:, :, t, :] - betaf[:, :, t].unsqueeze(-1) * weighted_sum
        u_list.append(u_t)
    u = torch.stack(u_list, dim=2)  # [B,H,T,D]

    o = naive_causal_attention_bhtd(q, k, u.to(orig_dtype))
    return o.to(orig_dtype)


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

    q_bhtd = q.transpose(1, 2)  # [B, T, H, D] -> [B, H, T, D]
    k_bhtd = k.transpose(1, 2)  # [B, T, H, D] -> [B, H, T, D]
    v_bhtd = v.transpose(1, 2)  # [B, T, H, D] -> [B, H, T, D]

    if beta is not None:
        assert beta.shape == (B, T, H)
        beta_bhtd = beta.transpose(1, 2)  # [B, T, H] -> [B, H, T]
    else:
        beta_bhtd = None

    o_bhtd = naive_deltaformer_attn_head_first(q_bhtd, k_bhtd, v_bhtd, beta_bhtd)

    o_bthd = o_bhtd.transpose(1, 2)  # [B, H, T, D] -> [B, T, H, D]

    return o_bthd


__all__ = [
    'naive_deltaformer_attn',
    'tril_softmax',
]
