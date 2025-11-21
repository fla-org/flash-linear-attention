# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F


def naive_stickbreaking_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive stick-breaking attention reference implementation.

    Args:
        q, k, v: [B, T, H, D]
        scale: inverse temperature (1/sqrt(D))
    Returns:
        o: [B, T, H, D]
        rem: [B, T, H] (1 - sum of attention up to t)
    """
    _, T, _, D = q.shape
    orig_dtype = q.dtype
    if scale is None:
        scale = D ** -0.5

    logits = torch.einsum('bthd,bshd->bhts', q, k) * scale
    logits = logits.float()

    mask = torch.ones(T, T, device=q.device).triu(0).bool()  # exclude diagonal
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    log_z = F.logsigmoid(logits).masked_fill(mask, -1e5).to(orig_dtype)
    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0).to(orig_dtype)

    cum_weight = torch.ones(T, T, device=q.device).tril(-1)

    re_cum_log_beta = torch.einsum("bhij,jk->bhik", log_beta, cum_weight.to(log_beta))
    log_att = log_z + re_cum_log_beta
    att = log_att.exp()
    o = torch.einsum('bhts,bshd->bthd', att, v)
    rem = 1 - att.sum(dim=-1).transpose(1, 2)

    return o.to(orig_dtype), rem.to(orig_dtype)


__all__ = [
    'naive_stickbreaking_attn',
]
