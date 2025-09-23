# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Tuple

import torch


def _tril_mask(T: int, strict: bool = True, device=None) -> torch.Tensor:
    i = torch.arange(T, device=device).view(1, 1, T, 1)
    j = torch.arange(T, device=device).view(1, 1, 1, T)
    return (j < i) if strict else (j <= i)


def sb_attn_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    inv_temp: float,
    attend_current: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive stick-breaking attention reference implementation.

    Args:
        q, k, v: [B, T, H, D]
        inv_temp: inverse temperature (1/sqrt(D))
        attend_current: include diagonal when computing weights

    Returns:
        o: [B, T, H, D]
        rem: [B, T, H] (1 - sum of attention up to t)
    """
    B, T, H, D = q.shape
    orig_dtype = q.dtype

    logits = torch.einsum('bthd,bshd->bhts', q, k) * inv_temp
    logits = logits.float()

    if attend_current:
        mask = torch.ones(T, T, device=q.device).triu(1).bool()  # exclude diagonal
    else:
        mask = torch.ones(T, T, device=q.device).triu(0).bool()  # include diagonal
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    log_z = torch.nn.functional.logsigmoid(logits).masked_fill(mask, -1e5).to(orig_dtype)
    log_beta = torch.nn.functional.logsigmoid(-logits).masked_fill(mask, 0).to(orig_dtype)

    cum_weight = torch.ones(T, T, device=q.device).tril(-1)

    re_cum_log_beta = torch.einsum("bhij,jk->bhik", log_beta, cum_weight.to(log_beta))
    log_att = log_z + re_cum_log_beta
    att = log_att.exp()
    o = torch.einsum('bhts,bshd->bthd', att, v)
    rem = 1 - att.sum(dim=-1).transpose(1, 2)

    return o.to(orig_dtype), rem.to(orig_dtype)


__all__ = [
    'sb_attn_naive',
]
