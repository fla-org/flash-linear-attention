# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch


def naive_parallel_rebased(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    use_norm: bool = True,
) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = attn ** 2
    attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    if use_norm:
        z = attn.sum(-1)
        return o / (z[..., None] + 1e-6)
    else:
        return o
