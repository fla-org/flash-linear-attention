# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
from einops import rearrange

from fla.ops.linear_attn.utils import normalize_output


def naive_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = 64
    q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size) * scale
    k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = ((
        q @ k.transpose(-1, -2)).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
        0
    )) @ v
    o = inter + intra
    if normalize:
        o = normalize_output(q * scale, k, o)
    return rearrange(o, 'b h n c d -> b (n c) h d')
