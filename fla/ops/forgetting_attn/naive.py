# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def naive_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
):
    """
    Reference PyTorch implementation of forgetting attention.

    Args:
        q: [B, T, HQ, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        g: [B, T, HQ]
        scale: float, optional. If None, defaults to 1 / sqrt(D)

    Returns:
        output: [B, T, HQ, D]
    """
    _, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H

    if scale is None:
        scale = D ** -0.5

    gc = g.float().cumsum(1)
    mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=q.device))

    ref = torch.einsum("bqhd,bkhd->bhqk", q.float() * scale, repeat(k, "b t h d -> b t (h g) d", g=G).float())
    ref = ref + rearrange(gc, "b t h -> b h t 1") - rearrange(gc, "b t h -> b h 1 t")
    ref = ref.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float('inf'))
    ref = torch.einsum("bhqk,bkhd->bqhd", F.softmax(ref, dim=-1), repeat(v, "b t h d -> b t (h g) d", g=G).float())

    return ref
