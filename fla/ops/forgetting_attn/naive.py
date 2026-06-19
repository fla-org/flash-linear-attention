# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def naive_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
):
    """
    Reference PyTorch implementation of forgetting attention.

    Args:
        q: [B, T, HQ, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        g: [B, T, HQ]
        scale: float, optional. If None, defaults to 1 / sqrt(D)
        window_size: int, optional. If provided, each query only attends to keys
            in [i - window_size + 1, i]. If None, full causal attention is used.

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
    if window_size is not None:
        mask = mask & torch.triu(torch.ones((T, T), dtype=torch.bool, device=q.device), diagonal=-(window_size - 1))

    ref = torch.einsum("bqhd,bkhd->bhqk", q.float() * scale, repeat(k, "b t h d -> b t (h g) d", g=G).float())
    ref = ref + rearrange(gc, "b t h -> b h t 1") - rearrange(gc, "b t h -> b h 1 t")
    ref = ref.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float('inf'))
    ref = torch.einsum("bhqk,bkhd->bqhd", F.softmax(ref, dim=-1), repeat(v, "b t h d -> b t (h g) d", g=G).float())

    return ref
