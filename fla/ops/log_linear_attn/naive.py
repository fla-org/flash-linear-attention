# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import numpy as np
import torch


def segsum(x):
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool))
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def construct_level_mask(level, L):
    T = L.size(-1)
    if level == 0:
        return torch.diag_embed(L[..., level, :])

    indices = torch.cartesian_prod(torch.arange(T), torch.arange(T)).to(L.device)

    mask = torch.where(
        torch.logical_and(
            torch.logical_and(
                indices[:, 0] % (1 << level) >= (1 << (level - 1)),
                indices[:, 1] + (1 << (level - 1))
                >= indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
            ),
            indices[:, 1] < indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
        ).view(T, T),
        L[..., level, :].unsqueeze(-1).expand(*([-1] * (len(L.shape) - 2)), T, T),
        0,
    )

    return mask


def construct_H_matrix(a, L):
    T = a.size(-1)
    A = torch.exp(segsum(a))
    H = torch.zeros_like(A)
    for level in range(int(np.ceil(np.log2(T))) + 1):
        mask = construct_level_mask(level, L)
        H += A * mask
    return H


def naive_log_linear_attn(q, k, v, g, level_scales, scale: float | None = None):
    """Compute grouped log-linear attention with the same scaling as the chunk operator."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    num_groups, num_heads = k.shape[2], v.shape[2]
    if num_heads % num_groups != 0:
        raise ValueError("The number of value heads must be divisible by the number of query/key heads.")
    if num_groups != num_heads:
        q = q.repeat_interleave(num_heads // num_groups, dim=2)
        k = k.repeat_interleave(num_heads // num_groups, dim=2)
    H = construct_H_matrix(g.permute(0, 2, 1), level_scales.permute(0, 2, 3, 1))
    M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
    return torch.einsum("bhlc,bchp->blhp", M, v) * scale
