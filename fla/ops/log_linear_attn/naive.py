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


def naive_log_linear_attn(q, k, v, g, level_scales):
    H = construct_H_matrix(g.permute(0, 2, 1), level_scales.permute(0, 2, 3, 1))
    M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
    return torch.einsum("bhlc,bchp->blhp", M, v)
