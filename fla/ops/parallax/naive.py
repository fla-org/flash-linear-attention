# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    r"""
    Reference PyTorch implementation of Parallax (parameterized local
    linear attention, Algorithm 1 of https://arxiv.org/abs/2605.29157).

    With ``s1 = scale * (q @ k^T)`` and ``s2 = r @ k^T`` (note: ``s2`` is NOT
    scaled), ``p1 = softmax(s1)``, ``d1 = sum(p1)``, ``p2 = p1 * s2``,
    ``d2 = sum(p2)``, ``O1 = p1 @ v``, ``O2 = p2 @ v``, the output is

        out = O1 / d1 * (1 + d2 / d1) - O2 / d1

    so ``r`` injects a first-order multiplicative correction onto the
    softmax-weighted values. Computation is performed in fp32 and returned in
    the input dtype.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            secondary queries of shape `[B, T, HQ, D]` (same shape as `q`).
        k (torch.Tensor):
            keys of shape `[B, T, H, D]`. GQA is applied when `HQ` is divisible by `H`.
        v (torch.Tensor):
            values of shape `[B, T, H, D]`.
        scale (float, Optional):
            Scale applied to the `q @ k^T` logits only. If `None`, defaults to `1 / sqrt(D)`.
            Default: `None`.
        window_size (int, Optional):
            Sliding-window length. If provided, each query at position `i` only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        causal (bool, Optional):
            Whether to apply the causal mask. Default: `True`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H

    if scale is None:
        scale = D ** -0.5

    dtype = q.dtype
    q = q.float().reshape(B, T, H, G, D)
    r = r.float().reshape(B, T, H, G, D)
    k = k.float()
    v = v.float()

    # [B, H, G, T, T]
    s1 = torch.einsum('bqhgd,bkhd->bhgqk', q, k) * scale
    s2 = torch.einsum('bqhgd,bkhd->bhgqk', r, k)

    if causal:
        row_idx = torch.arange(T, device=q.device)[:, None]
        col_idx = torch.arange(T, device=q.device)[None, :]
        mask = col_idx > row_idx
        if window_size is not None:
            mask = mask | (row_idx - col_idx >= window_size)
        s1 = s1.masked_fill(mask[None, None, None], float('-inf'))

    m = s1.amax(dim=-1, keepdim=True)
    # Fully-masked rows cannot occur with a causal mask (the diagonal is always
    # in-window), but guard the `-inf` pivot defensively.
    m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
    p1 = (s1 - m).exp()                               # [B, H, G, Tq, Tk]
    d1 = p1.sum(dim=-1)                               # [B, H, G, Tq]
    p2 = p1 * s2
    d2 = p2.sum(dim=-1)                               # [B, H, G, Tq]
    o1 = torch.einsum('bhgqk,bkhd->bqhgd', p1, v)     # [B, Tq, H, G, D]
    o2 = torch.einsum('bhgqk,bkhd->bqhgd', p2, v)

    c_norm = (d2 / d1).permute(0, 3, 1, 2)            # [B, Tq, H, G]
    inv_d1 = (1.0 / d1).permute(0, 3, 1, 2)           # [B, Tq, H, G]
    out = o1 * inv_d1[..., None] * (1.0 + c_norm[..., None]) - o2 * inv_d1[..., None]
    return out.reshape(B, T, HQ, D).to(dtype)
