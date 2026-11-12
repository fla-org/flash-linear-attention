# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0

"""Precision-oriented ``naive_chunk_linear_attn`` implementation."""

from __future__ import annotations

import torch


@torch.compiler.disable
def naive_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Evaluate the upstream naive-chunk API on Ascend.

    This is the accuracy-oriented chunk-formula path, not an alias of the
    optimized ``chunk`` kernel.  The chunk materialization keeps each row
    independent of the recurrent state scan, which matches upstream's
    ``naive_chunk`` numerical contract and remains fully differentiable.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 [B, T, H, D] tensors")

    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    if T == 0:
        zero = q.sum() * 0 + k.sum() * 0 + v.sum() * 0
        return torch.zeros(B, 0, H, V, device=v.device, dtype=dtype) + zero

    if scale is None:
        scale = K**-0.5
    chunk_size = 64
    padding = (-T) % chunk_size
    if padding:
        q = torch.cat(
            [q, torch.zeros(B, padding, H, K, device=q.device, dtype=q.dtype)], dim=1
        )
        k = torch.cat(
            [k, torch.zeros(B, padding, H, K, device=k.device, dtype=k.dtype)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(B, padding, H, V, device=v.device, dtype=v.dtype)], dim=1
        )

    padded_t = T + padding
    n_chunks = padded_t // chunk_size
    q32 = q.to(torch.float32).reshape(B, n_chunks, chunk_size, H, K)
    k32 = k.to(torch.float32).reshape(B, n_chunks, chunk_size, H, K)
    v32 = v.to(torch.float32).reshape(B, n_chunks, chunk_size, H, V)
    q32 = q32.permute(0, 3, 1, 2, 4) * scale
    k32 = k32.permute(0, 3, 1, 2, 4)
    v32 = v32.permute(0, 3, 1, 2, 4)

    chunk_state = (k32.transpose(-1, -2) @ v32).cumsum(2)
    previous_state = torch.cat(
        [torch.zeros_like(chunk_state[:, :, :1]), chunk_state[:, :, :-1]], dim=2
    )
    inter = q32 @ previous_state
    causal = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
    )
    scores = (q32 @ k32.transpose(-1, -2)).masked_fill(~causal, 0)
    o = (inter + scores @ v32).permute(0, 2, 3, 1, 4)
    o = o.reshape(B, padded_t, H, V)[:, :T]

    if normalize:
        k_cum = k32.permute(0, 2, 3, 1, 4)
        k_cum = k_cum.reshape(B, padded_t, H, K)[:, :T].cumsum(1)
        o = o / (
            (q32.permute(0, 2, 3, 1, 4).reshape(B, padded_t, H, K)[:, :T] * k_cum).sum(
                -1, keepdim=True
            )
            + 1e-10
        )
    return o.to(dtype)
