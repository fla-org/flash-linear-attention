# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Naive references for the attention of DeepSeek-V4: sliding-window attention over the tokens, extended with compressed
# KV entries that pool `compress_rate` tokens each. Compressed Sparse Attention (CSA) pools every 4 tokens over
# overlapping windows and lets a lightning indexer pick the top-k entries per query; Heavily Compressed Attention (HCA)
# pools every 128 tokens and attends to all of them. Keys and values share one tensor, and every head has a learned
# attention sink.

import torch
import torch.nn.functional as F


def naive_compress(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    compress_rate: int,
    overlap: bool = False,
) -> torch.Tensor:
    r"""
    Compress every window of `compress_rate` tokens into one entry, `sum_j softmax_j(gate_j + bias_j) * kv_j`, with the
    softmax over the window for every channel. Trailing tokens of an incomplete window are dropped.

    With `overlap` (CSA), `kv` and `gate` hold two series of `2 * D` channels: entry `w` pools the first half of window
    `w - 1` together with the second half of window `w`, so windows have width `2 * compress_rate` and stride
    `compress_rate`.

    Args:
        kv (torch.Tensor):
            Inputs of shape `[B, T, D]`, or `[B, T, 2 * D]` with `overlap`.
        gate (torch.Tensor):
            Gate logits of the same shape as `kv`.
        position_bias (torch.Tensor):
            Bias of shape `[compress_rate, D]` (or `[compress_rate, 2 * D]`) added to the gates by position in the window.
        compress_rate (int):
            Number of tokens per compressed entry.
        overlap (bool, Optional):
            Whether to pool overlapping windows from two series. Default: `False`.

    Returns:
        compressed (torch.Tensor):
            Compressed entries of shape `[B, T // compress_rate, D]`, before normalization and rotary embeddings.
    """
    B, T, C = kv.shape
    N = T // compress_rate
    kv = kv[:, :N * compress_rate].float().view(B, N, compress_rate, C)
    gate = gate[:, :N * compress_rate].float().view(B, N, compress_rate, C) + position_bias.float()
    if overlap:
        D = kv.shape[-1] // 2
        # the first half of the window is the first series of the previous window, empty for the first entry
        prev_kv = F.pad(kv[..., :D], (0, 0, 0, 0, 1, 0))[:, :N]
        prev_gate = F.pad(gate[..., :D], (0, 0, 0, 0, 1, 0), value=float('-inf'))[:, :N]
        kv = torch.cat([prev_kv, kv[..., D:]], 2)
        gate = torch.cat([prev_gate, gate[..., D:]], 2)
    return (kv * gate.softmax(2)).sum(2)


def naive_v4_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: torch.Tensor,
    topk: int,
    compress_rate: int,
    scale: float | None = None,
) -> torch.LongTensor:
    r"""
    Lightning indexer of CSA: scores the compressed index keys by `sum_h w_idx[t, h] * relu(scale * q_idx[t, h] . k_idx[w])`
    and keeps the top-k entries that query `t` can see, those with `w < (t + 1) // compress_rate`.

    Args:
        q_idx (torch.Tensor):
            Index queries of shape `[B, T, HI, DI]`.
        k_idx (torch.Tensor):
            Compressed index keys of shape `[B, N, DI]`.
        w_idx (torch.Tensor):
            Per-head weights of shape `[B, T, HI]`.
        topk (int):
            Number of entries selected per query.
        compress_rate (int):
            Number of tokens per compressed entry.
        scale (float, Optional):
            Scale of the dot products. If `None`, defaults to `1 / sqrt(DI)`. Default: `None`.

    Returns:
        indices (torch.LongTensor):
            Selected entries of shape `[B, T, min(topk, N)]`, with `-1` for slots without a visible entry.
    """
    if scale is None:
        scale = q_idx.shape[-1] ** -0.5
    T, N = q_idx.shape[1], k_idx.shape[1]
    scores = F.relu(torch.einsum('bthd,bnd->bthn', q_idx.float(), k_idx.float())) * scale
    scores = (scores * w_idx.float()[..., None]).sum(2)
    visible = (torch.arange(T, device=q_idx.device) + 1) // compress_rate
    future = torch.arange(N, device=q_idx.device)[None, :] >= visible[:, None]
    scores = scores.masked_fill(future, float('-inf'))
    indices = scores.topk(min(topk, N), dim=-1).indices
    return indices.masked_fill(indices >= visible[None, :, None], -1)


def naive_v4_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    sinks: torch.Tensor,
    window_size: int,
    compress_rate: int | None = None,
    indices: torch.LongTensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    r"""
    Attention of DeepSeek-V4 with a single key/value head (keys and values share `kv`): every query attends to the last
    `window_size` tokens up to itself, to the compressed entries it can see (`w < (t + 1) // compress_rate`; with
    `indices`, only the selected ones, as in CSA; otherwise all of them, as in HCA), and to a learned sink per head,
    under one softmax.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, D]`.
        kv (torch.Tensor):
            Keys and values of shape `[B, T, D]`.
        compressed_kv (torch.Tensor, Optional):
            Compressed keys and values of shape `[B, N, D]`, or `None` for sliding-window attention only.
        sinks (torch.Tensor):
            Sink logits of shape `[HQ]`, which take part in the softmax without a value.
        window_size (int):
            Number of tokens up to and including the query in the sliding window.
        compress_rate (int, Optional):
            Number of tokens per compressed entry. Required with `compressed_kv`. Default: `None`.
        indices (torch.LongTensor, Optional):
            Selected compressed entries of shape `[B, T, S]`, padded with `-1`. Default: `None`.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(D)`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, D]`.
    """
    B, T, HQ, D = q.shape
    if scale is None:
        scale = D ** -0.5
    q, kv = q.float(), kv.float()
    i = torch.arange(T, device=q.device)
    # [B, HQ, T, T]
    s = torch.einsum('bthd,bsd->bhts', q, kv) * scale
    s = s.masked_fill(~((i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window_size)), float('-inf'))
    values = kv
    if compressed_kv is not None:
        N = compressed_kv.shape[1]
        sc = torch.einsum('bthd,bnd->bhtn', q, compressed_kv.float()) * scale
        visible = torch.arange(N, device=q.device)[None, :] < ((i + 1) // compress_rate)[:, None]
        if indices is not None:
            selected = torch.zeros(B, T, N + 1, dtype=torch.bool, device=q.device)
            selected.scatter_(-1, indices.masked_fill(indices < 0, N), True)
            visible = visible[None] & selected[..., :N]
        sc = sc.masked_fill(~visible.view(-1, 1, T, N), float('-inf'))
        s = torch.cat([s, sc], -1)
        values = torch.cat([kv, compressed_kv.float()], 1)
    s = torch.cat([s, sinks.float().view(1, HQ, 1, 1).expand(B, HQ, T, 1)], -1)
    p = s.softmax(-1)[..., :-1]
    return torch.einsum('bhts,bsd->bthd', p, values).to(q.dtype)
