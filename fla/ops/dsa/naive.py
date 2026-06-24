# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import torch
import torch.nn.functional as F
from einops import repeat


def naive_dsa_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: torch.Tensor | None = None,
    topk: int = 2048,
    scale: float | None = None,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.LongTensor:
    r"""
    Lightning indexer of DeepSeek Sparse Attention (DSA), scoring causal keys and selecting the top-k.

    For a query at position `t` and a preceding token at position `s`,
    the index score is the weighted sum of head-wise similarities (MQA: a single index key is shared across heads):

        I[t, s] = \sum_j w_idx[t, j] * act(scale * q_idx[t, j] . k_idx[s])

    The `topk` highest-scoring causal keys are then kept per query.
    A positive `scale`, a monotone `activation`, and the weights do not change the ranking,
    so they only matter when the raw scores are consumed directly rather than for selection.

    Args:
        q_idx (torch.Tensor):
            Index queries of shape `[B, T, HI, DI]`, where `HI` is the number of indexer heads.
        k_idx (torch.Tensor):
            Index keys of shape `[B, T, DI]`. A single key per token is shared across all heads.
        w_idx (torch.Tensor, Optional):
            Per-head index weights of shape `[B, T, HI]`. If `None`, heads are summed with equal weight. Default: `None`.
        topk (int, Optional):
            Number of keys selected per query. Default: `2048`.
        scale (float, Optional):
            Scale factor applied to the dot product. If `None`, defaults to `1 / sqrt(DI)`. Default: `None`.
        activation (str, Optional):
            Name of the activation (a `torch.nn.functional` name) applied to head-wise similarities before aggregation,
            e.g. `'relu'`. If `None`, no activation is applied. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs.
            When provided, the batch size must be 1. Default: `None`.

    Returns:
        indices (torch.LongTensor):
            Selected key indices of shape `[B, T, topk]`, padded with `-1`.
            For varlen inputs, the indices are local offsets within each sequence.
    """
    if scale is None:
        scale = q_idx.shape[-1] ** -0.5
    q_idx, k_idx = q_idx.float(), k_idx.float()
    if w_idx is not None:
        w_idx = w_idx.float()

    def select(q_b, k_b, w_b):
        # q_b: [T, HI, DI], k_b: [T, DI], w_b: [T, HI] or None -> indices [T, topk]
        T = q_b.shape[0]
        # [HI, T, T]
        score = torch.einsum('m h d, n d -> h m n', q_b, k_b) * scale
        if activation is not None:
            score = getattr(F, activation)(score)
        # aggregate over indexer heads -> [T, T]
        logits = score.sum(0) if w_b is None else torch.einsum('h m n, m h -> m n', score, w_b)
        i_t = torch.arange(T, device=q_b.device)
        logits = logits.masked_fill(i_t[:, None] < i_t[None, :], float('-inf'))
        S = min(topk, T)
        values, indices = torch.topk(logits, S, dim=-1)
        # drop slots backed by invisible (`-inf`) positions, then pad up to `topk`
        indices = indices.masked_fill(torch.isinf(values), -1)
        if topk > S:
            indices = torch.cat([indices, indices.new_full((T, topk - S), -1)], dim=-1)
        return indices

    if cu_seqlens is None:
        indices = torch.stack([
            select(q_idx[i], k_idx[i], None if w_idx is None else w_idx[i]) for i in range(q_idx.shape[0])
        ], 0)
    else:
        assert q_idx.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        indices = q_idx.new_full((1, q_idx.shape[1], topk), -1, dtype=torch.long)
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
            w_b = None if w_idx is None else w_idx[0, bos:eos]
            indices[0, bos:eos] = select(q_idx[0, bos:eos], k_idx[0, bos:eos], w_b)
    return indices


def naive_dsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: torch.Tensor | None = None,
    indices: torch.LongTensor | None = None,
    topk: int = 2048,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.LongTensor]:
    r"""
    Naive reference for DeepSeek Sparse Attention (DSA).

    DSA scores every causal key with a lightweight indexer, keeps the top-k per query,
    and attends over those keys only. The selection is shared across all query heads.

    Reference:
        DeepSeek-V3.2 (https://arxiv.org/abs/2512.02556).

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`. GQA is supported with group size `G = HQ // H`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        q_idx (torch.Tensor):
            Index queries of shape `[B, T, HI, DI]`.
        k_idx (torch.Tensor):
            Index keys of shape `[B, T, DI]`, shared across indexer heads.
        w_idx (torch.Tensor, Optional):
            Per-head index weights of shape `[B, T, HI]`.
            If `None`, indexer heads are summed with equal weight. Default: `None`.
        indices (torch.LongTensor, Optional):
            Precomputed selection of shape `[B, T, topk]`, padded with `-1`.
            If provided, the indexer is skipped and `q_idx`/`k_idx`/`w_idx` are ignored. Default: `None`.
        topk (int, Optional):
            Number of keys selected per query. Default: `2048`.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs, consistent with the FlashAttention API.
            When provided, the batch size must be 1. Default: `None`.
        return_indices (bool, Optional):
            Whether to also return the selected indices. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
            When `return_indices=True`, also returns the selected indices of shape `[B, T, topk]`.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    q, k, v = (x.float() for x in (q, k, v))
    k, v = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v))

    if indices is None:
        indices = naive_dsa_indexer(q_idx, k_idx, w_idx, topk, activation='relu', cu_seqlens=cu_seqlens)
    elif q_idx is not None:
        warnings.warn("`indices` is provided, the lightning indexer is skipped")

    def attend(q_b, k_b, v_b, i_b):
        T = q_b.shape[0]
        # scatter selection into a boolean mask, with the dummy column `T` absorbing `-1` padding
        mask = q_b.new_zeros(T, T + 1, dtype=torch.bool)
        mask.scatter_(1, i_b.masked_fill(i_b < 0, T), True)
        mask = mask[:, :T]
        i_t = torch.arange(T, device=q_b.device)
        mask = mask & (i_t[:, None] >= i_t[None, :])
        # [HQ, T, T]
        score = torch.einsum('m h d, n h d -> h m n', q_b, k_b) * scale
        score = score.masked_fill(~mask[None], float('-inf'))
        p = torch.softmax(score, dim=-1)
        return torch.einsum('h m n, n h d -> m h d', p, v_b)

    o = torch.empty_like(v)
    if cu_seqlens is None:
        for i in range(q.shape[0]):
            o[i] = attend(q[i], k[i], v[i], indices[i])
    else:
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
            o[0, bos:eos] = attend(q[0, bos:eos], k[0, bos:eos], v[0, bos:eos], indices[0, bos:eos])

    o = o.to(dtype)
    return (o, indices) if return_indices else o
