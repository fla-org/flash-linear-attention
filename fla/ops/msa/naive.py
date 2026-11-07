# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def _sequences(T: int, cu_seqlens: torch.LongTensor | None):
    if cu_seqlens is None:
        return [(0, T)]
    return list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()))


def naive_msa_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    block_size: int = 128,
    topk: int = 16,
    local_blocks: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.LongTensor:
    r"""
    Lightning indexer of MiniMax Sparse Attention (MSA), selecting key blocks per GQA group.

    Every index head `g` (one per key/value head) scores the causal keys by `q_idx[t, g] . k_idx[s]`, max-pools the
    scores over blocks of `block_size` keys, forces the `local_blocks` blocks ending at the block of the query, and keeps
    the `topk` best blocks.

    Args:
        q_idx (torch.Tensor):
            Index queries of shape `[B, T, H, DI]`, one index head per key/value head.
        k_idx (torch.Tensor):
            Index keys of shape `[B, T, DI]`, shared across index heads.
        block_size (int, Optional):
            Number of keys per block. Default: 128.
        topk (int, Optional):
            Number of blocks selected per query and index head. Default: 16.
        local_blocks (int, Optional):
            Number of blocks ending at the block of the query that are always selected. Default: 1.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs, with blocks counted from the start of
            every sequence. When provided, the batch size must be 1. Default: `None`.

    Returns:
        indices (torch.LongTensor):
            Selected block indices of shape `[B, T, H, topk]`, left-packed and padded with `-1`.
    """
    B, T, H, _ = q_idx.shape
    indices = q_idx.new_full((B, T, H, topk), -1, dtype=torch.long)
    for bos, eos in _sequences(T, cu_seqlens):
        L = eos - bos
        NB = -(-L // block_size)
        # [B, H, L, L]
        scores = torch.einsum('bthd,bsd->bhts', q_idx[:, bos:eos].float(), k_idx[:, bos:eos].float())
        i = torch.arange(L, device=q_idx.device)
        scores = scores.masked_fill(i[None, None, :, None] < i[None, None, None, :], float('-inf'))
        scores = torch.nn.functional.pad(scores, (0, NB * block_size - L), value=float('-inf'))
        block_scores = scores.view(B, H, L, NB, block_size).amax(-1)
        if local_blocks > 0:
            local = (i // block_size)[:, None] - torch.arange(local_blocks, device=q_idx.device)[None, :]
            block_scores.scatter_(-1, local.clamp(min=0).expand(B, H, -1, -1), float('inf'))
        S = min(topk, NB)
        values, idx = block_scores.topk(S, dim=-1)
        indices[:, bos:eos, :, :S] = idx.masked_fill(values == float('-inf'), -1).transpose(1, 2)
    return indices


def naive_msa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    block_size: int = 128,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Naive reference for the main branch of MiniMax Sparse Attention (MSA): causal attention of every query head over
    the key blocks selected for its key/value head.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        indices (torch.LongTensor):
            Selected block indices of shape `[B, T, H, S]`, padded with `-1`.
        block_size (int, Optional):
            Number of keys per block. Default: 128.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    o = torch.empty(B, T, HQ, v.shape[-1], dtype=torch.float, device=q.device)
    for bos, eos in _sequences(T, cu_seqlens):
        L = eos - bos
        NB = -(-L // block_size)
        idx = indices[:, bos:eos]
        # [B, L, H, NB + 1], with the extra column absorbing the `-1` padding
        keep = torch.zeros(B, L, H, NB + 1, dtype=torch.bool, device=q.device)
        keep.scatter_(-1, idx.masked_fill(idx < 0, NB), True)
        keep = keep[..., :NB].repeat_interleave(block_size, -1)[..., :L]
        i = torch.arange(L, device=q.device)
        keep = keep & (i[None, :, None, None] >= i[None, None, None, :])
        keep = keep.repeat_interleave(G, dim=2)
        s = torch.einsum('bthd,bshd->bths', q[:, bos:eos].float(), k[:, bos:eos].float().repeat_interleave(G, 2)) * scale
        p = s.masked_fill(~keep, float('-inf')).softmax(-1)
        o[:, bos:eos] = torch.einsum('bths,bshd->bthd', p, v[:, bos:eos].float().repeat_interleave(G, 2))
    return o.to(q.dtype)
