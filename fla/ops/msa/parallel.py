# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# MiniMax Sparse Attention (MSA, https://arxiv.org/abs/2606.13392) selects key blocks per GQA group with a lightning
# indexer and attends over them exactly. The main branch is the selection branch of NSA, so it runs on its kernels.

import torch
import triton

from fla.ops.nsa.parallel import ParallelNSAFunction


@torch.no_grad()
def msa_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    block_size: int = 128,
    topk: int = 16,
    local_blocks: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 128,
) -> torch.LongTensor:
    r"""
    Lightning indexer of MSA, as `naive_msa_indexer`, scoring `chunk_size` queries at a time so the `[T, T]` score
    matrix is never materialized.

    Returns:
        indices (torch.LongTensor):
            Selected block indices of shape `[B, T, H, topk]`, left-packed and padded with `-1`.
    """
    B, T, H, _ = q_idx.shape
    indices = q_idx.new_full((B, T, H, topk), -1, dtype=torch.long)
    seqs = [(0, T)] if cu_seqlens is None else list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()))
    for bos, eos in seqs:
        L = eos - bos
        NB = -(-L // block_size)
        S = min(topk, NB)
        k = k_idx[:, bos:eos].float()
        for s in range(0, L, chunk_size):
            e = min(s + chunk_size, L)
            # [B, H, C, L]
            scores = torch.einsum('bthd,bsd->bhts', q_idx[:, bos + s:bos + e].float(), k)
            t = torch.arange(s, e, device=q_idx.device)
            scores = scores.masked_fill(t[:, None] < torch.arange(L, device=q_idx.device)[None, :], float('-inf'))
            scores = torch.nn.functional.pad(scores, (0, NB * block_size - L), value=float('-inf'))
            block_scores = scores.view(B, H, e - s, NB, block_size).amax(-1)
            if local_blocks > 0:
                local = (t // block_size)[:, None] - torch.arange(local_blocks, device=q_idx.device)[None, :]
                block_scores.scatter_(-1, local.clamp(min=0).expand(B, H, -1, -1), float('inf'))
            values, idx = block_scores.topk(S, dim=-1)
            indices[:, bos + s:bos + e, :, :S] = idx.masked_fill(values == float('-inf'), -1).transpose(1, 2)
    return indices


def parallel_msa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    block_size: int = 128,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Main branch of MiniMax Sparse Attention (MSA): causal attention of every query head over the key blocks selected for
    its key/value head, as returned by `msa_indexer`.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]`. The group size `HQ // H` must be a power of 2 and at least 16.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        indices (torch.LongTensor):
            Selected block indices of shape `[B, T, H, S]`, deduplicated and padded with `-1`.
        block_size (int, Optional):
            Number of keys per block. Default: 128.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs, consistent with the FlashAttention
            API. When provided, the batch size must be 1. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    G = q.shape[2] // k.shape[2]
    if G < 16 or G & (G - 1):
        raise ValueError(f"The group size HQ // H must be a power of 2 and at least 16, got {G}.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # the NSA kernels take a power-of-2 number of blocks; extra slots are `-1` and skipped
    S = indices.shape[-1]
    if S & (S - 1):
        indices = torch.nn.functional.pad(indices, (0, triton.next_power_of_2(S) - S), value=-1)
    return ParallelNSAFunction.apply(q, k, v, indices, indices.shape[-1], block_size, scale, cu_seqlens)
