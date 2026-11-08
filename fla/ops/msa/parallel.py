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
import triton.language as tl

from fla.ops.nsa.parallel import ParallelNSAFunction
from fla.utils import autotune_cache_kwargs, input_guard


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=['H', 'DI', 'BLOCK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'C', 't0'])
def msa_block_score_kernel(
    q,
    k,
    s,
    t0,
    T,
    C,
    H: tl.constexpr,
    DI: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
):
    # max over the causal keys of block `i_j` of the scores of the queries `t0 + [i_c * BT, (i_c + 1) * BT)`
    i_c, i_j, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_b = i_b.to(tl.int64)

    o_c = i_c * BT + tl.arange(0, BT)
    o_t = t0 + o_c
    o_d = tl.arange(0, BD)
    m_t = (o_c < C) & (o_t < T)
    m_d = o_d < DI
    p_s = s + ((i_b * C + o_c) * H + i_h) * NB + i_j

    b_max = tl.full([BT], float('-inf'), dtype=tl.float32)
    # blocks starting after the last query of the tile are invisible
    if i_j * BLOCK <= t0 + i_c * BT + BT - 1:
        b_q = tl.load(q + ((i_b * T + o_t[:, None]) * H + i_h) * DI + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.)
        for i_s in range(0, BLOCK, BS):
            o_s = i_j * BLOCK + i_s + tl.arange(0, BS)
            m_s = o_s < T
            b_k = tl.load(k + (i_b * T + o_s[None, :]) * DI + o_d[:, None], mask=m_d[:, None] & m_s[None, :], other=0.)
            # fp32 scores without TF32, as in the reference, since they only feed the selection
            b_x = tl.dot(b_q, b_k, allow_tf32=False)
            b_x = tl.where((o_t[:, None] >= o_s[None, :]) & m_s[None, :], b_x, float('-inf'))
            b_max = tl.maximum(b_max, tl.max(b_x, 1))
    tl.store(p_s, b_max, mask=m_t)


def msa_block_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, block_size: int, t0: int, C: int) -> torch.Tensor:
    """Max-pooled causal block scores of the queries `[t0, t0 + C)`, of shape `[B, C, H, NB]` in fp32."""
    B, T, H, DI = q_idx.shape
    C = min(C, T - t0)
    NB = triton.cdiv(T, block_size)
    s = q_idx.new_empty(B, C, H, NB, dtype=torch.float)

    def grid(meta): return (triton.cdiv(C, meta['BT']), NB, B * H)
    msa_block_score_kernel[grid](
        q=q_idx,
        k=k_idx,
        s=s,
        t0=t0,
        T=T,
        C=C,
        H=H,
        DI=DI,
        NB=NB,
        BLOCK=block_size,
        BD=max(16, triton.next_power_of_2(DI)),
        BS=min(block_size, 64),
    )
    return s


@input_guard
@torch.no_grad()
def msa_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    block_size: int = 128,
    topk: int = 16,
    local_blocks: int = 1,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int | None = None,
) -> torch.LongTensor:
    r"""
    Lightning indexer of MSA, with the same interface and outputs as `naive_msa_indexer`.

    A Triton kernel computes the causal block scores (the max over each key block of the fp32 dot products) directly,
    so the `[H, T, T]` token scores are never materialized. The local blocks and the top-k are then applied with PyTorch
    over chunks of `chunk_size` queries.

    Args:
        q_idx (torch.Tensor):
            Index queries of shape `[B, T, H, DI]`, one index head per key/value head.
        k_idx (torch.Tensor):
            Index keys of shape `[B, T, DI]`, shared across index heads.
        block_size (int, Optional):
            Number of keys per block, a power of 2 of at least 16. Default: 128.
        topk (int, Optional):
            Number of blocks selected per query and index head. Default: 16.
        local_blocks (int, Optional):
            Number of blocks ending at the block of the query that are always selected. Default: 1.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs, with blocks counted from the start of
            every sequence. When provided, the batch size must be 1. Default: `None`.
        chunk_size (int, Optional):
            Number of queries scored at a time. If `None`, chosen so that one chunk holds about `2^26` block scores.
            Default: `None`.

    Returns:
        indices (torch.LongTensor):
            Selected block indices of shape `[B, T, H, topk]`, left-packed and padded with `-1`.
    """
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError(f"`block_size` must be a power of 2 of at least 16, got {block_size}.")
    B, T, H, _ = q_idx.shape
    if cu_seqlens is not None and B != 1:
        raise ValueError("The batch size must be 1 when `cu_seqlens` is provided.")
    if chunk_size is None:
        chunk_size = max(64, (1 << 26) // max(1, B * H * triton.cdiv(T, block_size)))
    indices = q_idx.new_full((B, T, H, topk), -1, dtype=torch.long)
    seqs = [(0, T)] if cu_seqlens is None else list(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()))
    for bos, eos in seqs:
        L = eos - bos
        if L == 0:
            continue
        S = min(topk, triton.cdiv(L, block_size))
        q, k = q_idx[:, bos:eos], k_idx[:, bos:eos]
        for s in range(0, L, chunk_size):
            # [B, C, H, NB]
            block_scores = msa_block_scores(q, k, block_size, s, chunk_size)
            t = torch.arange(s, s + block_scores.shape[1], device=q_idx.device)
            if local_blocks > 0:
                local = (t // block_size)[:, None] - torch.arange(local_blocks, device=q_idx.device)[None, :]
                block_scores.scatter_(-1, local.clamp(min=0)[None, :, None].expand(B, -1, H, -1), float('inf'))
            values, idx = block_scores.topk(S, dim=-1)
            indices[:, bos + s:bos + s + len(t), :, :S] = idx.masked_fill(values == float('-inf'), -1)
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
