# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, input_guard


@triton.heuristics({
    'USE_WEIGHTS': lambda args: args['w'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT, 'BS': BS}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64]
        for BS in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=['HI', 'DI', 'RELU', 'USE_WEIGHTS'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'C', 't0'])
def dsa_indexer_score_kernel(
    q,
    k,
    w,
    s,
    scale,
    t0,
    T,
    C,
    HI: tl.constexpr,
    DI: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    RELU: tl.constexpr,
    USE_WEIGHTS: tl.constexpr,
):
    # scores of the queries `t0 + [i_c * BT, (i_c + 1) * BT)` against the keys `[i_s * BS, (i_s + 1) * BS)`
    i_c, i_s, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_b.to(tl.int64)

    o_c = i_c * BT + tl.arange(0, BT)
    o_t = t0 + o_c
    o_s = i_s * BS + tl.arange(0, BS)
    o_d = tl.arange(0, BD)
    m_t = (o_c < C) & (o_t < T)
    m_s = o_s < T
    m_d = o_d < DI

    p_s = s + i_b * C * T + o_c[:, None] * T + o_s[None, :]
    m_st = m_t[:, None] & m_s[None, :]
    # key blocks entirely after the last query of the block are invisible
    if i_s * BS > t0 + i_c * BT + BT - 1:
        tl.store(p_s, tl.full([BT, BS], float('-inf'), dtype=tl.float32), mask=m_st)
        return

    # [BD, BS]
    b_k = tl.load(k + (i_b * T + o_s[None, :]) * DI + o_d[:, None], mask=m_d[:, None] & m_s[None, :], other=0.)
    b_acc = tl.zeros([BT, BS], dtype=tl.float32)
    for i_h in range(HI):
        # [BT, BD]
        b_q = tl.load(q + ((i_b * T + o_t[:, None]) * HI + i_h) * DI + o_d[None, :], mask=m_t[:, None] & m_d[None, :], other=0.)
        b_x = tl.dot(b_q, b_k) * scale
        if RELU:
            b_x = tl.maximum(b_x, 0.)
        if USE_WEIGHTS:
            b_w = tl.load(w + (i_b * T + o_t) * HI + i_h, mask=m_t, other=0.).to(tl.float32)
            b_x = b_x * b_w[:, None]
        b_acc += b_x
    b_acc = tl.where(o_t[:, None] >= o_s[None, :], b_acc, float('-inf'))
    tl.store(p_s, b_acc, mask=m_st)


def dsa_indexer_score(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: torch.Tensor | None,
    scale: float,
    relu: bool,
    t0: int = 0,
    C: int | None = None,
) -> torch.Tensor:
    """Causal index scores of the queries `[t0, t0 + C)` against all keys, of shape `[B, C, T]` in fp32."""
    B, T, HI, DI = q_idx.shape
    C = T - t0 if C is None else min(C, T - t0)
    s = q_idx.new_empty(B, C, T, dtype=torch.float)

    def grid(meta): return (triton.cdiv(C, meta['BT']), triton.cdiv(T, meta['BS']), B)
    dsa_indexer_score_kernel[grid](
        q=q_idx,
        k=k_idx,
        w=w_idx,
        s=s,
        scale=scale,
        t0=t0,
        T=T,
        C=C,
        HI=HI,
        DI=DI,
        BD=max(16, triton.next_power_of_2(DI)),
        RELU=relu,
    )
    return s


def _select(q_idx, k_idx, w_idx, topk, scale, relu, chunk_size):
    B, T = q_idx.shape[:2]
    S = min(topk, T)
    indices = q_idx.new_full((B, T, topk), -1, dtype=torch.long)
    for t0 in range(0, T, chunk_size):
        scores = dsa_indexer_score(q_idx, k_idx, w_idx, scale, relu, t0, chunk_size)
        values, idx = scores.topk(S, dim=-1)
        indices[:, t0:t0 + scores.shape[1], :S] = idx.masked_fill(torch.isinf(values), -1)
    return indices


@input_guard
@torch.no_grad()
def dsa_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    w_idx: torch.Tensor | None = None,
    topk: int = 2048,
    scale: float | None = None,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int | None = None,
) -> torch.LongTensor:
    r"""
    Lightning indexer of DeepSeek Sparse Attention (DSA), with the same interface and outputs as `naive_dsa_indexer`.

    The index scores `I[t, s] = \sum_j w_idx[t, j] * act(scale * q_idx[t, j] . k_idx[s])` are computed by a fused Triton
    kernel, tile by tile with the heads accumulated in registers, so the `[HI, T, T]` head-wise scores are never
    materialized. The top-k is taken with `torch.topk` over chunks of `chunk_size` queries, which bounds the memory to
    `[B, chunk_size, T]` scores.

    Args:
        q_idx (torch.Tensor):
            Index queries of shape `[B, T, HI, DI]`.
        k_idx (torch.Tensor):
            Index keys of shape `[B, T, DI]`, shared across indexer heads.
        w_idx (torch.Tensor, Optional):
            Per-head index weights of shape `[B, T, HI]`. If `None`, heads are summed with equal weight. Default: `None`.
        topk (int, Optional):
            Number of keys selected per query. Default: `2048`.
        scale (float, Optional):
            Scale factor applied to the dot product. If `None`, defaults to `1 / sqrt(DI)`. Default: `None`.
        activation (str, Optional):
            `'relu'` or `None`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs.
            When provided, the batch size must be 1. Default: `None`.
        chunk_size (int, Optional):
            Number of queries scored at a time. If `None`, chosen so that one chunk holds about `2^28` scores.
            Default: `None`.

    Returns:
        indices (torch.LongTensor):
            Selected key indices of shape `[B, T, topk]`, padded with `-1`.
            For varlen inputs, the indices are local offsets within each sequence.
    """
    if activation not in (None, 'relu'):
        raise ValueError(f"Only `activation='relu'` or `None` is supported, got {activation!r}.")
    if scale is None:
        scale = q_idx.shape[-1] ** -0.5
    B, T = q_idx.shape[:2]
    if chunk_size is None:
        chunk_size = max(64, (1 << 28) // max(1, B * T))
    relu = activation == 'relu'
    if w_idx is not None:
        w_idx = w_idx.contiguous()
    if cu_seqlens is None:
        return _select(q_idx, k_idx, w_idx, topk, scale, relu, chunk_size)

    if B != 1:
        raise ValueError("The batch size must be 1 when `cu_seqlens` is provided.")
    indices = q_idx.new_full((1, T, topk), -1, dtype=torch.long)
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        if eos > bos:
            indices[:, bos:eos] = _select(
                q_idx[:, bos:eos], k_idx[:, bos:eos], None if w_idx is None else w_idx[:, bos:eos],
                topk, scale, relu, chunk_size,
            )
    return indices
