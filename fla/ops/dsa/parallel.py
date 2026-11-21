# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Sparse attention of DeepSeek Sparse Attention (DSA) over precomputed top-k token indices. Every program handles one
# query token, one key/value head and a block of the query heads sharing it: it gathers the selected keys and values
# in chunks of BC tokens and runs an online softmax. The selected keys are scattered over the sequence, so the backward
# pass accumulates their gradients with fp32 atomics.

import torch
import triton
import triton.language as tl

from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
from fla.ops.utils import prepare_token_indices
from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages)
             for num_warps in [2, 4, 8] for num_stages in [1, 2, 3]],
    key=['BG', 'BK', 'BV', 'BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_dsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    indices,
    scale,
    cu_seqlens,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BG: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_v, i_bhg = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    NG = tl.cdiv(G, BG)
    i_b, i_h, i_g = i_bhg // (H * NG), i_bhg // NG % H, i_bhg % NG
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int64), tl.load(token_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        bos = i_b * T

    # query heads i_h * G + [i_g * BG, i_g * BG + BG) of the key/value head i_h
    o_g = i_g * BG + tl.arange(0, BG)
    m_g = o_g < G
    o_hq = i_h * G + o_g
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = o_k < K, o_v < V

    b_q = tl.load(q + ((bos + i_t) * HQ + o_hq[:, None]) * K + o_k[None, :], mask=m_g[:, None] & m_k[None, :], other=0.)
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BG, BV], dtype=tl.float32)
    b_m = tl.full([BG], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BG], dtype=tl.float32)
    for i_c in range(0, tl.cdiv(S, BC)):
        o_c = i_c * BC + tl.arange(0, BC)
        b_i = tl.load(indices + (bos + i_t) * S + o_c, mask=o_c < S, other=-1).to(tl.int64)
        # padding and keys after the query are skipped
        m_i = (b_i >= 0) & (b_i <= i_t)
        # [BK, BC]
        b_k = tl.load(k + ((bos + b_i[None, :]) * H + i_h) * K + o_k[:, None], mask=m_k[:, None] & m_i[None, :], other=0.)
        # [BC, BV]
        b_v = tl.load(v + ((bos + b_i[:, None]) * H + i_h) * V + o_v[None, :], mask=m_i[:, None] & m_v[None, :], other=0.)
        b_s = tl.where(m_i[None, :], tl.dot(b_q, b_k), float('-inf'))
        b_mn = tl.maximum(b_m, tl.max(b_s, 1))
        # rows without any visible key so far keep a finite reference
        b_ms = tl.where(b_mn == float('-inf'), 0., b_mn)
        b_r = exp(b_m - b_ms)
        b_p = exp(b_s - b_ms[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_mn

    b_o = b_o / b_acc[:, None]
    tl.store(o + ((bos + i_t) * HQ + o_hq[:, None]) * V + o_v[None, :], b_o.to(o.dtype.element_ty),
             mask=m_g[:, None] & m_v[None, :])
    if i_v == 0:
        tl.store(lse + (bos + i_t) * HQ + o_hq, b_m + log(b_acc), mask=m_g)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages)
             for num_warps in [2, 4, 8] for num_stages in [1, 2, 3]],
    key=['BG', 'BK', 'BV', 'BC'],
    restore_value=['dk', 'dv'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_dsa_bwd_kernel(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    dk,
    dv,
    indices,
    scale,
    cu_seqlens,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BG: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bhg = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    NG = tl.cdiv(G, BG)
    i_b, i_h, i_g = i_bhg // (H * NG), i_bhg // NG % H, i_bhg % NG
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int64), tl.load(token_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        bos = i_b * T

    o_g = i_g * BG + tl.arange(0, BG)
    m_g = o_g < G
    o_hq = i_h * G + o_g
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_k, m_v = o_k < K, o_v < V

    b_q = tl.load(q + ((bos + i_t) * HQ + o_hq[:, None]) * K + o_k[None, :], mask=m_g[:, None] & m_k[None, :], other=0.)
    b_do = tl.load(do + ((bos + i_t) * HQ + o_hq[:, None]) * V + o_v[None, :], mask=m_g[:, None] & m_v[None, :], other=0.)
    b_lse = tl.load(lse + (bos + i_t) * HQ + o_hq, mask=m_g, other=0.)
    b_delta = tl.load(delta + (bos + i_t) * HQ + o_hq, mask=m_g, other=0.)
    b_dq = tl.zeros([BG, BK], dtype=tl.float32)
    for i_c in range(0, tl.cdiv(S, BC)):
        o_c = i_c * BC + tl.arange(0, BC)
        b_i = tl.load(indices + (bos + i_t) * S + o_c, mask=o_c < S, other=-1).to(tl.int64)
        m_i = (b_i >= 0) & (b_i <= i_t)
        p_k = ((bos + b_i[:, None]) * H + i_h) * K + o_k[None, :]
        p_v = ((bos + b_i[:, None]) * H + i_h) * V + o_v[None, :]
        # [BC, BK], [BC, BV]
        b_k = tl.load(k + p_k, mask=m_i[:, None] & m_k[None, :], other=0.)
        b_v = tl.load(v + p_v, mask=m_i[:, None] & m_v[None, :], other=0.)
        # [BG, BC]
        b_s = tl.dot(b_q, tl.trans(b_k)) * scale
        b_p = tl.where(m_g[:, None] & m_i[None, :], exp(b_s - b_lse[:, None]), 0.)
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_dp - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
        tl.atomic_add(dk + p_k, tl.dot(tl.trans(b_ds), b_q.to(tl.float32)) * scale, mask=m_i[:, None] & m_k[None, :])
        tl.atomic_add(dv + p_v, tl.dot(tl.trans(b_p), b_do.to(tl.float32)), mask=m_i[:, None] & m_v[None, :])
    tl.store(dq + ((bos + i_t) * HQ + o_hq[:, None]) * K + o_k[None, :], (b_dq * scale).to(dq.dtype.element_ty),
             mask=m_g[:, None] & m_k[None, :])


def _blocks(G: int, K: int, V: int):
    # tl.dot needs at least 16 rows
    BG = max(16, min(64, triton.next_power_of_2(G)))
    return BG, triton.next_power_of_2(K), triton.next_power_of_2(V)


def parallel_dsa_fwd(q, k, v, indices, scale, cu_seqlens, token_indices):
    B, T, HQ, K = q.shape
    H, V, S = k.shape[2], v.shape[-1], indices.shape[-1]
    G = HQ // H
    BG, BK, BV = _blocks(G, K, V)
    BV = min(BV, 128)
    BC = 32
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=v.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    NT = B * T if cu_seqlens is None else token_indices.shape[0]
    grid = (T if cu_seqlens is None else NT, triton.cdiv(V, BV), B * H * triton.cdiv(G, BG))
    parallel_dsa_fwd_kernel[grid](
        q, k, v, o, lse, indices, scale, cu_seqlens, token_indices, T,
        H=H, HQ=HQ, G=G, K=K, V=V, S=S, BG=BG, BK=BK, BV=BV, BC=BC,
    )
    return o, lse


def parallel_dsa_bwd(q, k, v, o, lse, do, indices, scale, cu_seqlens, token_indices):
    B, T, HQ, K = q.shape
    H, V, S = k.shape[2], v.shape[-1], indices.shape[-1]
    G = HQ // H
    BG, BK, BV = _blocks(G, K, V)
    BC = 16
    delta = parallel_attn_bwd_preprocess(o, do)
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k, dtype=torch.float)
    dv = torch.zeros_like(v, dtype=torch.float)
    grid = (T if cu_seqlens is None else token_indices.shape[0], B * H * triton.cdiv(G, BG))
    parallel_dsa_bwd_kernel[grid](
        q, k, v, lse, delta, do, dq, dk, dv, indices, scale, cu_seqlens, token_indices, T,
        H=H, HQ=HQ, G=G, K=K, V=V, S=S, BG=BG, BK=BK, BV=BV, BC=BC,
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype)


class ParallelDSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, indices, scale, cu_seqlens):
        token_indices = prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
        o, lse = parallel_dsa_fwd(q, k, v, indices, scale, cu_seqlens, token_indices)
        ctx.save_for_backward(q, k, v, o, lse, indices)
        ctx.scale, ctx.cu_seqlens, ctx.token_indices = scale, cu_seqlens, token_indices
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse, indices = ctx.saved_tensors
        dq, dk, dv = parallel_dsa_bwd(q, k, v, o, lse, do, indices, ctx.scale, ctx.cu_seqlens, ctx.token_indices)
        return dq, dk, dv, None, None, None


@torch.compiler.disable
def parallel_dsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Sparse attention of DeepSeek Sparse Attention (DSA) over the top-k keys selected for every query, shared across
    all query heads, as in `naive_dsa` with given `indices`.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`. GQA and MQA are supported with group size `G = HQ // H`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        indices (torch.LongTensor):
            Selected key indices of shape `[B, T, topk]`, padded with `-1`, as returned by `naive_dsa_indexer`.
            Indices after the query position are ignored. For varlen inputs, they are offsets within each sequence.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs, consistent with the FlashAttention
            API. When provided, the batch size must be 1. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    HQ, H = q.shape[2], k.shape[2]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ParallelDSAFunction.apply(q, k, v, indices, scale, cu_seqlens)
