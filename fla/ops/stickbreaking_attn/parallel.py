# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.ops.utils.softplus import softplus2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, contiguous


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HQ', 'G', 'K', 'V', 'BT', 'BS', 'BK', 'BV', 'ATTEND_CURRENT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_stickbreaking_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    rem,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_rem = rem + bos * HQ + i_hq + o_q * HQ

    # [BT, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    # [BT], log2 of the stick left over by the keys visited so far
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # keys are visited nearest first, so a key block only needs the stick left over by the blocks after it.
    # the first loop covers the diagonal block, where the causal mask applies
    for i_s in range(0, BT, BS):
        # [BS]
        o_k = i_t * BT + BT - BS - i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)

        if ATTEND_CURRENT:
            m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        else:
            m_s = (o_q[:, None] > o_k[None, :]) & m_k[None, :]
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        # log2(1 - sigmoid(z)), zeroed for masked keys so they take no part of the stick
        b_lb = tl.where(m_s, -softplus2(b_s), 0.)
        # log2(A) = b_s + reverse_cumsum(b_lb) + b_acc: log2(sigmoid(z)), then the stick left by the nearer keys
        b_p = tl.where(m_s, exp2(b_s + tl.cumsum(b_lb, axis=1, reverse=True) + b_acc[:, None]), 0.)
        b_o += tl.dot(b_p.to(b_v.dtype), b_v)
        b_acc += tl.sum(b_lb, 1)

    # the blocks below the diagonal are fully visible, walked from the nearest one down to the first
    for i_s in range(BT, (i_t + 1) * BT, BS):
        # [BS]
        o_k = (i_t + 1) * BT - BS - i_s + tl.arange(0, BS)
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=o_d[:, None] < K, other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=o_v[None, :] < V, other=0.0)

        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_lb = -softplus2(b_s)
        b_p = exp2(b_s + tl.cumsum(b_lb, axis=1, reverse=True) + b_acc[:, None])
        b_o += tl.dot(b_p.to(b_v.dtype), b_v)
        b_acc += tl.sum(b_lb, 1)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_rem, exp2(b_acc).to(p_rem.dtype.element_ty), mask=m_q)


def parallel_stickbreaking_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attend_current: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = 64
    if check_shared_mem('hopper', q.device.index):
        BS = 64
        BV = min(128, max(16, triton.next_power_of_2(V)))
    elif check_shared_mem('ampere', q.device.index):
        BS = 32
        BV = min(64, max(16, triton.next_power_of_2(V)))
    else:
        BS = 32
        BV = min(32, max(16, triton.next_power_of_2(V)))
    BK = max(16, triton.next_power_of_2(K))
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    rem = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    grid = (NV, NT, B * HQ)
    parallel_stickbreaking_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        rem=rem,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        ATTEND_CURRENT=attend_current,
    )
    return o, rem


class StickBreakingAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, attend_current, cu_seqlens):
        o, rem = parallel_stickbreaking_attn_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            attend_current=attend_current,
            cu_seqlens=cu_seqlens,
        )
        return o.to(q.dtype), rem.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, drem):
        raise NotImplementedError(
            'Backward pass is not implemented yet, see https://github.com/fla-org/flash-linear-attention/issues/481',
        )


def parallel_stickbreaking_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attend_current: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Stick-breaking attention (https://arxiv.org/abs/2410.17980), forward pass only for now.

    Query `i` gives each visible key `j` the weight `A_ij = sigmoid(z_ij) * prod_l (1 - sigmoid(z_il))`,
    where `z_ij = scale * q_i k_j` and `l` runs over the visible keys after `j`,
    that is `j < l < i`, or `j < l <= i` with `attend_current`.
    Walking back from the query, each key takes a sigmoid share of the stick that the nearer keys left over.
    The weights are not normalized, so a query can leave part of its stick unused.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (float, Optional):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        attend_current (bool, Optional):
            Whether a query also attends to the key at its own position.
            If `False`, query `i` only sees keys `j < i`, and the first query outputs zeros. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
        rem (torch.Tensor):
            The stick each query leaves unused, `1 - sum_j A_ij`, of shape `[B, T, HQ]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    HQ, H = q.shape[2], k.shape[2]
    if H == 0 or HQ % H != 0:
        raise ValueError(f"The number of query heads ({HQ}) must be divisible by the number of key/value heads ({H}).")
    if q.shape[-1] > 256:
        raise ValueError(f"The key dimension ({q.shape[-1]}) can not be larger than 256.")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )

    o, rem = StickBreakingAttentionFunction.apply(q, k, v, scale, attend_current, cu_seqlens)
    return o, rem
