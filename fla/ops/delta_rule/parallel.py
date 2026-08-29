# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.delta_rule.wy_fast import fwd_prepare_T
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT', 'K', 'V'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_fwd_kernel(
    q,
    k,
    v,
    beta,
    o,
    A,
    q_new,
    k_new,
    A_local,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_T = tl.arange(0, BT)
    m_b = o_t < T
    m_qk = m_b[:, None] & (o_k[None, :] < K)
    m_v = m_b[:, None] & (o_v[None, :] < V)
    m_T = m_b[:, None] & (o_T[None, :] < BT)
    p_q = q + i_bh * T*K + o_t[:, None] * K + o_k[None, :]
    p_k = k + i_bh * T*K + o_t[:, None] * K + o_k[None, :]
    p_v = v + i_bh * T*V + o_t[:, None] * V + o_v[None, :]
    b_q = (tl.load(p_q, mask=m_qk, other=0.0) * scale).to(p_q.dtype.element_ty)
    b_k = tl.load(p_k, mask=m_qk, other=0.0)
    b_v = tl.load(p_v, mask=m_v, other=0.0)

    p_T = A + i_bh * T * BT + o_t[:, None] * BT + o_T[None, :]
    b_T = tl.load(p_T, mask=m_T, other=0.0)

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0).to(b_q.dtype)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0).to(b_k.dtype)

    p_beta = beta + i_bh * T + o_t
    b_beta = tl.load(p_beta, mask=m_b, other=0.0)
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False).to(b_k.dtype)

    if OUTPUT_ATTENTIONS:
        p_a = A_local + i_bh * T * BT + o_t[:, None] * BT + o_T[None, :]
        tl.store(p_a, b_qkT.to(p_a.dtype.element_ty), mask=m_T)

    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False).to(b_k.dtype)
    p_o = o + i_bh * T*V + o_t[:, None] * V + o_v[None, :]
    tl.store(p_o, tl.dot(b_qkT, b_v).to(p_o.dtype.element_ty), mask=m_v)

    p_q_new = q_new + i_bh * T*K + o_t[:, None] * K + o_k[None, :]
    tl.store(p_q_new, (b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False)).to(p_q_new.dtype.element_ty), mask=m_qk)

    p_k_new = k_new + i_bh * T*K + o_t[:, None] * K + o_k[None, :]
    b_k_new = b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=False)
    tl.store(p_k_new, b_k_new.to(p_k_new.dtype.element_ty), mask=m_qk)


def chunk_transform_qk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    chunk_size: int,
    output_attentions: bool,
):
    B, H, T, K = k.shape
    BT = chunk_size
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    grid = (triton.cdiv(T, BT), B*H)
    V = v.shape[-1]
    A_local = torch.empty_like(A) if output_attentions else None
    chunk_transform_qk_fwd_kernel[grid](
        q,
        k,
        v,
        beta,
        o,
        A,
        q_new,
        k_new,
        A_local,
        scale=scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        OUTPUT_ATTENTIONS=output_attentions,
    )
    return q_new, k_new, o, A_local


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def save_intra_chunk_attn(
    A,
    A_local,
    T,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_l = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_t[None, :] < T)
    m_L = m_t[:, None] & (o_l[None, :] < BT)
    p_A = A + i_bh * T * T + o_t[:, None] * T + o_t[None, :]
    p_A_local = A_local + i_bh * T * BT + o_t[:, None] * BT + o_l[None, :]
    b_A_local = tl.load(p_A_local, mask=m_L, other=0.0)
    tl.store(p_A, b_A_local.to(p_A.dtype.element_ty), mask=m_A)


@triton.heuristics({
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_delta_rule_fwd_kernel(
    q,
    k,
    k2,  # original k
    v,
    beta,
    o,
    o_new,
    attn,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_q = m_t[:, None] & (o_k[None, :] < K)
    m_o = m_t[:, None] & (o_v[None, :] < V)
    p_q = q + i_bh * T*K + o_t[:, None] * K + o_k[None, :]

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, mask=m_q, other=0.0)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = o + i_bh * T*V + o_t[:, None] * V + o_v[None, :]
    b_o += tl.load(p_o, mask=m_o, other=0.0)

    # As opposed to Flashattention, this kernel requires scanning the KV blocks from right to left
    # Q block and K block have overlap.
    # masks required
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        o_s = offset + tl.arange(0, BS)
        m_k = (o_k[:, None] < K) & (o_s[None, :] < T)
        m_k2 = (o_s[:, None] < T) & (o_k[None, :] < K)
        m_v = (o_s[:, None] < T) & (o_v[None, :] < V)
        m_a = m_t[:, None] & (o_s[None, :] < T)
        p_k = k + i_bh * T*K + o_k[:, None] + o_s[None, :] * K
        p_k2 = k2 + i_bh * T*K + o_s[:, None] * K + o_k[None, :]
        p_v = v + i_bh * T*V + o_s[:, None] * V + o_v[None, :]
        p_beta = beta + i_bh * T + o_s
        # [BK, BS]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        # [BS]
        b_beta = tl.load(p_beta, mask=o_s < T, other=0.0)
        # [BT, BS]
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False)
        b_s = tl.where(m_s[:, None], b_s, 0)

        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (tl.load(p_k2, mask=m_k2, other=0.0) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False)

        if OUTPUT_ATTENTIONS:
            p_a = attn + i_bh * T * T + o_t[:, None] * T + o_s[None, :]
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), mask=m_a)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for offset in range(i_t * BT - BS, -BS, -BS):
        o_s = offset + tl.arange(0, BS)
        m_k = (o_k[:, None] < K) & (o_s[None, :] < T)
        m_k2 = (o_s[:, None] < T) & (o_k[None, :] < K)
        m_v = (o_s[:, None] < T) & (o_v[None, :] < V)
        m_a = m_t[:, None] & (o_s[None, :] < T)
        p_k = k + i_bh * T*K + o_k[:, None] + o_s[None, :] * K
        p_v = v + i_bh * T*V + o_s[:, None] * V + o_v[None, :]
        p_beta = beta + i_bh * T + o_s
        p_k2 = k2 + i_bh * T*K + o_s[:, None] * K + o_k[None, :]

        # [BK, BS]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        # [BS]
        b_beta = tl.load(p_beta, mask=o_s < T, other=0.0)
        # [BT, BS]
        b_s = (tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False))
        # [BT, BV]
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (tl.load(p_k2, mask=m_k2, other=0.0) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False).to(b_q.dtype)

        if OUTPUT_ATTENTIONS:
            p_a = attn + i_bh * T * T + o_t[:, None] * T + o_s[None, :]
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), mask=m_a)

    p_o_new = o_new + i_bh * T*V + o_t[:, None] * V + o_v[None, :]
    tl.store(p_o_new, b_o.to(p_o.dtype.element_ty), mask=m_o)


class ParallelDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, scale, output_attentions):
        B, H, T, K, V = *k.shape, v.shape[-1]
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 32
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        assert BT % BS == 0

        A = fwd_prepare_T(k, beta, BS)
        attn = q.new_zeros(B, H, T, T) if output_attentions else None
        q_new, k_new, o, A_local = chunk_transform_qk_fwd(
            q,
            k,
            v,
            beta,
            A,
            scale,
            BS,
            output_attentions,
        )

        num_stages = 3 if K <= 64 else 2
        num_warps = 4
        grid = (triton.cdiv(T, BT), B * H)
        o_new = torch.empty_like(o)

        parallel_delta_rule_fwd_kernel[grid](
            q=q_new,
            k=k_new,
            k2=k,
            v=v,
            beta=beta,
            o=o,
            o_new=o_new,
            attn=attn,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        if output_attentions:
            grid = (triton.cdiv(T, BS), B * H)
            save_intra_chunk_attn[grid](
                A=attn,
                A_local=A_local,
                T=T,
                BT=BS,
            )
        return o_new.to(q.dtype), attn

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, d_attn=None):
        raise NotImplementedError('Backward pass is not implemented. Stay tuned!')


def parallel_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    output_attentions: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape `[B, H, T, T]`. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    o, attn = ParallelDeltaRuleFunction.apply(q, k, v, beta, scale, output_attentions)
    return o, attn


def naive_delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    b, h, l, d_k = q.shape
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM]
        o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        o[:, :, i:i+BM] = o_i

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    return o, A
