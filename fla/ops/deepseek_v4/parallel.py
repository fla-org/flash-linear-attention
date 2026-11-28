# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit
def _online_update(b_q, b_k, m_s, b_m, b_l, b_acc, scale):
    # one block of keys: `b_k` of shape [BC, BD] with validity `m_s` of shape [BC]
    b_s = tl.dot(b_q, tl.trans(b_k)) * scale
    b_s = tl.where(m_s[None, :], b_s, float('-inf'))
    b_mn = tl.maximum(b_m, tl.max(b_s, 1))
    # keep the update finite while every key seen so far is masked
    b_mc = tl.where(b_mn == float('-inf'), 0., b_mn)
    b_r = exp(b_m - b_mc)
    b_p = exp(b_s - b_mc[:, None])
    b_acc = b_acc * b_r[:, None] + tl.dot(b_p.to(b_k.dtype), b_k)
    b_l = b_l * b_r + tl.sum(b_p, 1)
    return b_mn, b_l, b_acc


@triton.jit
def _grad_update(b_q, b_do, b_k, m_s, b_lse, b_delta, b_dq, scale):
    # one block of keys: returns the updated dq and the gradient of the block of keys/values of shape [BC, BD]
    b_s = tl.dot(b_q, tl.trans(b_k)) * scale
    b_p = tl.where(m_s[None, :], exp(b_s - b_lse[:, None]), 0.)
    b_dp = tl.dot(b_do, tl.trans(b_k))
    b_ds = b_p * (b_dp - b_delta[:, None])
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k) * scale
    b_dk = tl.dot(tl.trans(b_ds).to(b_q.dtype), b_q) * scale + tl.dot(tl.trans(b_p).to(b_do.dtype), b_do)
    return b_dq, b_dk


@triton.heuristics({
    'USE_COMPRESSED': lambda args: args['ckv'] is not None,
    'USE_INDICES': lambda args: args['indices'] is not None,
})
@triton.jit(do_not_specialize=['T', 'N'])
def parallel_v4_attn_fwd_kernel(
    q,
    kv,
    ckv,
    sinks,
    indices,
    o,
    lse,
    scale,
    T,
    N,
    HQ: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    M: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
    USE_COMPRESSED: tl.constexpr,
    USE_INDICES: tl.constexpr,
):
    i_t, i_h, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_b.to(tl.int64)
    o_h = i_h * BH + tl.arange(0, BH)
    o_d = tl.arange(0, BD)
    o_c = tl.arange(0, BC)
    m_h, m_d = o_h < HQ, o_d < D

    p_q = q + ((i_b * T + i_t) * HQ + o_h[:, None]) * D + o_d[None, :]
    b_q = tl.load(p_q, mask=m_h[:, None] & m_d[None, :], other=0.)
    b_m = tl.full([BH], float('-inf'), dtype=tl.float32)
    b_l = tl.zeros([BH], dtype=tl.float32)
    b_acc = tl.zeros([BH, BD], dtype=tl.float32)

    # sliding window: the last W tokens up to and including the query
    for i_c in range(0, W, BC):
        o_s = i_t - W + 1 + i_c + o_c
        m_s = (o_s >= 0) & (i_c + o_c < W)
        b_k = tl.load(kv + (i_b * T + o_s[:, None]) * D + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.)
        b_m, b_l, b_acc = _online_update(b_q, b_k, m_s, b_m, b_l, b_acc, scale)

    if USE_COMPRESSED:
        # compressed entries visible to the query
        NV = (i_t + 1) // M
        if USE_INDICES:
            for i_c in range(0, S, BC):
                b_i = tl.load(indices + (i_b * T + i_t) * S + i_c + o_c, mask=i_c + o_c < S, other=-1)
                m_s = (b_i >= 0) & (b_i < NV)
                b_k = tl.load(ckv + (i_b * N + b_i[:, None]) * D + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.)
                b_m, b_l, b_acc = _online_update(b_q, b_k, m_s, b_m, b_l, b_acc, scale)
        else:
            for i_c in range(0, NV, BC):
                b_i = i_c + o_c
                m_s = b_i < NV
                b_k = tl.load(ckv + (i_b * N + b_i[:, None]) * D + o_d[None, :], mask=m_s[:, None] & m_d[None, :], other=0.)
                b_m, b_l, b_acc = _online_update(b_q, b_k, m_s, b_m, b_l, b_acc, scale)

    # the sink takes part in the normalizer without a value
    b_sink = tl.load(sinks + o_h, mask=m_h, other=0.).to(tl.float32)
    b_mn = tl.maximum(b_m, b_sink)
    b_r = exp(b_m - b_mn)
    b_l = b_l * b_r + exp(b_sink - b_mn)
    b_o = b_acc * (b_r / b_l)[:, None]
    tl.store(o + ((i_b * T + i_t) * HQ + o_h[:, None]) * D + o_d[None, :], b_o.to(o.dtype.element_ty),
             mask=m_h[:, None] & m_d[None, :])
    tl.store(lse + (i_b * T + i_t) * HQ + o_h, b_mn + log(b_l), mask=m_h)


@triton.heuristics({
    'USE_COMPRESSED': lambda args: args['ckv'] is not None,
    'USE_INDICES': lambda args: args['indices'] is not None,
})
@triton.jit(do_not_specialize=['T', 'N'])
def parallel_v4_attn_bwd_kernel(
    q,
    kv,
    ckv,
    indices,
    lse,
    delta,
    do,
    dq,
    dkv,
    dckv,
    scale,
    T,
    N,
    HQ: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    M: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
    USE_COMPRESSED: tl.constexpr,
    USE_INDICES: tl.constexpr,
):
    i_t, i_h, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_b.to(tl.int64)
    o_h = i_h * BH + tl.arange(0, BH)
    o_d = tl.arange(0, BD)
    o_c = tl.arange(0, BC)
    m_h, m_d = o_h < HQ, o_d < D
    m_hd = m_h[:, None] & m_d[None, :]

    o_qh = ((i_b * T + i_t) * HQ + o_h[:, None]) * D + o_d[None, :]
    b_q = tl.load(q + o_qh, mask=m_hd, other=0.)
    b_do = tl.load(do + o_qh, mask=m_hd, other=0.)
    b_lse = tl.load(lse + (i_b * T + i_t) * HQ + o_h, mask=m_h, other=0.)
    b_delta = tl.load(delta + (i_b * T + i_t) * HQ + o_h, mask=m_h, other=0.)
    b_dq = tl.zeros([BH, BD], dtype=tl.float32)

    for i_c in range(0, W, BC):
        o_s = i_t - W + 1 + i_c + o_c
        m_s = (o_s >= 0) & (i_c + o_c < W)
        m_sd = m_s[:, None] & m_d[None, :]
        b_k = tl.load(kv + (i_b * T + o_s[:, None]) * D + o_d[None, :], mask=m_sd, other=0.)
        b_dq, b_dk = _grad_update(b_q, b_do, b_k, m_s, b_lse, b_delta, b_dq, scale)
        tl.atomic_add(dkv + (i_b * T + o_s[:, None]) * D + o_d[None, :], b_dk, mask=m_sd, sem='relaxed')

    if USE_COMPRESSED:
        NV = (i_t + 1) // M
        if USE_INDICES:
            for i_c in range(0, S, BC):
                b_i = tl.load(indices + (i_b * T + i_t) * S + i_c + o_c, mask=i_c + o_c < S, other=-1)
                m_s = (b_i >= 0) & (b_i < NV)
                m_sd = m_s[:, None] & m_d[None, :]
                b_k = tl.load(ckv + (i_b * N + b_i[:, None]) * D + o_d[None, :], mask=m_sd, other=0.)
                b_dq, b_dk = _grad_update(b_q, b_do, b_k, m_s, b_lse, b_delta, b_dq, scale)
                tl.atomic_add(dckv + (i_b * N + b_i[:, None]) * D + o_d[None, :], b_dk, mask=m_sd, sem='relaxed')
        else:
            for i_c in range(0, NV, BC):
                b_i = i_c + o_c
                m_s = b_i < NV
                m_sd = m_s[:, None] & m_d[None, :]
                b_k = tl.load(ckv + (i_b * N + b_i[:, None]) * D + o_d[None, :], mask=m_sd, other=0.)
                b_dq, b_dk = _grad_update(b_q, b_do, b_k, m_s, b_lse, b_delta, b_dq, scale)
                tl.atomic_add(dckv + (i_b * N + b_i[:, None]) * D + o_d[None, :], b_dk, mask=m_sd, sem='relaxed')

    tl.store(dq + o_qh, b_dq.to(dq.dtype.element_ty), mask=m_hd)


def _meta(q, indices):
    HQ, D = q.shape[2], q.shape[3]
    BH = max(16, min(64, triton.next_power_of_2(HQ)))
    BD = max(16, triton.next_power_of_2(D))
    S = indices.shape[-1] if indices is not None else 0
    return HQ, D, S, BH, BD


def parallel_v4_attn_fwd(q, kv, ckv, sinks, indices, window_size, compress_rate, scale):
    B, T = q.shape[:2]
    HQ, D, S, BH, BD = _meta(q, indices)
    N = ckv.shape[1] if ckv is not None else 0
    o = torch.empty_like(q)
    lse = q.new_empty(B, T, HQ, dtype=torch.float)
    parallel_v4_attn_fwd_kernel[(T, triton.cdiv(HQ, BH), B)](
        q=q, kv=kv, ckv=ckv, sinks=sinks, indices=indices, o=o, lse=lse, scale=scale, T=T, N=N,
        HQ=HQ, D=D, S=S, W=window_size, M=compress_rate or 1, BH=BH, BD=BD, BC=32,
    )
    return o, lse


def parallel_v4_attn_bwd(q, kv, ckv, sinks, indices, o, lse, do, window_size, compress_rate, scale):
    B, T = q.shape[:2]
    HQ, D, S, BH, BD = _meta(q, indices)
    N = ckv.shape[1] if ckv is not None else 0
    delta = (o.float() * do.float()).sum(-1)
    dq = torch.empty_like(q)
    dkv = torch.zeros_like(kv, dtype=torch.float)
    dckv = torch.zeros_like(ckv, dtype=torch.float) if ckv is not None else None
    parallel_v4_attn_bwd_kernel[(T, triton.cdiv(HQ, BH), B)](
        q=q, kv=kv, ckv=ckv, indices=indices, lse=lse, delta=delta, do=do, dq=dq, dkv=dkv, dckv=dckv, scale=scale,
        T=T, N=N, HQ=HQ, D=D, S=S, W=window_size, M=compress_rate or 1, BH=BH, BD=BD, BC=32,
    )
    # the sink has no value, so its logit only receives -p_sink * delta
    dsinks = -(torch.exp(sinks.float() - lse) * delta).sum((0, 1))
    return dq, dkv.to(kv.dtype), dckv.to(ckv.dtype) if ckv is not None else None, dsinks.to(sinks.dtype)


class ParallelV4AttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, kv, ckv, sinks, indices, window_size, compress_rate, scale):
        o, lse = parallel_v4_attn_fwd(q, kv, ckv, sinks, indices, window_size, compress_rate, scale)
        ctx.save_for_backward(q, kv, ckv, sinks, indices, o, lse)
        ctx.window_size, ctx.compress_rate, ctx.scale = window_size, compress_rate, scale
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, kv, ckv, sinks, indices, o, lse = ctx.saved_tensors
        dq, dkv, dckv, dsinks = parallel_v4_attn_bwd(
            q, kv, ckv, sinks, indices, o, lse, do, ctx.window_size, ctx.compress_rate, ctx.scale,
        )
        return dq, dkv, dckv, dsinks, None, None, None, None


def parallel_v4_attention(
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
    Triton kernel for the attention of DeepSeek-V4, with the same interface and outputs as `naive_v4_attention`.

    Every query attends, under one softmax with a learned sink per head, to the last `window_size` tokens of `kv` up to
    itself and to the compressed entries it can see (`w < (t + 1) // compress_rate`): the selected ones with `indices`
    (CSA), or all of them otherwise (HCA). Keys and values share `kv` and `compressed_kv`. The gradients of `kv` and
    `compressed_kv` are accumulated with fp32 atomics, so they are not bitwise deterministic.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, D]`.
        kv (torch.Tensor):
            Keys and values of shape `[B, T, D]`.
        compressed_kv (torch.Tensor, Optional):
            Compressed keys and values of shape `[B, N, D]`, or `None` for sliding-window attention only.
        sinks (torch.Tensor):
            Sink logits of shape `[HQ]`.
        window_size (int):
            Number of tokens up to and including the query in the sliding window.
        compress_rate (int, Optional):
            Number of tokens per compressed entry. Required with `compressed_kv`. Default: `None`.
        indices (torch.LongTensor, Optional):
            Selected compressed entries of shape `[B, T, S]`, padded with `-1` and without duplicates. Default: `None`.
        scale (float, Optional):
            Scale factor for attention scores. If `None`, defaults to `1 / sqrt(D)`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, D]`.
    """
    if compressed_kv is not None and compress_rate is None:
        raise ValueError("`compress_rate` is required with `compressed_kv`.")
    if indices is not None and compressed_kv is None:
        raise ValueError("`indices` select compressed entries, so `compressed_kv` is required.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ParallelV4AttentionFunction.apply(q, kv, compressed_kv, sinks, indices, window_size, compress_rate, scale)
