# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.linear_attn.utils import normalize_output
from fla.utils import input_guard


@triton.jit
def fused_recurrent_linear_attn_fwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    o,  # output [B, H, L, V]
    h0,
    ht,  # final hidden state [B, H, K, V]

    s_k_h,  # stride size: L * K
    s_v_h,  # stride size: L * V

    scale,
    B,  # batch size
    H,  # H
    T,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_linear_attn_bwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]

    do,  # gradient of output [B, H, L, V]
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dv,  # gradient of value [NK, B, H, L, V]
    h0,  # initial hidden state initialization [B, H, K, V]

    s_k_h,  # stride size: L * K
    s_v_h,  # stride size: L * V
    scale,  # K ** -0.5

    B,  # B
    H,  # H
    T,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV)

    p_dq = dq + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        b_h += b_k[:, None] * b_v[None, :]
        _d_q = b_h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += K
        p_do += V
        p_v += V
        p_dq += K

    # sync threads
    tl.debug_barrier()

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(T):
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * b_v[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)

        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V


class FusedRecurrentLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, scale, initial_state=None, output_final_state=False):
        B, H, T, K = q.shape
        V = v.shape[-1]

        BK, BV = min(K, 32), min(V, 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 1
        num_stages = 1

        o = q.new_empty(NK, B, H, T, V)
        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        grid = (NV, NK, B * H)
        fused_recurrent_linear_attn_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1),
            v.stride(1), scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = ctx.scale

        BK, BV = min(K, 32), min(V, 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_warps = 1
        num_stages = 1

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_recurrent_linear_attn_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1),
            v.stride(1),
            scale,
            B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            num_warps=num_warps,
            num_stages=num_stages
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv, None, None, None


def fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not head_first:
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, final_state = FusedRecurrentLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
