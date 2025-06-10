# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.linear_attn.utils import normalize_output
from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_linear_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * T*V + i_v * BV + tl.arange(0, BV)

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


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_linear_attn_bwd_kernel(
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    h0,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV)

    p_dq = dq + (i_bh + i_v * B * H) * T*K + i_k * BK + tl.arange(0, BK)
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

    p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dk = dk + (i_bh + i_v * B * H) * T*K + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * T*V + i_v * BV + tl.arange(0, BV) + (T - 1) * V
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

        o = q.new_empty(NK, B, H, T, V)
        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        grid = (NV, NK, B * H)
        fused_recurrent_linear_attn_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            initial_state,
            final_state,
            scale,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
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

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_recurrent_linear_attn_bwd_kernel[grid](
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            initial_state,
            scale,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
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
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first:
        if q.shape[1] < q.shape[2]:
            raise DeprecationWarning(
                f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
                "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
                "when head_first=False was specified. "
                "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
            )
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    o, final_state = FusedRecurrentLinearAttentionFunction.apply(
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state
    )
    if normalize:
        o = normalize_output(q * scale, k, o)
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
