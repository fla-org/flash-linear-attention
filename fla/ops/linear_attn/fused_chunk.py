# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from packaging import version

from fla.ops.linear_attn.utils import normalize_output
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [1]
    ],
    key=['B', 'H', 'K', 'V', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_linear_attn_fwd_kernel(
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
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    o_i = tl.arange(0, BT)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + (i_b * T*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (i_b * T*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (i_b * T*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o + (i_k*B*T*H + i_b*T*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if CHECK and i_t == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [1]
    ],
    key=['B', 'H', 'K', 'V', 'BK', 'BV'],
)
@triton.jit
def fused_chunk_linear_attn_bwd_kernel(
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
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)

    m_s = o_i[:, None] >= o_i[None, :]
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + (i_b * T*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + (i_b * T*H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (i_b * T*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_v*B*T*H+i_b*T*H+i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [V, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        # [BV, BK]
        if CHECK and i_t == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = o_i[:, None] <= o_i[None, :]
    for i_t in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + (i_b * T*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, T - i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + (i_b * T*H + i_h) * K, (T, K), (H*K, 1), (T - i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + (i_b * T*H + i_h) * V, (T, V), (H*V, 1), (T - i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (i_b * T*H + i_h) * V, (T, V), (H*V, 1), (T - i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_v*B*T*H+i_b*T*H+i_h) * K, (T, K), (H*K, 1), (T - i_t*BT, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*B*T*H+i_b*T*H+i_h) * V, (T, V), (H*V, 1), (T - i_t*BT, i_v*BV), (BT, BV), (1, 0))
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0).to(b_q.dtype)
        # [BT, BT]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0).to(b_q.dtype)
        # [BT, BK]
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        if CHECK and i_t == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        B, T, H, K, V = *k.shape, v.shape[-1]
        BT = min(64, max(16, triton.next_power_of_2(T)))
        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        o = q.new_empty(NK, *v.shape)
        final_state = q.new_empty(B, H, K, V, dtype=torch.float) if output_final_state else None
        # the bug still exists even for Triton 2.2 on H100 GPUs
        # so we always enable initial checks
        CHECK = True
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        grid = (NV, NK, B * H)
        fused_chunk_linear_attn_fwd_kernel[grid](
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
            BT=BT,
            BK=BK,
            BV=BV,
            CHECK=CHECK
        )
        o = o.sum(0) if NK > 1 else o[0]

        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.CHECK = CHECK
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, initial_state = ctx.saved_tensors
        B, T, H, K, V = *k.shape, v.shape[-1]
        scale = ctx.scale

        BT = min(64, max(16, triton.next_power_of_2(T)))
        BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

        dq = q.new_empty(NV, *q.shape)
        dk = q.new_empty(NV, *k.shape)
        dv = q.new_empty(NK, *v.shape)
        grid = (NV, NK, B * H)

        fused_chunk_linear_attn_bwd_kernel[grid](
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
            BT=BT,
            BK=BK,
            BV=BV,
            CHECK=ctx.CHECK
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def fused_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = True,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`
        scale (Optional[int]):
            Scale factor for linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    if not head_first:
        if q.shape[1] < q.shape[2]:
            raise DeprecationWarning(
                f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
                "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
                "when head_first=False was specified. "
                "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
            )
    o, final_state = FusedChunkLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    if normalize:
        o = normalize_output(q * scale, k, o)
    if head_first:
        o = o.transpose(1, 2)
    return o, final_state
