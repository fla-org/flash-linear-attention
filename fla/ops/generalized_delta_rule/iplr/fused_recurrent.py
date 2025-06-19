# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
)
@triton.jit
def fused_recurrent_fwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V].
    a,  # a [B, H, L, K]
    b,  # b [B, H, L, K]
    o,  # output [B, H, L, V]
    ha,  # tmp variable [B, H, L, V] for storing intermediate results of (h * a[None, :]).sum(0)
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    cu_seqlens,  # varlen cu_seqlens
    scale,  # K ** -0.5
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    p_q = q + (bos * H + i_h) * K + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + tl.arange(0, BK)
    p_a = a + (bos * H + i_h) * K + tl.arange(0, BK)
    p_b = b + (bos * H + i_h) * K + tl.arange(0, BK)
    p_ha = ha + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_o = o + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    mask_k = tl.arange(0, BK) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + (tl.arange(0, BK)[None, :]) * V + ((i_v * BV + tl.arange(0, BV))[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        # to store
        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h += (tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_ha, tmp.to(p_ha.dtype.element_ty), mask=mask_v)
        p_q += K*H
        p_k += K*H
        p_o += V*H
        p_v += V*H
        p_ha += V*H
        p_a += K*H
        p_b += K*H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + (tl.arange(0, BK)[None, :]) * V + ((i_v * BV + tl.arange(0, BV))[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_DHT': lambda args: args['dht'] is not None,
    'USE_DH0': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["BK", "BV"],
)
@triton.jit
def fused_recurrent_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: b_dhead
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    a,  # a [B, H, L, K]
    b,  # b [B, H, L, K]
    ha,  # ha [B, H, L, V]
    dht,  # gradient of final state [B, H, K, V]
    dh0,  # gradient of initial state [B, H, K, V]
    do,  # gradient of output [B, H, L, V]
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dv,  # gradient of value [NK, B, H, L, V]
    da,  # gradient of a [NV, B, H, L, K]
    db,  # gradient of b [NV, B, H, L, K]
    dha,  # gradient of ha [NK, B, H, L, V]
    h0,  # initial state [B, H, K, V]
    scale,  # K ** -0.5
    cu_seqlens,  # cu_seqlens
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state h0
    USE_DH0: tl.constexpr,  # whether to use dh0
    USE_DHT: tl.constexpr,  # whether to use dht
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    dk += i_v * B * H * K * T
    db += i_v * B * H * K * T
    dq += i_v * B * H * K * T
    da += i_v * B * H * K * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    mask_k = tl.arange(0, BK) < K
    mask_v = (tl.arange(0, BV) + i_v * BV) < V

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V + i_v * BV
    ha += (bos * H + i_h) * V + i_v * BV
    a += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    do += (bos * H + i_h) * V + i_v * BV
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V + i_v * BV
    da += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    dha += (bos * H + i_h) * V + i_v * BV

    p_q = q + tl.arange(0, BK) + (T - 1) * H*K
    p_k = k + tl.arange(0, BK) + (T - 1) * H*K
    p_v = v + tl.arange(0, BV) + (T - 1) * H*V
    p_ha = ha + tl.arange(0, BV) + (T - 1) * H*V
    p_a = a + tl.arange(0, BK) + (T - 1) * H*K
    p_b = b + tl.arange(0, BK) + (T - 1) * H*K
    p_do = do + tl.arange(0, BV) + (T - 1) * H*V
    p_dk = dk + tl.arange(0, BK) + (T - 1) * H*K
    p_dv = dv + tl.arange(0, BV) + (T - 1) * H*V
    p_dha = dha + tl.arange(0, BV) + (T - 1) * H*V
    p_db = db + tl.arange(0, BK) + (T - 1) * H*K
    p_da = da + tl.arange(0, BK) + (T - 1) * H*K
    p_dq = dq + tl.arange(0, BK) + (T - 1) * H*K

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = dht + i_nh * K * V + (tl.arange(0, BK)[:, None]) * V + ((i_v * BV + tl.arange(0, BV))[None, :])
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_ha = tl.load(p_ha, mask=mask_v, other=0).to(tl.float32)

        b_dh += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(b_dh * b_v[None, :], axis=1)
        d_v = tl.sum(b_dh * b_k[:, None], axis=0)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_v)

        b_dha = tl.sum(b_dh * b_b[:, None], axis=0)
        tl.store(p_dha, b_dha.to(p_dha.dtype.element_ty), mask=mask_v)
        b_db = tl.sum(b_dh * b_ha[None, :], axis=1)
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=mask_k)

        b_dh += b_dha[None, :] * b_a[:, None]
        p_do -= H*V
        p_q -= H*K
        p_k -= H*K
        p_v -= H*V
        p_dk -= H*K
        p_dv -= H*V
        p_b -= H*K
        p_db -= H*K
        p_a -= H*K
        p_dha -= H*V
        p_ha -= H*V

    if USE_DH0:
        p_dh0 = dh0 + i_nh * K * V + (tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_nh * K * V + (tl.arange(0, BK)[:, None]) * V + ((i_v * BV + tl.arange(0, BV))[None, :])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    p_k = k + tl.arange(0, BK)
    p_v = v + tl.arange(0, BV)
    p_ha = ha + tl.arange(0, BV)
    p_do = do + tl.arange(0, BV)
    p_dha = dha + tl.arange(0, BV)
    p_da = da + tl.arange(0, BK)
    p_dq = dq + tl.arange(0, BK)
    p_b = b + tl.arange(0, BK)

    for i in range(0, T):
        b_dha = tl.load(p_dha, mask=mask_v, other=0).to(tl.float32)
        d_a = tl.sum(b_dha[None, :] * b_h, axis=1)
        tl.store(p_da, d_a.to(p_da.dtype.element_ty), mask=mask_k)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_ha = tl.load(p_ha, mask=mask_v, other=0).to(tl.float32)
        b_h += b_k[:, None] * b_v[None, :] + b_b[:, None] * b_ha[None, :]
        _d_q = b_h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += H*K
        p_do += H*V
        p_v += H*V
        p_da += H*K
        p_dha += H*V
        p_ha += H*V
        p_dq += H*K
        p_b += H*K


class FusedRecurrentIPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        B, T, H, K, V = *k.shape, v.shape[-1]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        BK = triton.next_power_of_2(K)
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            final_state = None

        ha = torch.empty_like(v, dtype=torch.float32)

        def grid(meta): return (
            triton.cdiv(V, meta['BV']),
            N * H
        )
        o = torch.empty_like(v)
        fused_recurrent_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            o=o,
            ha=ha,
            h0=initial_state,
            ht=final_state,
            scale=scale,
            cu_seqlens=cu_seqlens,
            H=H,
            T=T,
            K=K,
            V=V,
            BK=BK,
        )
        ctx.save_for_backward(q, k, v, a, b, ha, initial_state)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, a, b, ha, initial_state = ctx.saved_tensors
        B, T, H, K, V = *q.shape, v.shape[-1]
        N = B if ctx.cu_seqlens is None else len(ctx.cu_seqlens) - 1
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 64)
        NV = triton.cdiv(V, BV)
        scale = ctx.scale

        dq = q.new_empty(NV, *q.shape)
        dk = k.new_empty(NV, *k.shape)
        da = a.new_empty(NV, *a.shape)
        db = b.new_empty(NV, *b.shape)
        dv = torch.empty_like(v)
        dha = torch.empty_like(ha)
        grid = (NV, N * H)

        if initial_state is not None and initial_state.requires_grad:
            dh0 = torch.empty_like(initial_state, dtype=torch.float32)
        else:
            dh0 = None

        fused_recurrent_bwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ha=ha,
            dht=dht,
            dh0=dh0,
            do=do,
            dq=dq,
            dk=dk,
            dv=dv,
            da=da,
            db=db,
            dha=dha,
            h0=initial_state,
            scale=scale,
            cu_seqlens=ctx.cu_seqlens,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        da = da.sum(0)
        db = db.sum(0)
        return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), None, dh0, None, None


def fused_recurrent_iplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence S_t = S_t @ (I + a_t b_t^T) + v_t k_t^T in a recurrent manner.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]`
        a (torch.Tensor):
            as of shape `[B, T, H, K]`
        b (torch.Tensor):
            bs of shape `[B, T, H, K]`
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    o, final_state = FusedRecurrentIPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens
    )
    return o, final_state
