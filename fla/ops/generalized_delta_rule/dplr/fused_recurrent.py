# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.utils import contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
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
    gk,  # gk [B, H, L, K]
    o,  # output [B, H, L, V]
    ha,  # tmp variable [B, H, L, V] for storing intermediate results of (h * a[None, :]).sum(0)
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    offsets,  # varlen offsets
    scale,  # K ** -0.5
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    # indices
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if HEAD_FIRST:
        p_q = q + i_nh * T*K + tl.arange(0, BK)
        p_k = k + i_nh * T*K + tl.arange(0, BK)
        p_gk = gk + i_nh * T*K + tl.arange(0, BK)
        p_a = a + i_nh * T*K + tl.arange(0, BK)
        p_b = b + i_nh * T*K + tl.arange(0, BK)
        p_o = o + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        p_v = v + i_nh * T*V + i_v * BV + tl.arange(0, BV)
        p_ha = ha + i_nh * T*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q + (bos * H + i_h) * K + tl.arange(0, BK)
        p_k = k + (bos * H + i_h) * K + tl.arange(0, BK)
        p_gk = gk + (bos * H + i_h) * K + tl.arange(0, BK)
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
        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        # to store
        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h = tl.exp(b_gk)[None, :] * b_h + (tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
        _o = b_h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_ha, tmp.to(p_ha.dtype.element_ty), mask=mask_v)
        p_q += K if HEAD_FIRST else K*H
        p_k += K if HEAD_FIRST else K*H
        p_gk += K if HEAD_FIRST else K*H
        p_o += V if HEAD_FIRST else V*H
        p_v += V if HEAD_FIRST else V*H
        p_ha += V if HEAD_FIRST else V*H
        p_a += K if HEAD_FIRST else K*H
        p_b += K if HEAD_FIRST else K*H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + (tl.arange(0, BK)[None, :]) * V + ((i_v * BV + tl.arange(0, BV))[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        a,
        b,
        gk,
        scale=None,
        initial_state=None,
        output_final_state=False,
        offsets=None,
        head_first=False
    ):
        if head_first:
            B, H, T, K, V = *k.shape, v.shape[-1]
        else:
            B, T, H, K, V = *k.shape, v.shape[-1]
        N = B if offsets is None else len(offsets) - 1

        BK = triton.next_power_of_2(K)
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            final_state = None

        ha = torch.empty_like(v, dtype=torch.float32)

        def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
        o = torch.empty_like(v)
        fused_recurrent_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            o=o,
            ha=ha,
            h0=initial_state,
            ht=final_state,
            scale=scale,
            offsets=offsets,
            H=H,
            T=T,
            K=K,
            V=V,
            BK=BK,
            HEAD_FIRST=head_first
        )
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass for fused_recurrent_dplr_delta_rule is not implemented and will not be supported. "
            "This kernel is only for inference. "
            "For training, please use `chunk_dplr_delta_rule`."
        )


def fused_recurrent_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence S_t = S_t @ (I + a_t b_t^T) + v_t k_t^T in a recurrent manner.

    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]`
        a (torch.Tensor):
            as of shape `[B, H, T, K]`
        b (torch.Tensor):
             bs of shape `[B, H, T, K]`
        gk (torch.Tensor):
            gk of shape `[B, H, T, K]`
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If None, it will default to `1 / sqrt(K)`. Default: `1.0`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    o, final_state = FusedRecurrentDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first
    )
    return o, final_state
