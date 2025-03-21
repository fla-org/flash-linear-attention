# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.exp import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, use_cuda_graph


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BK'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_dplr_delta_rule_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    gk,
    o,
    h0,
    ht,
    offsets,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if HEAD_FIRST:
        p_q = q + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_k = k + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_a = a + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_b = b + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_gk = gk + i_nh * T*K + ((T-1) * K if REVERSE else 0) + o_k
        p_v = v + i_nh * T*V + ((T-1) * V if REVERSE else 0) + o_v
        p_o = o + i_nh * T*V + ((T-1) * V if REVERSE else 0) + o_v

    else:
        p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_a = a + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_b = b + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_gk = gk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
        p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
        p_o = o + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h = exp(b_gk)[None, :] * b_h + (tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
        b_o = tl.sum(b_h * b_q[None, :], axis=1)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_a += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_b += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_gk += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_o += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_dplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if offsets is None else len(offsets) - 1
    BK = triton.next_power_of_2(K)

    h0 = initial_state
    if output_final_state:
        ht = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        ht = None
    o = torch.empty_like(v)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    fused_recurrent_dplr_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        gk,
        o,
        h0,
        ht,
        offsets,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        REVERSE=reverse,
        HEAD_FIRST=head_first
    )
    return o, ht


class FusedRecurrentDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: Optional[float] = 1.0,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = False
    ):
        o, ht = fused_recurrent_dplr_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            offsets=offsets,
            head_first=head_first
        )
        return o, ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
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
    reverse: bool = False,
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
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.
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
        reverse,
        cu_seqlens,
        head_first
    )
    return o, final_state
