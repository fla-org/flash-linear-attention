
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton

from fla.ops.generalized_delta_rule.dplr.chunk_A_bwd import \
    chunk_dplr_bwd_dqk_intra
from fla.ops.generalized_delta_rule.dplr.chunk_A_fwd import \
    chunk_fwd_intra_dplr_fn
from fla.ops.generalized_delta_rule.dplr.chunk_h_bwd import chunk_dplr_bwd_dhu
from fla.ops.generalized_delta_rule.dplr.chunk_h_fwd import chunk_dplr_fwd_h
from fla.ops.generalized_delta_rule.dplr.chunk_o_bwd import (
    chunk_dplr_bwd_dAu, chunk_dplr_bwd_dv, chunk_dplr_bwd_o)
from fla.ops.generalized_delta_rule.dplr.chunk_o_fwd import chunk_dplr_fwd_o
from fla.ops.generalized_delta_rule.dplr.wy_fast_bwd import chunk_dplr_bwd_wy
from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import fwd_prepare_wy_repr
from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


def chunk_dplr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, offsets=offsets, head_first=head_first)

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        offsets=offsets,
        indices=indices,
        BT=BT,
        head_first=head_first
    )

    w, u, A_ab_inv = fwd_prepare_wy_repr(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    h, v_new, final_state = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    o = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return o, final_state


class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ):
        chunk_size = 64

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        o, final_state = chunk_dplr_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.head_first = head_first
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        BT = ctx.chunk_size
        head_first = ctx.head_first
        offsets = ctx.offsets
        indices = ctx.indices
        scale = ctx.scale

        # ******* start recomputing everything, otherwise i believe the gpu memory will be exhausted *******
        gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, offsets=offsets, head_first=head_first)

        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            scale=scale,
            offsets=offsets,
            indices=indices,
            BT=BT,
            head_first=head_first
        )
        w, u, A_ab_inv = fwd_prepare_wy_repr(
            ag=ag,
            A_ab=A_ab,
            A_ak=A_ak,
            v=v,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT
        )
        h, v_new, _ = chunk_dplr_fwd_h(
            kg=kg,
            bg=bg,
            v=v,
            w=w,
            u=u,
            gk=gi,
            initial_state=initial_state,
            offsets=offsets,
            head_first=head_first,
            chunk_size=BT
        )
        u = None
        # ******* end of recomputation *******

        dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
            v=v,
            v_new=v_new,
            do=do,
            A_qb=A_qb,
            scale=scale,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT
        )

        dh, dh0, dv_new = chunk_dplr_bwd_dhu(
            qg=qg,
            bg=bg,
            w=w,
            gk=gi,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv_new_intra,
            offsets=offsets,
            head_first=head_first,
            chunk_size=BT
        )

        dv = chunk_dplr_bwd_dv(
            A_qk=A_qk,
            kg=kg,
            do=do,
            dh=dh,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT
        )

        dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
            k=kg,
            b=bg,
            v=v,
            v_new=v_new,
            do=do,
            h=h,
            dh=dh,
            dv=dv_new,
            w=w,
            gk=gi,
            offsets=offsets,
            indices=indices,
            chunk_size=BT,
            scale=scale,
            head_first=head_first,
        )

        dA_ab, dA_ak, dv2, dag = chunk_dplr_bwd_wy(
            A_ab_inv=A_ab_inv,
            A_ak=A_ak,
            v=v,
            ag=ag,
            dw=dw,
            du=dv_new,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT
        )

        dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            dAqk=dA_qk,
            dAqb=dA_qb,
            dAak=dA_ak,
            dAab=dA_ab,
            dgk_last=dgk_last,
            dqg=dqg,
            dkg=dkg,
            dag=dag,
            dbg=dbg,
            chunk_size=BT,
            scale=scale,
            head_first=head_first,
            offsets=offsets,
            indices=indices
        )
        dv.add_(dv2)
        return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dgk.to(gk), None, dh0, None, None, None


def chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            activations of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            betas of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        gk (torch.Tensor):
            gk of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    # assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    # gk = gk.float()

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDPLRDeltaRuleFunction.apply(
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
