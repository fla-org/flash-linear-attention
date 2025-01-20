# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import (chunk_gated_delta_rule_bwd_dhu,
                                          chunk_gated_delta_rule_fwd_h)
from fla.ops.common.chunk_o import (chunk_bwd_dqkwg, chunk_bwd_dv_local,
                                    chunk_fwd_o)
from fla.ops.gated_delta_rule.wy_fast import (bwd_prepare_wy_repr,
                                              fwd_prepare_wy_repr,
                                              fwd_recompute_w_u)
from fla.ops.utils import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    BT = chunk_size
    g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first)
    # obtain WY representation. u is actually the new v.
    w, u, Aw, Au = fwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        g=g,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )

    # obtain output
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return g, o, Aw, Au, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(
        k=k,
        v=v,
        beta=beta,
        Aw=Aw,
        Au=Au,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        dh=None,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dk2, dv, db, dg2 = bwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        g=g,
        Aw=Aw,
        Au=Au,
        dw=dw,
        du=dv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size, reverse=True, offsets=offsets, head_first=head_first)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
        use_qk_l2norm_in_kernel: bool = False
    ):
        chunk_size = 64
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        g, o, Aw, Au, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q_orig, k_orig, v, g, beta, Aw, Au, initial_state, offsets, indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.head_first = head_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, g, beta, Aw, Au, initial_state, offsets, indices = ctx.saved_tensors
        if ctx.use_qk_l2norm_in_kernel:
            q, q_orig = l2norm_fwd(q), q
            k, k_orig = l2norm_fwd(k), k
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            Aw=Aw,
            Au=Au,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
            head_first=ctx.head_first,
            chunk_size=ctx.chunk_size
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_orig, dq)
            dk = l2norm_bwd(k_orig, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    use_qk_l2norm_in_kernel: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
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
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(q, k, v, g, beta,
                                           initial_state=h0,
                                           output_final_state=True,
                                           head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(q, k, v, g, beta,
                                                   initial_state=h0,
                                                   output_final_state=True,
                                                   cu_seqlens=cu_seqlens,
                                                   head_first=False)
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, H, T] if head_first=True, or [B, T, H] if head_first=False."

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
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel
    )
    return o, final_state
