# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv, chunk_fwd_o
from fla.ops.utils import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = chunk_local_cumsum(g, chunk_size, offsets=offsets, head_first=head_first) if g is not None else None
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return g, o, ht


def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (SY 09/22) states_in_fp32 seems not affecting the error of dg but for safety, set to True
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        states_in_fp32=True,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        states_in_fp32=True,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dq, dk, _, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        do=do,
        dh=dh,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    dv = chunk_bwd_dv(
        q=q,
        k=k,
        g=g,
        do=do,
        dh=dh,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return dq, dk, dv, dg, dh0


class ChunkSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        offsets,
        head_first
    ):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        g, o, ht = chunk_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, k, v, g, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        return o.to(q.dtype), ht

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        chunk_size, scale, offsets, indices, head_first = ctx.chunk_size, ctx.scale, ctx.offsets, ctx.indices, ctx.head_first
        q, k, v, g, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            initial_state=initial_state,
            do=do,
            dht=dht,
            scale=scale,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        if g is not None:
            dg = chunk_local_cumsum(dg, chunk_size, reverse=True, offsets=offsets, head_first=head_first).to(g.dtype)
        else:
            dg = None
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg, None, dh0, None, None, None


def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # log decay
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        scale (Optional[int]):
            Scale factor for the attention scores.
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
        >>> from fla.ops.simple_gla import chunk_simple_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, device='cuda'))
        >>> o, ht = chunk_simple_gla(q, k, v, g,
                                     initial_state=None,
                                     output_final_state=True,
                                     head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_simple_gla(q, k, v, g,
                                             initial_state=None,
                                             output_final_state=True,
                                             cu_seqlens=cu_seqlens,
                                             head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
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
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkSimpleGLAFunction.apply(
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first
    )
    return o, final_state
