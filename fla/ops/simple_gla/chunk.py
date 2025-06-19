# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv, chunk_fwd_o
from fla.ops.utils import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    return o, ht


def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_gamma: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (SY 09/22) states_in_fp32 seems not affecting the error of dg but for safety, set to True
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        states_in_fp32=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        states_in_fp32=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dq, dk, _, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dv = chunk_bwd_dv(
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    return dq, dk, dv, dg, dh0


class ChunkSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens
    ):
        T = q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens) if g is not None else None
        o, ht = chunk_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        q, k, v, g, g_gamma, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            initial_state=initial_state,
            do=do,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )
        if g is not None:
            dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens).to(g)
        else:
            dg = None
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None


@torch.compiler.disable
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        g_gamma (torch.Tensor):
            Log decay of shape `[H]`.
            Head-wise data-independent decay is used if `g_gamma` is provided.
            Only one of `g` or `g_gamma` should be provided.
        scale (Optional[float]):
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
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
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
        >>> o, ht = chunk_simple_gla(
            q, k, v, g,
            initial_state=None,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_simple_gla(
            q, k, v, g,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
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
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkSimpleGLAFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens
    )
    return o, final_state
