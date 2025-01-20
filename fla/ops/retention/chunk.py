# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch

from fla.ops.simple_gla.chunk import chunk_simple_gla


def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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

    """
    if head_first:
        n_heads = q.shape[1]
    else:
        n_heads = q.shape[2]
    s = (1 - q.new_tensor(2., dtype=torch.float, device=q.device).pow(-5. -
         q.new_tensor(range(n_heads), dtype=torch.float, device=q.device))).log()
    if head_first:
        g = s[None, :, None].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    else:
        g = s[None, None, :].expand(q.shape[0], q.shape[1], q.shape[2]).contiguous()
    return chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
        cu_seqlens=cu_seqlens
    )
