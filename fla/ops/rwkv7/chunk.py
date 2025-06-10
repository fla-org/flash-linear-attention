# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch

from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule


def chunk_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        w (torch.Tensor):
            log decay of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            k of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            v of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        a (torch.Tensor):
            a of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        b (torch.Tensor):
            b of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
            Default: `False`.
    """
    return chunk_dplr_delta_rule(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
    )
