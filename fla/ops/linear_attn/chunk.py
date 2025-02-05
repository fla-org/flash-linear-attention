# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Yu Zhang, Songlin Yang

from typing import Optional, Tuple

import torch

from fla.ops.linear_attn.utils import normalize_output
from fla.ops.simple_gla import chunk_simple_gla


@torch.compiler.disable
def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    normalize: bool = True,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        scale (Optional[int]):
            Scale factor for the linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first
    )
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state
