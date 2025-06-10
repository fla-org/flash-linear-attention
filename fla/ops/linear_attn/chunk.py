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
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`
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
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`
    """

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
    if not head_first:
        if q.shape[1] < q.shape[2]:
            raise DeprecationWarning(
                f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
                "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
                "when head_first=False was specified. "
                "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
            )
    o, final_state = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state
    )
    if normalize:
        o = normalize_output(q * scale, k, o)
    if head_first:
        o = o.transpose(1, 2)
    return o, final_state
