# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch

from fla.ops.linear_attn.utils import normalize_output
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla


def fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    normalize: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    o, final_state = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state
