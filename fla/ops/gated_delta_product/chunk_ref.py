# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


@torch.compiler.disable
def chunk_gated_delta_product_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    assert q.dtype != torch.float32, "ChunkGatedDeltaProductFunction does not support float32. Please use bfloat16."
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)
    q_new = q.new_zeros(B, T, num_householder, H, K)
    q_new[:, :, -1] = q
    q = rearrange(q_new, 'b t n h d -> b (t n) h d')

    if g is not None:
        g_new = g.new_zeros(B, T, num_householder, H, dtype=torch.float32)
        g_new[:, :, 0] = g
        g = rearrange(g_new, 'b t n h -> b (t n) h')
        o, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens * num_householder if cu_seqlens is not None else None,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
    else:
        o, final_state = chunk_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens * num_householder if cu_seqlens is not None else None,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
    o = rearrange(o, 'b (t n) h d -> b t n h d', n=num_householder)
    return o[:, :, -1].contiguous(), final_state
