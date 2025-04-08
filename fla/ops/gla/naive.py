# -*- coding: utf-8 -*-

from typing import Optional

import torch


def ceildiv(a, b):
    return -(a // -b)


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False
):
    dtype = q.dtype
    q, k, v, gk = map(lambda x: x.transpose(1, 2).float(), (q, k, v, gk))
    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    scale = K ** -0.5

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        h += initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o[:, :, i] = (q_i[..., None] * h).sum(-2)

    if not output_final_state:
        h = None
    return o.transpose(1, 2).to(dtype), h
