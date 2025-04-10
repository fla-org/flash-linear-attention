# -*- coding: utf-8 -*-

from typing import Optional

import torch
from einops import repeat


def naive_recurrent_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
) -> torch.Tensor:
    dtype = q.dtype
    q, k, v, s, g = map(lambda x: x.transpose(1, 2).contiguous().float(), (q, k, v, s, g))

    NG = q.shape[1]//k.shape[1]
    # [batch_size, n_heads, seq_len, n_slots]
    if g is None:
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z)
    k, v, s, g = map(lambda x: repeat(x, 'b h t d -> b (h g) t d', g=NG), (k, v, s, g))
    if initial_state is not None:
        initial_state = tuple(map(lambda x: repeat(x, 'b h k v -> b (h g) k v', g=NG), initial_state))

    B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]

    hk = torch.zeros(B, H, K, M, dtype=torch.float, device=q.device)
    ok = torch.zeros_like(s)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    final_state = None
    if initial_state is not None:
        hk += initial_state[0]

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = s[:, :, i]
        g_i = g[:, :, i].exp()
        hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
        ok[:, :, i] = (q_i[..., None] * hk).sum(-2)

    qv = ok.softmax(-1)
    hv = torch.zeros(B, H, M, V, dtype=torch.float, device=q.device)
    ov = torch.zeros_like(v)
    if initial_state is not None:
        hv += initial_state[1]

    for i in range(T):
        q_i = qv[:, :, i]
        k_i = s[:, :, i]
        v_i = v[:, :, i]
        g_i = g[:, :, i].exp()
        hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
        ov[:, :, i] = (q_i[..., None] * hv).sum(-2)

    if output_final_state:
        final_state = (hk.view(B, -1, NG, K, M)[:, :, 0], hv.view(B, -1, NG, M, V)[:, :, 0])
    ov = ov.transpose(1, 2).contiguous()
    return ov.to(dtype), final_state
