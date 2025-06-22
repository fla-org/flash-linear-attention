# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    scale: Optional[float] = None,
):
    q, k, v, g = map(lambda x: rearrange(x, 'b t h ... -> b h t ...').to(torch.float32), [q, k, v, g])
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5

    T = q.shape[-2]
    BT = chunk_size
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len))
    decay = g
    B, H, T1, K = q.shape
    V = v.shape[-1]
    q = q * scale
    q, k, v, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    for i in range(0, T1 // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i])
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_i
        S = S * decay[:, :, i, -1, None, None].exp() + \
            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b (n c) h d')[:, :T]
    return o, S


def naive_recurrent_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True
):
    dtype = q.dtype
    q, k, v, g = map(lambda x: x.transpose(1, 2).float(), (q, k, v, g))
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q * scale
    o = v.new_zeros(B, H, T, V)

    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state

    for i in range(T):
        gate = g[:, :, i].exp()
        key = k[:, :, i]
        value = v[:, :, i]
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
        S = S * gate.unsqueeze(-1).unsqueeze(-1) + kv
        q_i = q[:, :, i, :]
        o_i = (q_i.unsqueeze(-1) * S).sum(-2)
        o[:, :, i] = o_i
    if not output_final_state:
        S = None
    return o.transpose(1, 2).to(dtype), S


def naive_parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None
):
    q, k, v, g = map(lambda x: rearrange(x, 'b t h ... -> b h t ...').to(torch.float32), [q, k, v, g])
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5
    dtype = q.dtype
    A = (q @ k.transpose(-1, -2) * scale)
    if g is not None:
        g = g.cumsum(-1)
        D = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
        A = A * D
    else:
        A = A.tril()
    o = A @ v
    o = o.transpose(1, 2)
    return o.to(dtype), A
