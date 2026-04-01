# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_comba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent COMBA.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        p: [B, T, H, K]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, p, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, p, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        b_p = p[:, :, i]
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_p[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def naive_chunk_comba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of chunk COMBA.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        p: [B, T, H, K]
        g: [B, T, H]
        beta: [B, T, H]
        chunk_size: int
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, p, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, p, beta, g])

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        p = F.pad(p, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    p_beta = p * beta[..., None]
    assert l % chunk_size == 0

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, p_beta, decay, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, p_beta, decay.unsqueeze(-1), g.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)  # [B, H, n, c]
    decay_0 = decay - g.squeeze(-1)  # [B, H, n, c]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    L_mask_0 = ((decay_0.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()

    # [B, H, n, c, d] @ [B, H, n, d, c] -> [B, H, n, c, c]
    attn = -((p_beta @ k.transpose(-1, -2)) * L_mask_0).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    # for U
    k_cumsum = attn @ v
    # for W
    k_cumdecay = attn @ (p_beta * decay_0[..., None].exp())
    v = k_cumsum

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
                                                        [..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None

    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S
