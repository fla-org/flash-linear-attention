# -*- coding: utf-8 -*-

import torch
from einops import rearrange

# S_t = S_t @ (I + alpha_t beta_t^T) + v_t k_t^T
# q, k, alpha, beta [B, H, L, D_K]
# v [B, H, L, D_V]


def dplr_recurrence(q, k, v, alpha, beta, gk, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    q, k, v, beta, gk = map(lambda x: x.float(), [q, k, v, beta, gk])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)

    if initial_state is not None:
        S += initial_state

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i]
        _alpha = alpha[:, :, i].clone()
        _beta = beta[:, :, i].clone()
        _kv = _k[..., None] * _v[..., None, :] + (S.clone() * _alpha[..., None]).sum(-2, keepdim=True) * _beta[..., None]
        S = S.clone() * gk[:, :, i].exp()[..., None] + _kv
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def dplr_chunkwise(q, k, v, alpha, beta, gk, initial_state=None, output_final_state=True, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v
    assert l % chunk_size == 0

    S = k.new_zeros(b, h, d_k, d_v).to(q)
    if initial_state is not None:
        S += initial_state

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, alpha, beta, gk = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d',
                                   c=chunk_size).float(), [q, k, v, alpha, beta, gk])

    gk_cumsum = gk.cumsum(-2)

    # v2 = (alpha @ k.transpose(-1, -2)).masked_fill_(mask, 0) @ v
    A_ab = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)
    A_qk = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)
    A_ak = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)
    A_qb = torch.zeros(b, h, l // chunk_size, chunk_size, chunk_size).to(q.device)

    for i in range(chunk_size):
        alpha_i = alpha[:, :, :, i, None]
        q_i = q[:, :, :, i, None]
        gk_i = gk_cumsum[:, :, :, i, None]
        mask = (torch.arange(chunk_size) <= i).to(q.device)
        attn_i = (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
        A_qb[:, :, :, i, :] = (q_i * beta * attn_i).sum(-1).clone()
        mask = (torch.arange(chunk_size) < i).to(q.device)
        # shift by one.
        attn_i = (gk_i - gk[:, :, :, i, None] - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_ab[:, :, :, i, :] = (alpha_i * beta * attn_i).sum(-1).clone()
        A_ak[:, :, :, i, :] = (alpha_i * k * attn_i).sum(-1).clone()

    A_ab = A_ab
    for i in range(1, chunk_size):
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)

    A_ab = A_ab + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    u = A_ab @ (A_ak @ v)
    w = A_ab @ ((gk_cumsum-gk).exp() * alpha)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i, u_i, w_i, beta_i = q[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], beta[:, :, i]
        v2_i = u_i + w_i @ S

        o_1 = A_qk[:, :, i] @ v_i
        o_2 = A_qb[:, :, i] @ v2_i
        o_3 = (q_i * gk_cumsum[:, :, i].exp()) @ S
        o[:, :, i] = o_1 + o_2 + o_3
        decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
        S = S*gk_cumsum[:, :, i, -1, :, None].exp() + (k_i * decay).transpose(-1, -2) @ v_i + \
            (beta_i * decay).transpose(-1, -2) @ v2_i
    S = None if output_final_state is False else S
    return rearrange(o, 'b h n c d -> b h (n c) d'), S
