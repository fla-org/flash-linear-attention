# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from einops import rearrange


def naive_mesa_net_decoding_one_step(q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv, max_CG_iteration=30):
    q = q.float().clone()
    k = k.float().clone()
    v = v.float().clone()
    g = g.float().clone()
    lamb = lamb.float().clone()
    beta = beta.float().clone()
    B, h, d = q.shape
    k_beta = k * beta.unsqueeze(-1)

    h_kk = prev_h_kk * g.exp()[..., None, None] + k_beta.unsqueeze(-1) * k.unsqueeze(-2)
    h_kv = prev_h_kv * g.exp()[..., None, None] + k_beta.unsqueeze(-1) * v.unsqueeze(-2)
    diag_H = torch.diagonal(h_kk, dim1=-2, dim2=-1)
    lamb = lamb.unsqueeze(0)
    x = q / (diag_H + lamb)
    r = q - (x.unsqueeze(-1) * h_kk).sum(-2) - (lamb * x)
    p = r.clone()
    delta_old = (r * r).sum(-1)
    # CG iteration
    for i in range(max_CG_iteration):
        q = (p.unsqueeze(-1) * h_kk).sum(-2) + (lamb * p)
        alpha = (delta_old / ((p * q).sum(-1) + 1e-5))
        x = x + (alpha[..., None] * p)
        r = r - (alpha[..., None] * q)
        delta_new = (r * r).sum(-1)
        beta = delta_new / (delta_old + 1e-5)
        p = r + (beta[..., None] * p)
        delta_old = delta_new
    o = (x.unsqueeze(-1) * h_kv).sum(-2)
    return o, h_kk, h_kv


def naive_mesa_net_exact(q, k, v, g, lamb, beta, h_kk_init=None, h_kv_init=None):
    B, L, h, d = q.shape
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    lamb = lamb.float()
    beta = beta.float()

    h_kk = h_kk_init.clone() if h_kk_init is not None else torch.zeros(B, h, d, d, device=q.device)
    h_kv = h_kv_init.clone() if h_kv_init is not None else torch.zeros(B, h, d, d, device=q.device)

    h_kk_all = torch.zeros(B, L, h, d, d, device=q.device)
    h_kv_all = torch.zeros(B, L, h, d, d, device=q.device)
    for i in range(L):
        h_kk = h_kk * g[:, i, :, None, None].exp() + (k[:, i, :, :] * beta[:, i, :, None]
                                                      )[..., None] * k[:, i, :, None, :]
        h_kv = h_kv * g[:, i, :, None, None].exp() + (k[:, i, :, :] * beta[:, i, :, None]
                                                      )[..., None] * v[:, i, :, None, :]
        h_kk_all[:, i] = h_kk
        h_kv_all[:, i] = h_kv

    q_star_gold = torch.linalg.solve(h_kk_all + torch.diag_embed(lamb)[None, None, ...,], q)
    o_gold = (q_star_gold[..., :, None] * h_kv_all).sum(-2)
    return o_gold, h_kk, h_kv


def naive_mesa_net_CG(q, k, v, g, lamb, beta, chunk_size, max_CG_iteration=30, h_kk_init=None, h_kv_init=None):
    B, L, h, d = q.shape
    C = chunk_size

    def chunk_fn(x): return rearrange(x, 'b (n c) h ... -> b h n c ...', c=C).float()

    q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk = map(chunk_fn, [q, k, v, g, beta])

    g_chunk = g_chunk.cumsum(dim=-1)

    pairwise_decay = (g_chunk[..., None] - g_chunk[..., None, :]).exp().tril() * beta_chunk[..., None, :]

    num_chunks = q_chunk.shape[2]

    h_kv_all = torch.zeros(B, h, num_chunks, d, d, device=q.device)
    h_kk_all = torch.zeros(B, h, num_chunks, d, d, device=q.device)

    h_kv = torch.zeros(B, h, d, d, device=q.device)
    h_kk = torch.zeros(B, h, d, d, device=q.device)

    if h_kk_init is not None:
        h_kk += h_kk_init
    if h_kv_init is not None:
        h_kv += h_kv_init

    chunk_decay_k = (g_chunk[..., -1, None] - g_chunk).exp()
    chunk_decay_q = g_chunk.exp()

    k_chunk_processed = k_chunk * chunk_decay_k[..., None] * beta_chunk[..., None]

    for i in range(num_chunks):
        h_kv_all[:, :, i, :, :] = h_kv
        h_kk_all[:, :, i, :, :] = h_kk

        k_chunk_i = k_chunk[:, :, i, :, :]
        v_chunk_i = v_chunk[:, :, i, :, :]
        k_chunk_i_processed = k_chunk_processed[:, :, i, :, :]

        h_kk = h_kk * g_chunk[:, :, i, -1, None, None].exp() + (k_chunk_i_processed).transpose(-2, -1) @ k_chunk_i
        h_kv = h_kv * g_chunk[:, :, i, -1, None, None].exp() + (k_chunk_i_processed).transpose(-2, -1) @ v_chunk_i

    # CG solver to approximate the matrix inverse solution.
    # diag_H = torch.diagonal(h_kk_all, dim1=-2, dim2=-1)
    lamb = lamb[None, :, None, None, :]
    x = torch.zeros_like(q_chunk)
    r = q_chunk - (x * chunk_decay_q[..., None]) @ h_kk_all - ((x @ k_chunk.transpose(-2, -1))
                                                               * pairwise_decay) @ k_chunk - (lamb * x)
    p = r.clone()
    delta_old = (r * r).sum(-1)

    # CG iteration
    for i in range(max_CG_iteration):
        q = (p * chunk_decay_q[..., None]) @ h_kk_all + ((p @ k_chunk.transpose(-1, -2))
                                                         * pairwise_decay) @ k_chunk + (lamb * p)
        alpha = (delta_old / ((p * q).sum(-1) + 1e-5))
        x = x + (alpha[..., None] * p)
        r = r - (alpha[..., None] * q)
        delta_new = (r * r).sum(-1)
        beta = delta_new / (delta_old + 1e-5)
        p = r + (beta[..., None] * p)
        delta_old = delta_new

    o = (x * chunk_decay_q[..., None]) @ h_kv_all + ((x @ k_chunk.transpose(-1, -2))
                                                     * pairwise_decay) @ v_chunk
    return rearrange(o, 'b h n c d -> b (n c) h d'), h_kk, h_kv
