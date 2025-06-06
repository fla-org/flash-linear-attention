# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch


def naive_mesa_net_decoding_one_step(q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv, max_CG_iteration=30):
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    lamb = lamb.float()
    beta = beta.float()
    B, h, d = q.shape
    k_beta = k * beta.unsqueeze(-1)
    h_kk = prev_h_kk * g.exp()[..., None, None] + k_beta.unsqueeze(-1) * k.unsqueeze(-2)
    h_kv = prev_h_kv * g.exp()[..., None, None] + k_beta.unsqueeze(-1) * v.unsqueeze(-2)
    diag_H = torch.diagonal(h_kk, dim1=-2, dim2=-1)
    lamb = lamb.unsqueeze(0)
    x = q / (diag_H + lamb)
    r = q - (x.unsqueeze(-1) * h_kk).sum(-2) - (lamb * x)
    p = r.clone()
    delta_old = (r * r).sum(-1) + 1e-5
    # CG iteration
    for i in range(max_CG_iteration):
        q = (p.unsqueeze(-1) * h_kk).sum(-2) + (lamb * p)
        alpha = (delta_old / ((p * q).sum(-1) + 1e-5))
        x = x + (alpha[..., None] * p)
        r = r - (alpha[..., None] * q)

        delta_new = (r * r).sum(-1) + 1e-5
        beta = delta_new / (delta_old)
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
