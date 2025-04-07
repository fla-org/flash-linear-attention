# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Yuqi Pan

import torch
import torch.nn.functional as F


def ttt_linear(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float,
    eps: float,
    mini_batch_size: int,
    initial_state: torch.Tensor,
    initial_state_bias: torch.Tensor,
    output_final_state: bool
):
    B, H, T, D = q.shape
    BT = mini_batch_size
    NT = T // BT
    # [NT, B, H, mini_batch_size, D]
    _q = q.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _k = k.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    _v = v.reshape(B, H, NT, BT, D).permute(2, 0, 1, 3, 4)
    # [NT, B, H, BT, 1]
    _eta = eta.reshape(B, H, NT, BT, 1).permute(2, 0, 1, 3, 4)
    # [H, 1, D]
    w = w.reshape(H, 1, D).to(torch.float32)
    b = b.reshape(H, 1, D).to(torch.float32)

    h = torch.zeros((B, H, D, D), device=v.device, dtype=torch.float32) if initial_state is None else initial_state
    hb = torch.zeros((B, H, 1, D), device=v.device, dtype=torch.float32) if initial_state_bias is None else initial_state_bias
    q *= scale
    # [NT, B, H, BT, D]
    o = torch.empty_like(_v)

    for i in range(NT):
        q_i, k_i, v_i, eta_i = [x[i] for x in [_q, _k, _v, _eta]]
        kh = k_i @ h + hb
        reconstruction_target = v_i - k_i

        mean = kh.mean(-1, True)
        var = kh.var(-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        kh_hat = (kh - mean) / rstd

        g = w * kh_hat + b - reconstruction_target
        g *= w
        v_new = (D * g - g.sum(-1, True) - kh_hat * (g * kh_hat).sum(-1, True)) / (rstd * D)

        Attn = torch.tril(q_i @ k_i.transpose(-2, -1))
        o_i = q_i @ h - (eta_i * Attn) @ v_new + hb - torch.tril(eta_i.expand_as(Attn)) @ v_new
        h = h - (eta_i[:, :, -1, :, None] * k_i).transpose(-1, -2) @ v_new
        hb = hb - torch.sum(eta_i[:, :, -1, :, None] * v_new, dim=-2, keepdim=True)
        # layer norm with residuals

        mean = o_i.mean(dim=-1, keepdim=True)
        var = o_i.var(dim=-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        o[i] = o_i + (o_i - mean) / rstd * w + b

    # [B, H, T, D]
    o = o.permute(1, 2, 0, 3, 4).reshape(B, H, T, D)
    h = h if output_final_state else None
    hb = hb if output_final_state else None
    return o, h, hb


def chunk_ttt_linear_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    scale: float = None,
    eps: float = 1e-6,
    mini_batch_size: int = 16,
    initial_state: torch.Tensor = None,
    initial_state_bias: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = False,
):
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "The key and value dimension must be the same."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        eta = eta.transpose(1, 2)
    T = q.shape[-2]
    padded = (mini_batch_size - (T % mini_batch_size)) % mini_batch_size
    if padded > 0:
        q = F.pad(q, (0, 0, 0, padded))
        k = F.pad(k, (0, 0, 0, padded))
        v = F.pad(v, (0, 0, 0, padded))
        eta = F.pad(eta, (0, 0, 0, padded))
        eta[:, :, -1, :] = eta[:, :, -(padded+1), :]
    assert q.shape[-2] % mini_batch_size == 0, "Sequence length should be a multiple of mini_batch_size."
    q, k, v, eta, w, b = map(lambda x: x.to(torch.float32), [q, k, v, eta, w, b])
    o, final_state, final_state_bias = ttt_linear(
        q,
        k,
        v,
        w,
        b,
        eta,
        scale,
        eps,
        mini_batch_size,
        initial_state,
        initial_state_bias,
        output_final_state,
    )
    o = o[:, :, :T, :].contiguous()
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state, final_state_bias
