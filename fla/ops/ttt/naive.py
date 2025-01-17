# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def ttt_linear(q, k, v, w, b, eta, scale, eps, mini_batch_size, initial_state, output_final_state):
    B, num_heads, seq_len, head_dim = q.shape
    num_batch = seq_len // mini_batch_size
    # [num_batch, B, num_heads, mini_batch_size, head_dim]
    _q = q.reshape(B, num_heads, num_batch, mini_batch_size, head_dim).permute(2, 0, 1, 3, 4)
    _k = k.reshape(B, num_heads, num_batch, mini_batch_size, head_dim).permute(2, 0, 1, 3, 4)
    _v = v.reshape(B, num_heads, num_batch, mini_batch_size, head_dim).permute(2, 0, 1, 3, 4)
    # [num_batch, B, num_heads, mini_batch_size, 1]
    _eta = eta.reshape(B, num_heads, num_batch, mini_batch_size, 1).permute(2, 0, 1, 3, 4)
    # [num_heads, 1, head_dim]
    w = w.reshape(num_heads, 1, head_dim).to(torch.float32)
    b = b.reshape(num_heads, 1, head_dim).to(torch.float32)
    h = initial_state
    if initial_state is None:
        h = torch.zeros((B, num_heads, head_dim, head_dim), device=v.device, dtype=v.dtype).to(torch.float32)
    q *= scale
    # [num_batch, B, num_heads, mini_batch_size, head_dim]
    out = torch.empty_like(_v)

    for i in range(num_batch):
        q, k, v, eta = [x[i] for x in [_q, _k, _v, _eta]]
        
        kh = k @ h
        reconstruction_target = v - k

        mean = kh.mean(dim=-1, keepdim=True)
        var = kh.var(dim=-1, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        kh_hat = (kh - mean) / rstd

        g = w * kh_hat + b - reconstruction_target
        g *= w
        v_new = (head_dim * g - g.sum(dim=-1, keepdim=True) - kh_hat * (g * kh_hat).sum(dim=-1, keepdim=True)) / (rstd * head_dim)

        Attn = torch.tril(q @ k.transpose(-2, -1))
        o = q @ h - 2 * (eta * Attn) @ v_new
        h = h - 2 * (eta[:, :, -1, :, None] * k).transpose(-1, -2) @ v_new

        mean = o.mean(dim=-1, keepdim=True)
        var = o.var(dim=-1, keepdim=True).to(torch.float32)
        o += (o - mean) / torch.sqrt(var + eps) * w + b

        out[i] = o

    # [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
    out = out.permute(1, 2, 0, 3, 4).reshape(B, num_heads, seq_len, head_dim)
    h = h if output_final_state else None
    return out, h


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
    output_final_state: bool = False,
    head_first: bool = True,
):
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if isinstance(eta, float):
        eta = torch.full_like(q[:, :, :, :1], eta)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        eta = eta.transpose(1, 2)
    seq_len = q.shape[-2]
    pad_len = (mini_batch_size - (seq_len % mini_batch_size)) % mini_batch_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        eta = F.pad(eta, (0, 0, 0, pad_len))
        eta[:,:,-1,:] = eta[:,:,-(pad_len+1),:]
    assert q.shape[-2] % mini_batch_size == 0, "Sequence length should be a multiple of mini_batch_size."
    q, k, v, w, b = map(lambda x: x.to(torch.float32), [q, k, v, w, b])
    o, final_state = ttt_linear(q, k, v, w, b, eta, scale, eps, mini_batch_size, initial_state, output_final_state)
    o = o[:, :, :seq_len, :]
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
