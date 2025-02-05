# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def combine_params(theta, alpha, eta, seq_len):
    beta = torch.cumprod(1 - alpha, dim=2)  # β_t = ∏(1 - α_t) in titans paper

    m = torch.cumprod(eta, dim=2)  # [batch_size, head_dim, sequence_length, 1]
    m[:, :, 0:1, :] = 1

    n = m * theta  # n_i=m_i*theta_i
    beta_T = beta[:, :, -1:, :].clone()  # [batch_size, head_dim, 1, 1]
    # calculate beta_T/beta_j
    beta_ratio = beta_T / beta  # [batch_size, head_dim, sequence_length, 1]
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=beta.dtype)).to(
        beta.device)  # [sequence_length, sequence_length]
    mask = mask.view(1, 1, seq_len, seq_len)  # [1, 1, sequence_length, sequence_length]
    beta_ratio = beta_ratio.view(*beta_ratio.shape[:-1], 1)  # [batch_size, head_dim, sequence_length, 1]
    n = n.view(*n.shape[:-1], 1)  # [batch_size, head_dim, sequence_length, 1]
    f = mask @ beta_ratio * n
    return f, beta, beta_T


# with no layernorm and residual
def titans_linear(q, k, v, w, b, theta, alpha, eta, eps, BT, initial_state, output_final_state):
    B, H, T, D = q.shape
    num_batch = T // BT
    # [num_batch, B, num_heads, mini_batch_size, head_dim]
    _q = q.reshape(B, H, num_batch, BT, D).permute(2, 0, 1, 3, 4)
    _k = k.reshape(B, H, num_batch, BT, D).permute(2, 0, 1, 3, 4)
    _v = v.reshape(B, H, num_batch, BT, D).permute(2, 0, 1, 3, 4)
    # [num_batch, B, num_heads, mini_batch_size, 1]
    _eta = eta.reshape(B, H, num_batch, BT, 1).permute(2, 0, 1, 3, 4)
    _theta = theta.reshape(B, H, num_batch, BT, 1).permute(2, 0, 1, 3, 4)
    _alpha = alpha.reshape(B, H, num_batch, BT, 1).permute(2, 0, 1, 3, 4)
    # [H, 1, D]
    w = w.reshape(H, 1, D).to(torch.float32)
    b = b.reshape(H, 1, D).to(torch.float32)
    # [num_heads, 1, head_dim]
    h = initial_state
    if initial_state is None:
        h = torch.zeros((B, H, D, D), device=v.device, dtype=v.dtype).to(torch.float32)
    # [num_batch, B, num_heads, mini_batch_size, head_dim]
    o = torch.empty_like(_v)

    for i in range(num_batch):
        q_i, k_i, v_i, eta_i, theta_i, alpha_i = [x[i] for x in [_q, _k, _v, _eta, _theta, _alpha]]

        f_i, beta_i, beta_T = combine_params(theta_i, alpha_i, eta_i, BT)
        f_i = torch.diag_embed(f_i.squeeze(-1)).to(q_i.dtype)
        kh = k_i @ h
        reconstruction_target = v_i - k_i

        mean = kh.mean(-1, True)
        var = kh.var(-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        kh_hat = (kh - mean) / rstd

        g = w * kh_hat + b - reconstruction_target
        g *= w
        v_new = (D * g - g.sum(-1, True) - kh_hat * (g * kh_hat).sum(-1, True)) / (rstd * D)

        Attn = torch.tril(q_i @ k_i.transpose(-2, -1))
        o_i = beta_T * q_i @ h - 2 * (f_i @ Attn) @ v_new
        h = beta_T * h - 2 * (f_i @ k_i).transpose(-1, -2) @ v_new
        # layer norm with residuals

        mean = o_i.mean(dim=-1, keepdim=True)
        var = o_i.var(dim=-1, unbiased=False, keepdim=True).to(torch.float32)
        rstd = torch.sqrt(var + eps).to(torch.float32)
        o[i] = o_i + (o_i - mean) / rstd * w + b

    # [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
    o = o.permute(1, 2, 0, 3, 4).reshape(B, H, T, D)
    h = h if output_final_state else None
    return o, h


# most of the code is copied from ttt
def chunk_titans_linear_ref(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        theta: torch.Tensor,
        alpha: torch.Tensor,
        eta: torch.Tensor,
        eps: float = 1e-6,
        BT: int = 16,  # chunk size
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        head_first: bool = True,
):
    assert q.dtype == k.dtype == v.dtype
    assert k.shape[-1] == v.shape[-1], "DK must equal to DV."
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        eta = theta.transpose(1, 2)
        alpha = alpha.transpose(1, 2)
        theta = theta.transpose(1, 2)
    seq_len = q.shape[-2]
    pad_len = (BT - (seq_len % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        theta = F.pad(theta, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, pad_len))
        eta = F.pad(eta, (0, 0, 0, pad_len))
        theta[:, :, -1, :] = theta[:, :, -(pad_len + 1), :]
        alpha[:, :, -1, :] = alpha[:, :, -(pad_len + 1), :]
        eta[:, :, -1, :] = eta[:, :, -(pad_len + 1), :]
    assert q.shape[-2] % BT == 0, "Sequence length should be a multiple of BT."
    q, k, v, w, b = map(lambda x: x.to(torch.float32), [q, k, v, w, b])
    o, final_state = titans_linear(q, k, v, w, b, theta, alpha, eta, eps, BT, initial_state, output_final_state)
    o = o[:, :, :seq_len, :]
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
