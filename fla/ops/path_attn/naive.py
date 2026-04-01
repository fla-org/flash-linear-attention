# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_path_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
):
    """
    Reference PyTorch implementation of path attention.

    Args:
        q: [B, T, HQ, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        w: [B, T, H, D]
        beta: [B, T, H]
        g: [B, T, HQ]
        scale: float
        chunk_size: int, default 64

    Returns:
        output: [B, T, HQ, D]
    """
    original_dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    BT = chunk_size

    q, k, v, w, beta, g = map(lambda x: x.to(torch.float).transpose(1, 2), [q, k, v, w, beta, g])
    g_cumsum = g.cumsum(-1)

    q = q.unsqueeze(2).expand(-1, -1, HQ // HQ, -1, -1).flatten(1, 2)
    k = k.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    w = w.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    beta = beta.unsqueeze(2).expand(-1, -1, HQ // H, -1).flatten(1, 2)

    b, h, l, _ = q.shape
    if l % BT != 0:
        padding_size = BT - l % BT
        q, k, w = map(lambda x: F.pad(x, (0, 0, 0, padding_size)), [q, k, w])
        beta = F.pad(beta, (0, padding_size))

    seq_len = q.shape[2]
    w_beta = w * beta[..., None]
    q, k, w, w_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT), [q, k, w, w_beta])

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    T_mat = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)

    for i in range(1, BT):
        T_mat[..., i, :i] = T_mat[..., i, :i].clone() + (T_mat[..., i, :, None].clone() * T_mat[..., :, :i].clone()).sum(-2)

    T_mat = T_mat + torch.eye(BT, dtype=q.dtype, device=q.device)
    Twbk = T_mat @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    qw = (q @ w.transpose(-1, -2)).tril()
    Twb = T_mat @ w_beta
    A_local = (q @ k.transpose(-1, -2)).tril() - qw @ Twbk
    q = q - qw @ Twb
    k = k - Twbk.transpose(-1, -2) @ w
    H_mat = w.transpose(-1, -2) @ Twb

    A = torch.zeros(b, h, seq_len, seq_len, device=q.device)
    q, k, w, w_beta = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, w, w_beta])

    for i in range(0, seq_len, BT):
        q_i = q[:, :, i:i+BT].clone()
        for j in range(i - BT, -BT, -BT):
            k_j = k[:, :, j:j+BT]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BT, j:j+BT] = A_ij
            q_i = q_i - q_i @ H_mat[:, :, j // BT]

    for i in range(0, seq_len // BT):
        A[:, :, i*BT:i*BT+BT, i*BT:i*BT+BT] = A_local[:, :, i]

    A = A.masked_fill_(~torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)), float("-inf"))
    A = A[:, :, :l, :l]
    A = A + g_cumsum[..., None] - g_cumsum[..., None, :]
    ref_o = (A * scale).softmax(-1).to(v) @ v

    return ref_o.to(original_dtype).transpose(1, 2)
