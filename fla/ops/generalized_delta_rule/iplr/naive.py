# -*- coding: utf-8 -*-

import torch
from einops import rearrange


# S_t = S_t @ (I + alpha_t beta_t^T) + v_t k_t^T
# q, k, alpha, beta [B, H, L, D_K]
# v [B, H, L, D_V]
def iplr_recurrence(q, k, v, alpha, beta, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
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
        _alpha = alpha[:, :, i]
        _beta = beta[:, :, i]
        _kv = _k[..., None] * _v[..., None, :] + (S.clone() * _alpha[..., None]).sum(-2, keepdim=True) * _beta[..., None]
        S = S + _kv
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def iplr_chunkwise(q, k, v, alpha, beta, initial_state=None, output_final_state=True, chunk_size=32):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v
    assert l % chunk_size == 0

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S += initial_state

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, alpha, beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, alpha, beta])

    v2 = (alpha @ k.transpose(-1, -2)).masked_fill_(mask, 0) @ v
    attn = (alpha @ beta.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    u = attn @ v2
    w = attn @ alpha
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i, u_i, w_i, beta_i = q[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], beta[:, :, i]
        o_1 = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0) @ v_i
        v2_i = u_i + w_i @ S
        o_2 = (q_i @ beta_i.transpose(-1, -2)).masked_fill_(mask, 0) @ (v2_i)
        o_3 = q_i @ S
        o[:, :, i] = o_1 + o_2 + o_3
        S = S + k_i.transpose(-1, -2) @ v_i + beta_i.transpose(-1, -2) @ v2_i
    S = None if output_final_state is False else S
    return rearrange(o, 'b h n c d -> b h (n c) d'), S


if __name__ == '__main__':
    from fla.utils import device
    B = 2
    H = 4
    L = 128
    DK = 128
    DV = 128
    q = (torch.randn(B, H, L, DK)).to(device).requires_grad_(True)
    k = (torch.randn(B, H, L, DK)).to(device).requires_grad_(True)
    v = (torch.randn(B, H, L, DV)).to(device).requires_grad_(True)
    alpha = torch.randn(B, H, L, DK).to(device).softmax(-1).requires_grad_(True)
    beta = torch.randn(B, H, L, DK).to(device).softmax(-1).requires_grad_(True)

    o, s = iplr_recurrence(q, k, v, -alpha, beta)
    do = torch.randn_like(o).to(device)
    o.backward(do, retain_graph=True)
    q_grad, q.grad = q.grad, None
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None

    o2, s2 = iplr_chunkwise(q, k, v, -alpha, beta)
    o2.backward(do)
    assert torch.allclose(o, o2, atol=1e-4), breakpoint()
    assert torch.allclose(s, s2, atol=1e-4), breakpoint()
    assert torch.allclose(q.grad, q_grad, atol=1e-4), breakpoint()
    assert torch.allclose(k.grad, k_grad, atol=1e-4), breakpoint()
    assert torch.allclose(v.grad, v_grad, atol=1e-4), breakpoint()
    assert torch.allclose(beta.grad, beta_grad, atol=1e-4), breakpoint()
    print("All passed!")
