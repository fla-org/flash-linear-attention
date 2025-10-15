

import torch
from einops import rearrange


def naive_parallel_based(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    use_norm: bool = True,
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = 1 + attn + 1/2 * (attn ** 2)
    attn.masked_fill_(~torch.tril(torch.ones(
        q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    if use_norm:
        z = attn.sum(-1)
        return o / (z[..., None] + 1e-6)
    else:
        return o


def naive_chunk_based(q, k, v, chunk_size=256):
    q = q * (q.shape[-1] ** -0.5)
    # compute normalizer.
    k_cumsum = torch.cumsum(k, dim=-2)
    kk_cumsum = torch.cumsum(k.unsqueeze(-1) * k.unsqueeze(-2), dim=-3)
    # first
    z = (q * k_cumsum).sum(-1)
    # second order
    z += (q.unsqueeze(-1) * q.unsqueeze(-2) * kk_cumsum).sum((-1, -2)) * 0.5
    # zero-th order
    z += (torch.arange(0, q.shape[-2]).to(z.device) * 1.0 + 1.0)[None, None, :]

    # compute o
    # constant term
    _o = v.cumsum(-2)

    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)

    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)

    intra_chunk_attn = q @ k.transpose(-2, -1)
    intra_chunk_attn = intra_chunk_attn + 1/2 * (intra_chunk_attn ** 2)
    intra_chunk_attn.masked_fill_(~torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)), 0)
    o = intra_chunk_attn @ v

    # quadractic term
    kv = torch.einsum('b h n c x, b h n c y, b h n c z -> b h n x y z', k, k, v)
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)

    o += 0.5 * torch.einsum('b h n x y z, b h n c x, b h n c y -> b h n c z', kv, q, q)

    # linear term
    kv = torch.einsum('b h n c x, b h n c y -> b h n x y', k, v)
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    o += torch.einsum('b h n x y, b h n c x -> b h n c y', kv, q)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o + _o
    return o / (z[..., None] + 1e-6)
