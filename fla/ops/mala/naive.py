# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import torch.nn.functional as F

"""
MALA (Magnitude-Aware Linear Attention) implementation.

Based on the paper:
"Rectifying Magnitude Neglect in Linear Attention"
ICCV 2025 (highlight)
https://arxiv.org/abs/2507.00698

Original implementation:
https://github.com/aldjalkdf/MAViT
"""

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def naive_mala_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sin: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    scale: float | None = None,
):
    """
    Naive implementation of MALA (Magnitude-Aware Linear Attention).
    
    Based on the paper:
    "Rectifying Magnitude Neglect in Linear Attention"
    ICCV 2025 (highlight)
    https://arxiv.org/abs/2507.00698
    
    Args:
        q (torch.Tensor): Queries of shape [B, T, H, K].
        k (torch.Tensor): Keys of shape [B, T, H, K].
        v (torch.Tensor): Values of shape [B, T, H, V].
        sin (torch.Tensor, optional): Sinusoidal position embeddings of shape [T, K].
        cos (torch.Tensor, optional): Cosine position embeddings of shape [T, K].
        scale (float, optional): Scale factor for attention scores.
        
    Returns:
        torch.Tensor: Output of shape [B, T, H, V].
    """
    dtype = q.dtype
    q, k, v = map(lambda x: x.float(), (q, k, v))
    B, T, H, K, V = *q.shape, v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    # Apply RoPE if provided (before computing normalization factor)
    if sin is not None and cos is not None:
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)
    
    # Apply ELU activation and add 1
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    
    # Compute normalization factor
    z = q @ k.mean(dim=1, keepdim=True).transpose(-1, -2) * scale
    
    # Compute key-value product (sum over sequence dimension)
    kv = (k.transpose(-2, -1) @ v) * (scale ** 2 / T)
    
    # Compute attention output
    res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=1, keepdim=True)
    
    return res.to(dtype)
