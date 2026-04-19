# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from einops import rearrange


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.
        chunk_size (int):
            Chunk size for the chunked computation. Default: 64.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    # Rearrange into chunks: [B, head, NT, BT, ...]
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [v, g, beta]]
    # Expand q/k to value head dim for GVA: [B, H, ...] -> [B, HV, ...]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)          # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # Akk uses k (expanded to HV) and g (per value head)
    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        # [B, HV, BT, ...]
        q_i = q[:, :, i]      # [B, HV, BT, K]
        k_i = k[:, :, i]      # [B, HV, BT, K]
        u_i = u[:, :, i]        # [B, HV, BT, V]
        g_i = g[:, :, i]        # [B, HV, BT, K]
        w_i = w[:, :, i]        # [B, HV, BT, K]
        # Aqk: per value head (q from qk head, g from value head, k from qk head)
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S
