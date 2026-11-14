# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For the list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_triadic_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""
    Reference recurrence of Triadic KDA (https://github.com/fla-org/flash-linear-attention/issues/1730):
    Triadic Linear Attention with KDA's per-channel decay.

    Each head keeps `E` state slices `S_e` of shape `[K, V]`, written with the joint key
    `k2 ⊗ k` and read with the joint query `q2 ⊗ q`. Slice `e` decays by `exp(g[..., e])`
    and channel `c` of every slice decays by `exp(gk[..., c])`, so with `gk = 0` this is
    Triadic GDN.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        k2 (torch.Tensor):
            Second keys of shape `[B, T, H, E]`.
        q2 (torch.Tensor):
            Second queries of shape `[B, T, H, E]`.
        g (torch.Tensor):
            Per-slice decays in log space of shape `[B, T, H, E]`.
        gk (torch.Tensor):
            Per-channel decays in log space of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            Write strengths of shape `[B, T, H]`.
        scale (float, Optional):
            Scale factor of the output. Default: `1 / sqrt(K)`.
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[B, H, E, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final state. Default: `False`.

    Returns:
        A tuple `(o, S)` where `o` has shape `[B, T, H, V]` and
        `S` has shape `[B, H, E, K, V]` if `output_final_state` else `None`.
    """
    dtype = v.dtype
    B, T, H, K, V, E = *q.shape, v.shape[-1], k2.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, k2, q2, g, gk, beta = map(
        lambda x: x.to(torch.float), [q, k, v, k2, q2, g, gk, beta],
    )
    q = q * scale

    S = q.new_zeros(B, H, E, K, V)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        # joint key and query of shape [B, H, E, K]
        kappa = k2[:, i, ..., None] * k[:, i, :, None]
        rho = q2[:, i, ..., None] * q[:, i, :, None]
        # per-slice decay on the slice axis and per-channel decay on the key axis
        S = S * (g[:, i, ..., None, None] + gk[:, i, ..., None, :, None]).exp()
        v_i = beta[:, i, :, None] * (v[:, i] - torch.einsum('b h e k, b h e k v -> b h v', kappa, S))
        S = S + torch.einsum('b h e k, b h v -> b h e k v', kappa, v_i)
        o[:, i] = torch.einsum('b h e k, b h e k v -> b h v', rho, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_triadic_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""
    Reference chunkwise form of Triadic KDA.

    The per-channel decay multiplies the intra-chunk key products by `exp(Gk_i - Gk_j)`,
    so every slice still shares one `[C, C]` UT transform and only the state update and
    readout loop over slices. Both factors are formed from gate differences rather than as
    a product of `exp(G)` and `exp(-G)`, which can overflow.

    Takes the arguments of `naive_recurrent_triadic_kda` plus `chunk_size` (Default: 64)
    and returns the same outputs.
    """
    dtype = v.dtype
    B, T, H, K, V, E = *q.shape, v.shape[-1], k2.shape[-1]
    BT = chunk_size
    if scale is None:
        scale = K ** -0.5

    # padded steps have zero keys and gates, so they leave the state untouched
    pad = (BT - T % BT) % BT
    q, k, v, k2, q2, g, gk, beta = map(
        lambda x: F.pad(x.to(torch.float), (0, 0) * (x.ndim - 2) + (0, pad)).transpose(1, 2),
        [q, k, v, k2, q2, g, gk, beta],
    )
    q, k, v, k2, q2, g, gk = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT), [q, k, v, k2, q2, g, gk])
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=BT)
    q = q * scale
    g = g.cumsum(-2)
    gk = gk.cumsum(-2)

    mask = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=q.device))
    # [B, H, N, C, C, E]
    g_diff = (g[..., :, None, :] - g[..., None, :, :]).masked_fill(~mask[..., None], float('-inf')).exp()
    R = (k2[..., :, None, :] * k2[..., None, :, :] * g_diff).sum(-1)
    Rq = (q2[..., :, None, :] * k2[..., None, :, :] * g_diff).sum(-1)
    # [B, H, N, C, C, K]: the per-channel decay is per key channel, so the decayed inner
    # product <k_i, k_j>_Gk = sum_k k_ik k_jk exp(Gk_ik - Gk_jk) cannot be factored into
    # (k_i . k_j) times a scalar; it is formed from gate differences to avoid overflow.
    gk_diff = (gk[..., :, None, :] - gk[..., None, :, :]).masked_fill(~mask[..., None], float('-inf')).exp()
    kk = (k[..., :, None, :] * k[..., None, :, :] * gk_diff).sum(-1)
    qk = (q[..., :, None, :] * k[..., None, :, :] * gk_diff).sum(-1)
    A = (beta[..., None] * kk * R).tril(-1) + torch.eye(BT, device=q.device)
    Aqk = qk * Rq

    S = q.new_zeros(B, H, E, K, V)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(q.shape[2]):
        q_i, k_i, v_i, k2_i, q2_i, g_i, gk_i, b_i = (
            x[:, :, i] for x in (q, k, v, k2, q2, g, gk, beta)
        )
        # [B, H, C, E, V]: the per-channel decay enters before the key axis is summed away
        k_S = torch.einsum('b h c k, b h e k v -> b h c e v', k_i * gk_i.exp(), S)
        q_S = torch.einsum('b h c k, b h e k v -> b h c e v', q_i * gk_i.exp(), S)
        u_i = b_i[..., None] * (
            v_i - torch.einsum('b h c e, b h c e v -> b h c v', k2_i * g_i.exp(), k_S)
        )
        u_i = torch.linalg.solve_triangular(A[:, :, i], u_i, upper=False, unitriangular=True)
        o[:, :, i] = torch.einsum('b h c e, b h c e v -> b h c v', q2_i * g_i.exp(), q_S) + Aqk[:, :, i] @ u_i
        g_last = g_i[..., -1, :]
        gk_last = gk_i[..., -1, :]
        S = S * (g_last[..., None, None] + gk_last[..., None, :, None]).exp() + torch.einsum(
            'b h c k, b h c e, b h c v -> b h e k v',
            k_i * (gk_last[:, :, None, :] - gk_i).exp(),
            k2_i * (g_last[..., None, :] - g_i).exp(),
            u_i,
        )
    if not output_final_state:
        S = None
    o = rearrange(o, 'b h n c d -> b (n c) h d')[:, :T]
    return o.to(dtype), S
