# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
from einops import repeat


def _init_state(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None:
        return initial_state
    state_dtype = torch.float32 if q.dtype in {torch.float16, torch.bfloat16} else q.dtype
    state_shape = (batch, heads)
    S = q.new_zeros((*state_shape, key_dim, key_dim), dtype=state_dtype)
    C = q.new_zeros((*state_shape, key_dim, value_dim), dtype=state_dtype)
    m = q.new_zeros((*state_shape, key_dim), dtype=state_dtype)
    G = q.new_zeros((*state_shape, key_dim, value_dim), dtype=state_dtype)
    h = q.new_zeros((*state_shape, key_dim), dtype=state_dtype)
    return S, C, m, G, h


def _signed_clamp_min(x: torch.Tensor, eps: float) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return sign * x.abs().clamp_min(eps)


def recurrent_hla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    normalize: bool = False,
    eps: float = 1e-6,
    ridge: float = 0.0,
    scale: float | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    r"""
    Reference recurrent implementation of masked second-order Higher-order Linear Attention.

    This implements the second-order masked streaming identity from
    "Higher-order Linear Attention" (arXiv:2510.27258):
    ``S_t = sum_{i<=t} k_i k_i^T``, ``C_t = sum_{i<=t} q_i v_i^T``,
    ``m_t = sum_{i<=t} q_i``, ``G_t = sum_{i<=t} k_i k_i^T C_{i-1}``,
    and ``h_t = sum_{i<=t} k_i k_i^T m_{i-1}``. The unnormalized output is
    ``o_t = q_t^T (S_t C_t - G_t)``; the optional normalized variant divides
    by ``q_t^T (S_t m_t - h_t)`` with a signed epsilon floor.

    Args:
        q:
            Queries of shape ``[B, T, H, K]``.
        k:
            Keys of shape ``[B, T, H, K]``.
        v:
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        initial_state:
            Optional tuple ``(S, C, m, G, h)`` with shapes ``[B, HV, K, K]``,
            ``[B, HV, K, V]``, ``[B, HV, K]``, ``[B, HV, K, V]``, and
            ``[B, HV, K]``.
        output_final_state:
            Whether to return the final recurrent state.
        normalize:
            Whether to divide by the optional masked denominator.
        eps:
            Numerical epsilon used by the normalized variant.
        ridge:
            Optional diagonal ridge added to ``S`` before querying. A nonzero value
            is a stabilized variant and no longer exactly equals the pure masked
            bilinear form.
        scale:
            Optional scale applied to queries. Defaults to ``K ** -0.5``.

    Returns:
        ``(o, final_state)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``final_state`` is the state tuple if requested, otherwise ``None``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must all be rank-4 tensors")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError(f"q/k and v must share [B, T], got {q.shape[:2]} and {v.shape[:2]}")
    if v.shape[2] % q.shape[2] != 0:
        raise ValueError(f"value heads ({v.shape[2]}) must be divisible by q/k heads ({q.shape[2]})")

    dtype = v.dtype
    key_dim = q.shape[-1]
    if scale is None:
        scale = key_dim ** -0.5
    if v.shape[2] != q.shape[2]:
        groups = v.shape[2] // q.shape[2]
        q = repeat(q, "b t h d -> b t (h g) d", g=groups)
        k = repeat(k, "b t h d -> b t (h g) d", g=groups)

    compute_dtype = torch.float32 if q.dtype in {torch.float16, torch.bfloat16} else q.dtype
    q = (q * scale).to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)
    S, C, m, G, h = _init_state(q, v, initial_state)
    S, C, m, G, h = (x.to(device=q.device, dtype=compute_dtype) for x in (S, C, m, G, h))

    outputs = []
    eye = None
    if ridge != 0.0:
        eye = torch.eye(key_dim, device=q.device, dtype=compute_dtype).view(1, 1, key_dim, key_dim)

    for idx in range(q.shape[1]):
        q_t = q[:, idx]
        k_t = k[:, idx]
        v_t = v[:, idx]

        S_prev, C_prev, m_prev, G_prev, h_prev = S, C, m, G, h
        dS = torch.einsum("bhk,bhd->bhkd", k_t, k_t)
        dC = torch.einsum("bhk,bhv->bhkv", q_t, v_t)
        dm = q_t

        S = S_prev + dS
        C = C_prev + dC
        m = m_prev + dm
        G = G_prev + torch.einsum("bhkd,bhdv->bhkv", dS, C_prev)
        h = h_prev + torch.einsum("bhkd,bhd->bhk", dS, m_prev)

        S_eff = S if eye is None else S + ridge * eye
        u = torch.einsum("bhk,bhkd->bhd", q_t, S_eff)
        o = torch.einsum("bhk,bhkv->bhv", u, C) - torch.einsum("bhk,bhkv->bhv", q_t, G)
        if normalize:
            den = torch.einsum("bhk,bhk->bh", u, m) - torch.einsum("bhk,bhk->bh", q_t, h)
            o = o / _signed_clamp_min(den, eps).unsqueeze(-1)
        outputs.append(o)

    output = torch.stack(outputs, dim=1).to(dtype)
    final_state = (S, C, m, G, h) if output_final_state else None
    return output, final_state
