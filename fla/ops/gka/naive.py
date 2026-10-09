# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def _solve_exact(h_kk: torch.Tensor, lamb: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Returns `x` solving `(h_kk + lamb * I) x = q` exactly.
    `h_kk` is `[B, H, K, K]`, `lamb` is `[B, H]`, and `q` and `x` are `[B, H, K]`.
    """
    eye = torch.eye(h_kk.shape[-1], dtype=h_kk.dtype, device=h_kk.device)
    return torch.linalg.solve(h_kk + lamb[..., None, None] * eye, q.unsqueeze(-1)).squeeze(-1)


def _solve_chebyshev(
    h_kk: torch.Tensor,
    lamb: torch.Tensor,
    fro: torch.Tensor,
    q: torch.Tensor,
    num_iter: int,
) -> torch.Tensor:
    """
    Returns `x` solving `(h_kk + lamb * I) x = q` approximately, with `num_iter` Chebyshev iterations,
    given `fro = ||h_kk||_F`. Shapes as in `_solve_exact`.
    """
    # the spectrum of `h_kk + lamb * I` lies in `[lamb, lamb + fro]`, since `fro` bounds the spectral norm
    stepsize = (2 / (2 * lamb + fro))[..., None]
    rho_sq_4 = ((fro / (2 * lamb + fro) / 2) ** 2)[..., None]
    x_prev = torch.zeros_like(q)
    x = stepsize * q
    omega = torch.full_like(stepsize, 2.0)
    for _ in range(num_iter):
        omega = 1 / (1 - rho_sq_4 * omega)
        residual = torch.einsum('b h i j, b h j -> b h i', h_kk, x) + lamb[..., None] * x - q
        x, x_prev = omega * x - (stepsize * omega * residual + (omega - 1) * x_prev), x
    return x


def _naive_recurrent_gka(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    alpha: torch.Tensor | None,
    scale: float | None,
    ridge_ratio: float,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    output_final_state: bool,
    num_iter: int | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Token-by-token GKA recurrence shared by `naive_recurrent_gka` and `naive_recurrent_gka_chebyshev`;
    `num_iter=None` solves the ridge system exactly.
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    # float64 inputs stay in float64 so that gradients can be checked at full precision
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = (x.to(compute_dtype) for x in (q, k, v))
    g = g.to(compute_dtype) if g is not None else None
    gk = gk.to(compute_dtype) if gk is not None else None
    alpha = alpha.to(compute_dtype) if alpha is not None else None

    if initial_state is not None:
        h_kk, h_kv = (s.to(compute_dtype) for s in initial_state)
    else:
        h_kk = q.new_zeros(B, H, K, K)
        h_kv = q.new_zeros(B, H, K, V)

    o = []
    for t in range(T):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        if gk is not None:
            h_kk = h_kk * gk[:, t].exp()[..., None, None]
        if g is not None:
            h_kv = h_kv * g[:, t].exp()[..., None, None]
        h_kk = h_kk + k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        h_kv = h_kv + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        fro = torch.linalg.matrix_norm(h_kk, ord='fro')
        lamb = ridge_ratio * fro
        if num_iter is None:
            x_t = _solve_exact(h_kk, lamb, q_t)
        else:
            x_t = _solve_chebyshev(h_kk, lamb, fro, q_t, num_iter)

        q_t = x_t if alpha is None else q_t + alpha[:, t, :, None] * (x_t - q_t)
        o.append(scale * torch.einsum('b h k, b h k v -> b h v', q_t, h_kv))
    o = torch.stack(o, dim=1)

    final_state = (h_kk, h_kv) if output_final_state else None
    return o.to(dtype), final_state


def naive_recurrent_gka(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    scale: float | None = None,
    ridge_ratio: float = 0.02,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
):
    r"""
    Token-by-token GKA reference that solves each ridge system exactly with `torch.linalg.solve`.
    At every token it updates `h_kk <- exp(gk_t) h_kk + k_t k_t^T` and `h_kv <- exp(g_t) h_kv + k_t v_t^T`.

    Its autograd gradients are the exact gradients of the ridge solution, including the path through
    `lamb = ridge_ratio * ||h_kk||_F`, which is what the chunked kernels implement.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor, Optional):
            Log-space decay of the value state `h_kv`, of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Log-space decay of `h_kk`, of shape `[B, T, H]`.
            Pass the same tensor as `g` to decay both states together. Default: `None`.
        alpha (torch.Tensor, Optional):
            Mixing weights of shape `[B, T, H]`; the readout query is `q + alpha * (x - q)`.
            If `None`, the readout query is the ridge solution `x`, as with `alpha = 1`. Default: `None`.
        scale (float, Optional):
            Scale applied to the readout. Default: `1 / sqrt(K)`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb = ridge_ratio * ||h_kk||_F`. Default: `0.02`.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial states `(h_kk, h_kv)` of shapes `[B, H, K, K]` and `[B, H, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final states. Default: `False`.

    Returns:
        A tuple `(o, final_state)` where `o` has shape `[B, T, H, V]` and `final_state` is `(h_kk, h_kv)`
        if `output_final_state` else `None`.
    """
    return _naive_recurrent_gka(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        alpha=alpha,
        scale=scale,
        ridge_ratio=ridge_ratio,
        initial_state=initial_state,
        output_final_state=output_final_state,
        num_iter=None,
    )


def naive_recurrent_gka_chebyshev(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    scale: float | None = None,
    ridge_ratio: float = 0.02,
    num_iter: int = 30,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
):
    r"""
    Token-by-token GKA reference that runs the same Chebyshev iteration as the kernels.
    At every token it updates `h_kk <- exp(gk_t) h_kk + k_t k_t^T` and `h_kv <- exp(g_t) h_kv + k_t v_t^T`.

    Its outputs match the kernels' at a fixed `num_iter`. Its gradients do not since autograd differentiates through the
    iterations, while the kernels' backward uses the exact gradient of the ridge solution, as `naive_recurrent_gka`
    does.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor, Optional):
            Log-space decay of the value state `h_kv`, of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Log-space decay of `h_kk`, of shape `[B, T, H]`.
            Pass the same tensor as `g` to decay both states together. Default: `None`.
        alpha (torch.Tensor, Optional):
            Mixing weights of shape `[B, T, H]`; the readout query is `q + alpha * (x - q)`.
            If `None`, the readout query is the ridge solution `x`, as with `alpha = 1`. Default: `None`.
        scale (float, Optional):
            Scale applied to the readout. Default: `1 / sqrt(K)`.
        ridge_ratio (float, Optional):
            Sets the ridge strength `lamb = ridge_ratio * ||h_kk||_F`. Default: `0.02`.
        num_iter (int, Optional):
            Number of Chebyshev iterations. Default: `30`.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial states `(h_kk, h_kv)` of shapes `[B, H, K, K]` and `[B, H, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final states. Default: `False`.

    Returns:
        A tuple `(o, final_state)` where `o` has shape `[B, T, H, V]` and `final_state` is `(h_kk, h_kv)`
        if `output_final_state` else `None`.
    """
    return _naive_recurrent_gka(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        alpha=alpha,
        scale=scale,
        ridge_ratio=ridge_ratio,
        initial_state=initial_state,
        output_final_state=output_final_state,
        num_iter=num_iter,
    )
