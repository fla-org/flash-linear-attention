# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_diag_kdn_gain(
    k: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    info_scale: float | None = None,
):
    r"""
    Sequential reference of the DiagKDN gain.

    .. math::
        z_t = \exp(2 g_t) \odot p_{t-1} + \omega_t, \quad
        \kappa_t = z_t \odot k_t / (r_t + \langle z_t, k_t^2 \rangle), \quad
        p_t = r_t z_t / (r_t + \text{info\_scale} \, k_t^2 \odot z_t),

    where ``p = 1 / c`` is the covariance of each key channel.

    Args:
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, H, K]``.
        omega (torch.Tensor):
            Process noise of shape ``[B, T, H, K]``.
        r (torch.Tensor):
            Observation noise of shape ``[B, T, H]``.
        initial_state (torch.Tensor, Optional):
            Initial precision of shape ``[B, H, K]``. A precision of 1 is used if `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key. Default: `None`, i.e., ``K``.

    Returns:
        A tuple ``(kappa, c)`` where ``kappa`` has shape ``[B, T, H, K]`` and
        ``c`` has shape ``[B, H, K]`` if ``output_final_state`` else `None`.
    """
    B, T, H, K = k.shape
    if info_scale is None:
        info_scale = K
    k, g, omega, r = map(lambda x: x.to(torch.float), [k, g, omega, r])
    a, k2, r = (2 * g).exp(), k.pow(2), r[..., None]

    p = k.new_ones(B, H, K) if initial_state is None else 1 / initial_state.to(torch.float)
    kappa = torch.zeros_like(k)
    for i in range(0, T):
        z = a[:, i] * p + omega[:, i]
        kappa[:, i] = z * k[:, i] / (r[:, i] + (z * k2[:, i]).sum(-1, keepdim=True))
        p = r[:, i] * z / (r[:, i] + info_scale * k2[:, i] * z)
    c = 1 / p if output_final_state else None
    return kappa, c


def naive_recurrent_diag_kdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    output_final_state: bool = False,
    info_scale: float | None = None,
):
    r"""
    Tokenwise reference of DiagKDN: the diagonal Kalman gain followed by the memory update

    .. math::
        S_t = \text{Diag}(\exp(g_t)) S_{t-1}, \quad
        S_t \leftarrow S_t + \kappa_t (v_t - S_t^\top k_t)^\top, \quad
        o_t = \text{scale} \cdot S_t^\top q_t.

    Keys are used as given, without normalization.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, H, K]``.
        omega (torch.Tensor):
            Process noise of shape ``[B, T, H, K]``.
        r (torch.Tensor):
            Observation noise of shape ``[B, T, H]``.
        scale (float, Optional):
            Scale factor of the attention scores. Default: `None`, i.e., ``1 / sqrt(K)``.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial memory of shape ``[B, H, K, V]`` and precision of shape ``[B, H, K]``.
            Either entry may be `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final memory and precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key. Default: `None`, i.e., ``K``.

    Returns:
        A tuple ``(o, (S, c))`` where ``o`` has shape ``[B, T, H, V]``,
        ``S`` has shape ``[B, H, K, V]`` and ``c`` has shape ``[B, H, K]``.
        The state is `None` if ``output_final_state=False``.
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    S0, c0 = initial_state if initial_state is not None else (None, None)
    kappa, c = naive_diag_kdn_gain(
        k=k,
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=True,
        info_scale=info_scale,
    )

    q, k, v, g = map(lambda x: x.to(torch.float), [q, k, v, g])
    S = k.new_zeros(B, H, K, V)
    if S0 is not None:
        S += S0
    o = torch.zeros_like(v)
    for i in range(0, T):
        S = S * g[:, i, ..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', kappa[:, i], v[:, i] - (k[:, i, ..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q[:, i] * scale, S)
    return o.to(dtype), ((S, c) if output_final_state else None)
