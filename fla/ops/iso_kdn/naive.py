# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.kda.naive import naive_recurrent_kda


def naive_iso_kdn_gain(
    k: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    info_scale: float | None = None,
):
    r"""
    Sequential reference of the IsoKDN gain for keys of arbitrary norm.

    .. math::
        a_t = \text{mean}(\exp(2 g_t)), \quad
        z_t = a_t / c_{t-1} + \omega_t, \quad
        \beta_t = z_t / (r_t + z_t \|k_t\|^2), \quad
        c_t = 1 / z_t + \text{info\_scale} \|k_t\|^2 / (K r_t).

    Args:
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, HV, K]``. ``HV`` must be divisible by ``H``.
        omega (torch.Tensor):
            Process noise of shape ``[B, T, HV]``.
        r (torch.Tensor):
            Observation noise of shape ``[B, T, HV]``.
        initial_state (torch.Tensor, Optional):
            Initial precision of shape ``[B, HV]``. A precision of 1 is used if `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key, relative to ``K``. Default: `None`, i.e., ``K``.

    Returns:
        A tuple ``(beta, c)`` where ``beta`` has shape ``[B, T, HV]`` and
        ``c`` has shape ``[B, HV]`` if ``output_final_state`` else `None`.
    """
    B, T, H, K, HV = *k.shape, g.shape[2]
    if info_scale is None:
        info_scale = K
    k, g, omega, r = map(lambda x: x.to(torch.float), [k, g, omega, r])
    k2 = k.repeat_interleave(HV // H, dim=2).pow(2).sum(-1)
    a = (2 * g).exp().mean(-1)

    c = k.new_ones(B, HV) if initial_state is None else initial_state.to(torch.float)
    beta = torch.zeros_like(omega)
    for i in range(0, T):
        z = a[:, i] / c + omega[:, i]
        beta[:, i] = z / (r[:, i] + z * k2[:, i])
        c = 1 / z + info_scale * k2[:, i] / (K * r[:, i])
    if not output_final_state:
        c = None
    return beta, c


def naive_recurrent_iso_kdn(
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
    Tokenwise reference of IsoKDN: the isotropic Kalman gain followed by the KDA memory recurrence.
    Keys are used as given, without normalization.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, HV, K]``.
        omega (torch.Tensor):
            Process noise of shape ``[B, T, HV]``.
        r (torch.Tensor):
            Observation noise of shape ``[B, T, HV]``.
        scale (float, Optional):
            Scale factor of the attention scores. Default: `None`, i.e., ``1 / sqrt(K)``.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial memory of shape ``[B, HV, K, V]`` and precision of shape ``[B, HV]``.
            Either entry may be `None`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final memory and precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key, relative to ``K``. Default: `None`, i.e., ``K``.

    Returns:
        A tuple ``(o, (S, c))`` where ``o`` has shape ``[B, T, HV, V]``,
        ``S`` has shape ``[B, HV, K, V]`` and ``c`` has shape ``[B, HV]``.
        The state is `None` if ``output_final_state=False``.
    """
    S0, c0 = initial_state if initial_state is not None else (None, None)
    beta, c = naive_iso_kdn_gain(
        k=k,
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=True,
        info_scale=info_scale,
    )
    o, S = naive_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=S0,
        output_final_state=True,
    )
    return o, ((S, c) if output_final_state else None)
