# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_recurrent_precond_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_atk: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_A_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor | float = None,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, g_atk, beta_atk, beta = map(
        lambda x: x.to(torch.float32),
        [q, k, v, g, g_atk, beta_atk, beta]
    )
    q = q * scale

    if log_atk_scale is None:
        log_atk_scale = -0.2
    if isinstance(log_atk_scale, (int, float)):
        center = torch.tensor(log_atk_scale, dtype=torch.float32, device=k.device)
    else:
        center = log_atk_scale.to(torch.float32).view(1, H, 1)

    logx = torch.log(torch.tensor(x, dtype=torch.float32, device=k.device))

    A = k.new_zeros(B, H, K)
    if initial_A_state is not None:
        A += initial_A_state

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state

    o = torch.zeros_like(v)

    for i in range(T):
        q_i, k_i, v_i, g_i = q[:, i], k[:, i], v[:, i], g[:, i]
        g_atk_i, beta_atk_i, beta_i = g_atk[:, i], beta_atk[:, i], beta[:, i]

        A = g_atk_i.exp().unsqueeze(-1) * A + beta_atk_i.unsqueeze(-1) * (k_i ** 2)

        ell = torch.log(A + eps)
        r = ell - center
        s = r / (1.0 + torch.abs(r))
        M = torch.exp(-logx * s)
        k_precond = k_i * M

        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', beta_i[..., None] * k_precond, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

    if not output_final_state:
        S = None
        A = None
    return o.to(dtype), S, A
