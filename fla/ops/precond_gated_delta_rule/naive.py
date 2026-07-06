# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_recurrent_precond_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_atk: torch.Tensor,
    g: torch.Tensor,
    beta_atk: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_A_state: torch.Tensor = None,
    output_final_state: bool = False,
    x: float = 1.5,
    eps: float = 1e-6,
    log_atk_scale: torch.Tensor | float = None,
):
    """
    Reference PyTorch implementation of recurrent preconditioned gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g_atk: [B, T, H]
        g: [B, T, H]
        beta_atk: [B, T, H]
        beta: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        initial_A_state: [B, H, K], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
        final_A_state: [B, H, K] if output_final_state else None
    """
    q, k, v, g_atk, g, beta_atk, beta = map(
        lambda t: t.transpose(1, 2).contiguous().to(torch.float32),
        [q, k, v, g_atk, g, beta_atk, beta]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    S = torch.zeros(B, H, K, V).to(v)
    A = torch.zeros(B, H, K).to(v)
    if initial_state is not None:
        S = initial_state.clone()
    if initial_A_state is not None:
        A = initial_A_state.clone()
    if scale is None:
        scale = 1 / (K ** 0.5)

    # L2 normalize q and k, then apply scale
    q = F.normalize(q, p=2, dim=-1) * scale
    k = F.normalize(k, p=2, dim=-1)

    if log_atk_scale is None:
        log_atk_scale = -0.2
    if isinstance(log_atk_scale, (int, float)):
        center = torch.tensor(log_atk_scale, dtype=torch.float32, device=k.device)
    else:
        center = log_atk_scale.to(torch.float32).view(1, H, 1)

    logx = torch.log(torch.tensor(x, dtype=torch.float32, device=k.device))

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        b_g_atk = g_atk[:, :, i]
        b_g = g[:, :, i]
        b_beta_atk = beta_atk[:, :, i]
        b_beta = beta[:, :, i]

        # ATK recurrence
        A = A.clone() * b_g_atk.exp().unsqueeze(-1) + b_beta_atk.unsqueeze(-1) * (b_k ** 2)

        # Symmetric fast squash
        ell = torch.log(A + eps)
        r = ell - center
        s = r / (1.0 + torch.abs(r))
        M = torch.exp(-logx * s)
        b_k_precond = b_k * M

        # Gated delta rule
        S = S.clone() * b_g.exp()[..., None, None]
        b_v = b_v - (S.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        S = S.clone() + b_k_precond.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, S)

    if not output_final_state:
        S = None
        A = None
    o = o.transpose(1, 2).contiguous()
    return o, S, A
