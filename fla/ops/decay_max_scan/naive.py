# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def naive_decay_max_scan(
    x: torch.Tensor,
    decay_v: torch.Tensor | None = None,
    decay_b: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential reference for the decay-max scan.

    Recurrence per lane (b, d, h):

        g_t = exp(-softplus(x_t * v_d + b_d))     (g_t = 1 when decay disabled)
        e_t = max(x_t, g_t * e_{t-1}),            e_0 = -inf (or initial_state)

    Args:
        x (torch.Tensor):
            Scan inputs of shape `[B, T, D, H]`.
        decay_v (torch.Tensor, Optional):
            Per-channel decay slope `[D]`. `None` disables decay (pure running max). Default: `None`.
        decay_b (torch.Tensor, Optional):
            Per-channel decay bias `[D]`. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Running state `[B, D, H]`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence offsets for varlen packing; state resets at each boundary. Default: `None`.

    Returns:
        e (torch.Tensor):
            Scan outputs `[B, T, D, H]` in float32.
        final_state (torch.Tensor):
            Final running state `[B, D, H]`.
    """
    B, T, D, H = x.shape
    xf = x.float()
    if decay_v is not None:
        pre = xf * decay_v.float().view(1, 1, D, 1) + decay_b.float().view(1, 1, D, 1)
        g = torch.exp(-F.softplus(pre))
    else:
        g = torch.ones_like(xf)
    bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, T]
    e = torch.empty(B, T, D, H, dtype=torch.float32, device=x.device)
    # -1e30 (not -inf): keeps d * state gradients finite on the cold-start branch
    s = torch.full((B, D, H), -1e30, dtype=torch.float32, device=x.device) \
        if initial_state is None else initial_state.float().clone()
    for i in range(len(bounds) - 1):
        s = torch.full_like(s, -1e30) if i > 0 else s
        for t in range(bounds[i], bounds[i + 1]):
            s = torch.maximum(xf[:, t], g[:, t] * s)
            e[:, t] = s
    return e, s
