# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from fla.utils import input_guard


@input_guard
def fused_recurrent_decay_max_scan(
    x: torch.Tensor,
    decay_v: torch.Tensor | None = None,
    decay_b: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step (O(1) decoding) decay-max scan update.

    Args:
        x (torch.Tensor):
            Current-token inputs `[B, 1, D, H]`.
        decay_v (torch.Tensor, Optional):
            Per-channel decay slope `[D]`. Default: `None`.
        decay_b (torch.Tensor, Optional):
            Per-channel decay bias `[D]`. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Running state `[B, D, H]`; zeros channel treated as cold start if `None`. Default: `None`.

    Returns:
        e (torch.Tensor):
            Updated outputs `[B, 1, D, H]`.
        final_state (torch.Tensor):
            Updated running state `[B, D, H]`.
    """
    D = x.shape[2]
    xf = x.float()
    if decay_v is not None:
        pre = xf * decay_v.float().view(1, 1, D, 1) + decay_b.float().view(1, 1, D, 1)
        g = torch.exp(-F.softplus(pre))
    else:
        g = torch.ones_like(xf)
    s = torch.full((x.shape[0], D, x.shape[3]), float('-inf'), dtype=torch.float32, device=x.device) \
        if initial_state is None else initial_state.float()
    s = torch.maximum(xf[:, 0], g[:, 0] * s)
    return s.unsqueeze(1), s
