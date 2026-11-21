# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_recurrent_mamba(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Token-by-token selective scan in fp32, with the same layout as `fused_recurrent_mamba`."""
    dtype = x.dtype
    x, dt, A, B, C = x.float(), dt.float(), A.float(), B.float(), C.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)
    h = x.new_zeros(x.shape[0], *A.shape) if initial_state is None else initial_state.float()
    ys = []
    for t in range(x.shape[1]):
        h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t] * x[:, t])[..., None] * B[:, t, None, :]
        ys.append((h * C[:, t, None, :]).sum(-1))
    y = torch.stack(ys, 1)
    if D is not None:
        y = y + x * D.float()
    return y.to(dtype), (h if output_final_state else None)
