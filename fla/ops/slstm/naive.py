# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_recurrent_slstm(
    x: torch.Tensor,
    R: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    """Token-by-token sLSTM in fp32, with the same layout as `fused_recurrent_slstm`."""
    B, T, _, H, D = x.shape
    if initial_state is None:
        h, c, n, m = (x.new_zeros(B, H, D, dtype=torch.float) for _ in range(4))
    else:
        h, c, n, m = (s.float() for s in initial_state)
    ys = []
    for t in range(T):
        ri, rf, rz, ro = (x[:, t].float() + torch.einsum('bhk,hgkj->bghj', h, R.float())).unbind(1)
        logf = F.logsigmoid(rf)
        m_new = torch.maximum(ri, logf + m)
        i, f = torch.exp(ri - m_new), torch.exp(logf + m - m_new)
        c = f * c + i * torch.tanh(rz)
        n = f * n + i
        m = m_new
        h = torch.sigmoid(ro) * c / n
        ys.append(h)
    return torch.stack(ys, 1).to(x.dtype), ((h, c, n, m) if output_final_state else None)
