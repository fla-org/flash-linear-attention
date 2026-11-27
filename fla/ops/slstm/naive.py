# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def _naive_recurrent_slstm_dense(
    x: torch.Tensor,
    recurrent_weight: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    B, T, _, H, D = x.shape
    if initial_state is None:
        h, c, n, m = (x.new_zeros(B, H, D, dtype=torch.float) for _ in range(4))
    else:
        h, c, n, m = (s.float() for s in initial_state)
    rw = recurrent_weight.float()
    ys = []
    for t in range(T):
        ri, rf, rz, ro = (x[:, t].float() + torch.einsum('bhk,hgkj->bghj', h, rw)).unbind(1)
        logf = F.logsigmoid(rf)
        m_new = torch.maximum(ri, logf + m)
        i, f = torch.exp(ri - m_new), torch.exp(logf + m - m_new)
        c = f * c + i * torch.tanh(rz)
        n = f * n + i
        m = m_new
        h = torch.sigmoid(ro) * c / n
        ys.append(h)
    y = torch.stack(ys, 1).to(x.dtype) if ys else x.new_empty(B, 0, H, D)
    return y, ((h, c, n, m) if output_final_state else None)


def naive_recurrent_slstm(
    x: torch.Tensor,
    recurrent_weight: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Token-by-token sLSTM in fp32, with the same layout and varlen contract as `fused_recurrent_slstm`."""
    if cu_seqlens is None:
        return _naive_recurrent_slstm_dense(
            x=x,
            recurrent_weight=recurrent_weight,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    if x.shape[0] != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {x.shape[0]} when using `cu_seqlens`.")
    ys = []
    hts = []
    for i in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        h0_i = tuple(s[i:i + 1] for s in initial_state) if initial_state is not None else None
        y_i, ht_i = _naive_recurrent_slstm_dense(
            x=x[:, bos:eos],
            recurrent_weight=recurrent_weight,
            initial_state=h0_i,
            output_final_state=output_final_state,
        )
        ys.append(y_i)
        if output_final_state:
            hts.append(ht_i)
    y = torch.cat(ys, dim=1)
    final_state = tuple(torch.cat(s, dim=0) for s in zip(*hts)) if output_final_state else None
    return y, final_state
