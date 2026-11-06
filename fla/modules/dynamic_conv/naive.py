# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


@torch.autocast(device_type="cuda", enabled=False)
def naive_dynamic_conv1d(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    kernel_bias: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Materialized FP32 reference, including differentiable raw-input history."""
    B, T, D = x.shape
    R, W, _ = weight.shape
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        segments = [(0, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    else:
        segments = [(b, 0, T) for b in range(B)]
    basis = weight.float().reshape(R, W * D)
    static = kernel_bias.float() if kernel_bias is not None else None
    channel_bias = bias.float() if bias is not None else None
    outputs, states = [], []
    for n, (b, start, end) in enumerate(segments):
        values = x[b, start:end].float()
        history = values.new_zeros(W, D) if initial_state is None else initial_state[n].T.float()
        padded = torch.cat((history, values), dim=0)
        kernels = (z[b, start:end].float() @ basis).reshape(-1, W, D)
        if static is not None:
            kernels = kernels + static
        windows = padded.unfold(0, W, 1)[1:].flip(-1).transpose(-1, -2)
        output = (kernels * windows).sum(1)
        if channel_bias is not None:
            output = output + channel_bias
        if activation is not None:
            output = F.silu(output)
        if residual is not None:
            output = output + residual[b, start:end].float()
        outputs.append(output)
        states.append(padded[-W:].T)
    y = torch.cat(outputs, dim=0).reshape(B, T, D).to(x.dtype)
    ht = torch.stack(states).to(x.dtype) if output_final_state else None
    return y, ht
