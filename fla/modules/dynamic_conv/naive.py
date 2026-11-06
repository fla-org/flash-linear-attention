# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F


def naive_dynamic_conv1d(
    x: torch.Tensor,
    h: torch.Tensor | None,
    in_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    dynamic_activation: str | None = None,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute dynamic depthwise causal convolution with PyTorch.

    .. math::
        z_t = \phi(A h_t), \qquad K_t = \operatorname{reshape}(B z_t + b, [D, W]),

        y_t = \psi\!\left(\sum_{j=0}^{W-1} K_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Phi is dynamic_activation and psi is activation; `None` selects the identity.
    Projections and convolution use FP32. Outputs and final states use the input dtype.
    Negative positions read initial_state or zero padding; packed sequences have separate histories.

    Args:
        x (torch.Tensor):
            Convolution input [B, T, D]. Packed inputs require B=1.
        h (torch.Tensor | None):
            Conditioning input [B, T, D]. `None` uses x.
        in_proj_weight (torch.Tensor):
            First projection A with shape [rank, D].
        out_proj_weight (torch.Tensor):
            Second projection B with shape [D * W, rank].
        out_proj_bias (torch.Tensor):
            Second projection bias b with shape [D * W].
        residual (torch.Tensor, Optional):
            Tensor [B, T, D] added after the output activation. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Input history [N, D, W], ordered oldest to newest. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final input history. Default: `False`.
        dynamic_activation (str, Optional):
            Activation between projections: `None`, `"silu"`, or `"swish"`. Default: `None`.
        activation (str, Optional):
            Convolution output activation: `None`, `"silu"`, or `"swish"`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Packed sequence boundaries [N+1]. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            Output [B, T, D] and final history [N, D, W] when requested, otherwise `None`.
    """
    B, T, D = x.shape
    W = out_proj_weight.shape[0] // D
    h = x if h is None else h
    z = F.linear(input=h.float(), weight=in_proj_weight.float())
    if dynamic_activation in ('silu', 'swish'):
        z = F.silu(z)
    weight = F.linear(input=z, weight=out_proj_weight.float(), bias=out_proj_bias.float()).reshape(B, T, D, W)

    if cu_seqlens is None:
        sequences = [(i, 0, T) for i in range(B)]
    else:
        boundaries = cu_seqlens.tolist()
        sequences = [(0, bos, eos) for bos, eos in zip(boundaries[:-1], boundaries[1:])]
    outputs, states = [], []
    for i_n, (i_b, bos, eos) in enumerate(sequences):
        values = x[i_b, bos:eos].float()
        history = values.new_zeros(W, D) if initial_state is None else initial_state[i_n].T.float()
        values = torch.cat((history, values), dim=0)
        y = (weight[i_b, bos:eos] * values.unfold(0, W, 1)[1:]).sum(-1)
        if activation in ('silu', 'swish'):
            y = F.silu(y)
        if residual is not None:
            y = y + residual[i_b, bos:eos].float()
        outputs.append(y)
        if output_final_state:
            states.append(values[-W:].T)
    y = torch.cat(outputs, dim=0).reshape(B, T, D).to(x.dtype)
    final_state = torch.stack(states).to(x.dtype) if output_final_state else None
    return y, final_state
