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
    z: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference for low-rank dynamic depthwise causal convolution.

    For each sequence, with batch indices omitted:

        K[t, j, d] = sum_r z[t, r] * out_proj_weight[r, j, d] + out_proj_bias[j, d]
        y[t, d] = activation(sum_j K[t, j, d] * x[t-j, d]) + residual[t, d]

    Missing bias and residual terms are zero; activation=None is the identity.
    Negative input positions read from initial_state, or use zero padding when it is absent.

    Args:
        x (torch.Tensor):
            Convolution input [B, T, D].
        z (torch.Tensor):
            Generator latents [B, T, R].
        out_proj_weight (torch.Tensor):
            Generator output projection weight [R, W, D]. Tap j=0 uses the current token.
        out_proj_bias (torch.Tensor, Optional):
            Generator output projection bias [W, D]. Default: `None`.
        residual (torch.Tensor, Optional):
            Tensor [B, T, D] added after activation. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Input history [N, D, W], oldest token first; N=B for dense inputs. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the last W inputs, including initial history for short sequences. Default: `False`.
        activation (str, Optional):
            Convolution output activation: `None`, `"silu"`, or `"swish"`. Default: `None`.
        cu_seqlens (torch.Tensor, Optional):
            Packed sequence boundaries [N+1], requiring B=1. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            Output [B, T, D] and optional final history [N, D, W], both in x.dtype.
    """
    B, T, D = x.shape
    R, W, _ = out_proj_weight.shape
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        segments = [(0, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    else:
        segments = [(b, 0, T) for b in range(B)]
    proj_weight = out_proj_weight.float().reshape(R, W * D)
    proj_bias = out_proj_bias.float() if out_proj_bias is not None else None
    outputs, states = [], []
    for n, (b, start, end) in enumerate(segments):
        values = x[b, start:end].float()
        history = values.new_zeros(W, D) if initial_state is None else initial_state[n].T.float()
        padded = torch.cat((history, values), dim=0)
        kernels = (z[b, start:end].float() @ proj_weight).reshape(-1, W, D)
        if proj_bias is not None:
            kernels = kernels + proj_bias
        windows = padded.unfold(0, W, 1)[1:].flip(-1).transpose(-1, -2)
        output = (kernels * windows).sum(1)
        if activation is not None:
            output = F.silu(output)
        if residual is not None:
            output = output + residual[b, start:end].float()
        outputs.append(output)
        states.append(padded[-W:].T)
    y = torch.cat(outputs, dim=0).reshape(B, T, D).to(x.dtype)
    ht = torch.stack(states).to(x.dtype) if output_final_state else None
    return y, ht
