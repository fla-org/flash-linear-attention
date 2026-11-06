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
    dynamic_proj_w1: torch.Tensor,
    dynamic_proj_w2: torch.Tensor,
    dynamic_proj_bias: torch.Tensor,
    dynamic_activation: str | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute dynamic depthwise causal convolution with PyTorch.

    Let D be the channel count, R the projection rank, and W the convolution width.

    .. math::
        z_t = \phi(W_1 h_t), \qquad w_t = \operatorname{reshape}(W_2 z_t + b, [D, W]),

        y_t = \psi\!\left(\sum_{j=0}^{W-1} w_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Phi is dynamic_activation and psi is activation; `None` selects the identity.
    Outputs and final states use the input dtype.
    Negative positions read initial_state or zero padding; packed sequences have separate histories.

    Args:
        x (torch.Tensor):
            Convolution input [B, T, D]. Packed inputs require B=1.
        h (torch.Tensor | None):
            Conditioning input [B, T, D]. `None` uses x.
        dynamic_proj_w1 (torch.Tensor):
            First projection W_1 with shape [R, D], where 16 <= R <= 128.
        dynamic_proj_w2 (torch.Tensor):
            Second projection W_2 with shape [D * W, R].
        dynamic_proj_bias (torch.Tensor):
            Second projection bias b with shape [D * W].
        dynamic_activation (str, Optional):
            Activation between projections: `None`, `"silu"`, or `"swish"`. Default: `None`.
        residual (torch.Tensor, Optional):
            Tensor [B, T, D] added after the output activation. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Input history [N, D, W], ordered oldest to newest. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final input history. Default: `False`.
        activation (str, Optional):
            Convolution output activation: `None`, `"silu"`, or `"swish"`. Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Packed sequence boundaries [N+1]. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            Output [B, T, D] and final history [N, D, W] when requested, otherwise `None`.
    """
    dtype = x.dtype
    B, T, D = x.shape
    W = dynamic_proj_w2.shape[0] // D
    h = x if h is None else h
    z = F.linear(input=h, weight=dynamic_proj_w1)
    if dynamic_activation in ('silu', 'swish'):
        z = F.silu(z)
    w = F.linear(input=z, weight=dynamic_proj_w2, bias=dynamic_proj_bias).reshape(B * T, D, W)
    x, w = x.float(), w.float()

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.tolist()
    x = x.reshape(B * T, D)
    y = torch.empty_like(x)
    ht = x.new_empty(N, D, W) if output_final_state else None
    for i_n in range(N):
        if cu_seqlens is None:
            bos, eos = i_n * T, (i_n + 1) * T
        else:
            bos, eos = cu_seqlens[i_n:i_n+2]
        b_h0 = x.new_zeros(W, D) if initial_state is None else initial_state[i_n].T
        b_x = torch.cat((b_h0, x[bos:eos]), dim=0)
        b_w = w[bos:eos]
        y[bos:eos] = (b_w * b_x.unfold(dimension=0, size=W, step=1)[1:]).sum(-1)
        if ht is not None:
            ht[i_n] = b_x[-W:].T

    y = y.reshape(B, T, D)
    if activation in ('silu', 'swish'):
        y = F.silu(y)
    if residual is not None:
        y = y + residual
    return y.to(dtype), ht.to(dtype) if ht is not None else None
