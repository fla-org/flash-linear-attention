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
    w2: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute dynamic depthwise causal convolution with PyTorch.

    .. math::
        w_t = \operatorname{reshape}(W_2 z_t + b, [D, W]),

        y_t = \psi\!\left(\sum_{j=0}^{W-1} w_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Psi is activation; `None` selects the identity.
    Outputs and final states use the input dtype.
    Negative positions read initial_state or zero padding; packed sequences have separate histories.

    Args:
        x (torch.Tensor):
            Convolution input [B, T, D]. Packed inputs require B=1.
        z (torch.Tensor):
            Projected conditioning input [B, T, R], after the optional intermediate activation.
        w2 (torch.Tensor):
            Second projection W_2 with shape [D * W, R].
        b (torch.Tensor):
            Second projection bias b with shape [D * W].
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
        chunk_indices (torch.LongTensor, Optional):
            Unused; accepted to match the Triton interface. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            Output [B, T, D] and final history [N, D, W] when requested, otherwise `None`.
    """
    dtype = x.dtype
    B, T, D = x.shape
    W = w2.shape[0] // D
    w = F.linear(input=z, weight=w2, bias=b).reshape(B * T, D, W)
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
