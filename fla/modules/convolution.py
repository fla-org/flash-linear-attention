# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from fla.modules.conv import (
    ImplicitLongConvolution,
    LongConvolution,
    PositionalEmbedding,
    ShortConvolution,
    causal_conv1d,
    fft_conv,
)
from fla.modules.conv.cp import CausalConv1dFunctionCP, causal_conv1d_cp
from fla.modules.conv.cuda import FastCausalConv1dFn, fast_causal_conv1d_fn
from fla.modules.conv.triton import (
    CausalConv1dFunction,
    causal_conv1d_bwd,
    causal_conv1d_fwd,
    causal_conv1d_update,
    causal_conv1d_update_states,
)


def can_fuse_qkv_short_conv(
    q_conv: ShortConvolution,
    k_conv: ShortConvolution,
    v_conv: ShortConvolution,
) -> bool:
    """Whether the q/k/v short convolutions can collapse into a single call.

    The three depthwise convolutions can only be merged when they share the
    same backend, activation, and kernel size, so that one `causal_conv1d`
    over the concatenated channels is exactly equivalent to running them
    separately. Layers should additionally gate this on the dense no-cache
    path (no `last_state`, no `use_cache`, no `cu_seqlens`) before calling
    `fused_qkv_short_conv`.
    """
    return (
        q_conv.backend == k_conv.backend == v_conv.backend
        and q_conv.activation == k_conv.activation == v_conv.activation
        and q_conv.kernel_size == k_conv.kernel_size == v_conv.kernel_size
    )


def fused_qkv_short_conv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_conv: ShortConvolution,
    k_conv: ShortConvolution,
    v_conv: ShortConvolution,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the q/k/v short convolutions as a single `causal_conv1d`.

    This is the dense, cache-less fast path shared by all q/k/v short-conv
    layers. It concatenates the projected q/k/v along the channel dimension,
    concatenates the depthwise weights (and biases, when present) in the same
    order, and issues one convolution before splitting the result back into
    q/k/v. The split sizes are read from the convolution weights, so grouped
    (GQA) and multi-householder channel counts are handled transparently.

    It mirrors `ShortConvolution.forward` for the dense path: an optional
    attention mask zeroes out padded positions before the convolution, and the
    convs' shared activation/backend are applied inside `causal_conv1d`. Any
    layer-specific post-activation (for example an extra `F.silu`) stays in the
    layer. Callers must guard this with `can_fuse_qkv_short_conv` and only
    invoke it when there is no cache or variable-length state.
    """
    qkv = torch.cat([q, k, v], dim=-1)
    if mask is not None:
        qkv.mul_(mask.unsqueeze(-1))
    qkv_weight = torch.cat(
        [
            q_conv.weight.squeeze(1),
            k_conv.weight.squeeze(1),
            v_conv.weight.squeeze(1),
        ],
        dim=0,
    )
    qkv_bias = None
    if q_conv.bias is not None:
        qkv_bias = torch.cat([q_conv.bias, k_conv.bias, v_conv.bias], dim=0)
    qkv, _ = causal_conv1d(
        x=qkv,
        weight=qkv_weight,
        bias=qkv_bias,
        activation=q_conv.activation,
        backend=q_conv.backend,
    )
    return torch.split(
        qkv,
        [q_conv.weight.shape[0], k_conv.weight.shape[0], v_conv.weight.shape[0]],
        dim=-1,
    )


__all__ = [
    'CausalConv1dFunction',
    'CausalConv1dFunctionCP',
    'FastCausalConv1dFn',
    'ImplicitLongConvolution',
    'LongConvolution',
    'PositionalEmbedding',
    'ShortConvolution',
    'can_fuse_qkv_short_conv',
    'causal_conv1d',
    'causal_conv1d_bwd',
    'causal_conv1d_cp',
    'causal_conv1d_fwd',
    'causal_conv1d_update',
    'causal_conv1d_update_states',
    'fast_causal_conv1d_fn',
    'fft_conv',
    'fused_qkv_short_conv',
]
