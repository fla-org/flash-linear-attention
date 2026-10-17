# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fla.ops.cp import conv_cp_send_recv_bwd, conv_cp_send_recv_fwd
from fla.ops.utils import prepare_sequence_ids

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from fla.ops.cp import FLACPContext


class _ConvBoundaryExchangeFunction(torch.autograd.Function):
    """Exchange convolution history and return its gradients to the sending rank."""

    @staticmethod
    def forward(ctx, tails: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return conv_cp_send_recv_fwd(tails=tails, group=group)

    @staticmethod
    def backward(ctx, dheads: torch.Tensor) -> tuple[torch.Tensor, None]:
        return conv_cp_send_recv_bwd(d_initial_state=dheads.contiguous(), group=ctx.group), None


def causal_conv1d_cuda_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    activation: str | None,
    cp_context: FLACPContext,
) -> torch.Tensor:
    """Apply CUDA convolution to contiguous CP inputs of shape `[1, T, D]`.

    Each rank needs at least `W - 1` tokens, with `W` in {2, 3, 4}.
    Packed sequence boundaries mask both history values and their gradients.
    """
    from fla.modules.causal_conv1d.backends.cuda.ops import causal_conv1d_fn_cuda

    if causal_conv1d_fn_cuda is None:
        raise ImportError("CUDA CP requires the causal-conv1d package")
    if cp_context.layout != 'contiguous':
        raise NotImplementedError("CUDA convolution supports only contiguous CP")
    width = weight.shape[-1]
    if width not in (2, 3, 4):
        raise NotImplementedError("CUDA CP convolution supports kernel widths 2, 3, and 4")
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("CUDA CP convolution requires input shape [1, T, D]")
    if x.shape[1] < width - 1:
        raise ValueError("CUDA CP convolution requires at least kernel_width - 1 tokens per rank")
    if cp_context.conv1d_kernel_size != width:
        raise ValueError("CP context kernel width must match the convolution weight")
    if activation not in (None, 'silu', 'swish'):
        raise NotImplementedError("activation must be None, silu, or swish")

    initial_state = None
    if cp_context.group is not None:
        tails = x[0, -(width - 1):].contiguous()
        heads = _ConvBoundaryExchangeFunction.apply(tails, cp_context.group)
        valid_len = min(width - 1, cp_context.pre_num_conv_tokens or 0)
        # history must not cross the first local document's global start
        mask = torch.arange(width - 1, device=x.device) >= width - 1 - valid_len
        initial_state = heads.masked_fill(~mask[:, None], 0).unsqueeze(0)

    channels = x.shape[-1]
    # the CUDA channel-last kernels require strides divisible by eight
    padding = -channels % 8
    x = F.pad(x, (0, padding)) if padding else x.contiguous()
    if padding:
        weight = F.pad(weight, (0, 0, 0, padding))
        bias = F.pad(bias, (0, padding)) if bias is not None else None
        initial_state = F.pad(initial_state, (0, padding)) if initial_state is not None else None

    seq_idx = prepare_sequence_ids(cu_seqlens=cp_context.cu_seqlens, cu_seqlens_cpu=cp_context.cu_seqlens_cpu)
    seq_idx = seq_idx.to(torch.int32).unsqueeze(0)

    y = causal_conv1d_fn_cuda(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_state.transpose(1, 2) if initial_state is not None else None,
        activation=activation,
    )
    y = y.transpose(1, 2)[..., :channels]
    return y + residual if residual is not None else y
