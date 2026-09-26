# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F

from fla.ops.cp import FLACPContext, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd
from fla.ops.utils import prepare_sequence_ids


class ConvBoundaryExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tails, group):
        ctx.group = group
        return conv_cp_send_recv_fwd(tails, group)

    @staticmethod
    def backward(ctx, grad):
        return conv_cp_send_recv_bwd(grad.contiguous(), ctx.group), None


def causal_conv1d_cuda_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_context: FLACPContext,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    from fla.modules.conv.cuda.ops import causal_conv1d_fn_cuda

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

    prefix = None
    if cp_context.group is not None:
        heads = ConvBoundaryExchange.apply(x[0, -(width - 1):].contiguous(), cp_context.group)
        valid = min(width - 1, cp_context.pre_num_conv_tokens or 0)
        # history must not cross the first local document's global start
        mask = torch.arange(width - 1, device=x.device) >= width - 1 - valid
        prefix = heads.masked_fill(~mask[:, None], 0).unsqueeze(0)

    channels = x.shape[-1]
    padding = -channels % 8
    x = F.pad(x, (0, padding)) if padding else x.contiguous()
    if padding:
        weight = F.pad(weight, (0, 0, 0, padding))
        bias = F.pad(bias, (0, padding)) if bias is not None else None
        prefix = F.pad(prefix, (0, padding)) if prefix is not None else None
    seq_idx = prepare_sequence_ids(
        cp_context.cu_seqlens, cu_seqlens_cpu=cp_context.cu_seqlens_cpu,
    ).to(torch.int32).unsqueeze(0)
    y = causal_conv1d_fn_cuda(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=prefix.transpose(1, 2) if prefix is not None else None,
        activation=activation,
    ).transpose(1, 2)[..., :channels]
    return y + residual if residual is not None else y
