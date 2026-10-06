# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from fla.modules.layernorm import RMSNorm
from fla.modules.residuals.base import BaseResidual


class MHCHistory(NamedTuple):
    """Streams, pending routing coefficients, and the stream readout before branch normalization."""

    streams: torch.Tensor
    pre: torch.Tensor | None
    post: torch.Tensor | None
    mixing: torch.Tensor | None
    hidden_state: torch.Tensor


def sinkhorn(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Apply column/row Sinkhorn normalization in log space without exponent overflow."""
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    values = logits if logits.dtype == torch.float64 else logits.float()
    for _ in range(num_iters):
        values = values - values.logsumexp(dim=-2, keepdim=True)
        values = values - values.logsumexp(dim=-1, keepdim=True)
    return values.exp()


class _MHCRouting(nn.Module):
    """Dynamic pre/post/res mappings from Eqs. 7-9 of https://arxiv.org/abs/2512.24880."""

    def __init__(self, hidden_size: int, num_streams: int, num_iters: int, eps: float, init_scale: float):
        super().__init__()
        self.num_streams, self.num_iters, self.eps, self.init_scale = num_streams, num_iters, eps, init_scale
        self.weight = nn.Parameter(torch.empty(num_streams * (num_streams + 2), num_streams * hidden_size))
        self.scale = nn.Parameter(torch.empty(3))
        self.pre_bias = nn.Parameter(torch.empty(num_streams))
        self.post_bias = nn.Parameter(torch.empty(num_streams))
        self.res_bias = nn.Parameter(torch.empty(num_streams, num_streams))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.weight.shape[-1] ** -0.5)
        nn.init.constant_(self.scale, self.init_scale)
        # start near a mean readout and unit write-back on replicated embedding streams.
        nn.init.constant_(self.pre_bias, -math.log(self.num_streams - 1))
        nn.init.zeros_(self.post_bias)
        nn.init.zeros_(self.res_bias)

    def forward(self, streams: torch.Tensor) -> MHCHistory:
        dtype = torch.float64 if streams.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=streams.device.type, enabled=False):
            flat = streams.to(dtype).flatten(-2)
            flat = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.eps)
            logits = F.linear(flat, self.weight.to(dtype))
            pre, post, mixing = logits.split((self.num_streams, self.num_streams, self.num_streams ** 2), dim=-1)
            pre = (pre * self.scale[0].to(dtype) + self.pre_bias.to(dtype)).sigmoid()
            post = 2 * (post * self.scale[1].to(dtype) + self.post_bias.to(dtype)).sigmoid()
            mixing = mixing.unflatten(-1, (self.num_streams, self.num_streams))
            mixing = sinkhorn(mixing * self.scale[2].to(dtype) + self.res_bias.to(dtype), num_iters=self.num_iters)
        hidden_state = (pre.unsqueeze(-1) * streams.to(dtype)).sum(dim=-2).to(streams.dtype)
        return MHCHistory(streams=streams, pre=pre, post=post, mixing=mixing, hidden_state=hidden_state)


class ManifoldHyperConnection(BaseResidual[MHCHistory]):
    """Update mHC streams, then prepare the next sublayer's input and routing history.

    Streams have shape `[..., num_streams, hidden_size]`; branch inputs retain `[..., hidden_size]`.
    Routing and stream accumulation use FP32 (FP64 for double inputs). The first input replicates
    embeddings, and the final output uses a mean over streams followed by RMSNorm. This is a
    PyTorch implementation of the mHC recurrence, without the paper's fused infrastructure kernels.
    """

    def __init__(
        self,
        hidden_size: int,
        sub_layer_idx: int = 0,
        num_streams: int = 4,
        num_iters: int = 20,
        init_scale: float = 0.01,
        num_sublayers: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(sub_layer_idx, bool) or not isinstance(sub_layer_idx, int) or sub_layer_idx < 0:
            raise ValueError('sub_layer_idx must be a non-negative integer')
        self.hidden_size = hidden_size
        self.sub_layer_idx = sub_layer_idx
        norm_eps = kwargs.pop('norm_eps', 1e-6)
        norm_cls = RMSNorm if kwargs.pop('fuse_norm', True) else nn.RMSNorm
        if kwargs:
            raise TypeError(f'Unexpected ManifoldHyperConnection arguments: {sorted(kwargs)}')
        self.norm = norm_cls(hidden_size, eps=norm_eps)
        self.input_norm = norm_cls(hidden_size, eps=norm_eps) if sub_layer_idx == 0 else None
        if isinstance(num_streams, bool) or not isinstance(num_streams, int) or num_streams < 2:
            raise ValueError('num_streams must be an integer >= 2')
        if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
            raise ValueError('num_iters must be a positive integer')
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError('init_scale must be finite and positive')
        if num_sublayers is not None and (not isinstance(num_sublayers, int) or num_sublayers <= sub_layer_idx):
            raise ValueError('num_sublayers must exceed sub_layer_idx')
        self.num_streams = num_streams
        self.is_last = num_sublayers is not None and sub_layer_idx == num_sublayers - 1
        routing_kwargs = dict(
            hidden_size=hidden_size, num_streams=num_streams, num_iters=num_iters,
            eps=self.norm.eps, init_scale=init_scale,
        )
        self.routing = None if self.is_last else _MHCRouting(**routing_kwargs)
        self.input_routing = _MHCRouting(**routing_kwargs) if sub_layer_idx == 0 else None

    def initialize(self, x: torch.Tensor) -> tuple[torch.Tensor, MHCHistory]:
        if self.sub_layer_idx != 0:
            raise ValueError('Only the first sublayer can initialize residual history')
        streams = x.unsqueeze(-2).expand(*x.shape[:-1], self.num_streams, self.hidden_size)
        history = self.input_routing(streams)
        return self.input_norm(history.hidden_state), history

    def forward(self, branch_output: torch.Tensor, history: MHCHistory) -> tuple[torch.Tensor, MHCHistory]:
        with torch.autocast(device_type=branch_output.device.type, enabled=False):
            dtype = history.mixing.dtype
            streams = history.mixing @ history.streams.to(dtype)
            streams = streams + history.post.unsqueeze(-1) * branch_output.to(dtype).unsqueeze(-2)
            streams = streams.to(history.streams.dtype)
        if self.is_last:
            hidden_state = streams.to(dtype).mean(dim=-2).to(streams.dtype)
            history = MHCHistory(streams=streams, pre=None, post=None, mixing=None, hidden_state=hidden_state)
        else:
            history = self.routing(streams)
        return self.norm(history.hidden_state), history

    @staticmethod
    def get_hidden_state(history: MHCHistory) -> torch.Tensor:
        return history.hidden_state
