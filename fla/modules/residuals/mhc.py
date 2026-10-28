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

from fla.modules.residuals.base import BaseResidual
from fla.ops.hc import fused_hc_post, fused_hc_pre
from fla.ops.utils import sinkhorn


class MHCState(NamedTuple):
    """Residual streams of shape `[..., num_streams, hidden_size]` and prepared routing coefficients.

    `pre`, `post`, and `mixing` belong to the next sublayer consuming this state.
    Intermediate hidden-state readouts use its learned stream aggregation without recomputing routing coefficients.
    All coefficients are `None` after the final sublayer's write.
    """

    streams: torch.Tensor
    pre: torch.Tensor | None
    post: torch.Tensor | None
    mixing: torch.Tensor | None


class ManifoldHyperConnection(BaseResidual[MHCState, None]):
    """Learn mHC routing and compose the HC and Sinkhorn operators.

    Routing parameters are shifted one sublayer earlier; normalization parameters stay with their consuming sublayer.
    Each sublayer except the last owns the routing parameters for the next sublayer.
    `post` writes with the incoming state's coefficients, then prepares the next coefficients from the updated streams.
    `pre` and `get_hidden_state` reuse these prepared coefficients; per-call context is `None`.
    Reentrant checkpointing flattens both the streams and prepared coefficients to preserve their autograd connections.

    The output module owns the first sublayer's routing parameters and uses them in `initialize`.
    The last sublayer owns no routing parameters and clears the state's coefficients after its write.
    The output module then averages the final streams and applies its final normalization.

    Routing uses FP32, and HC operators accumulate in FP32 before returning the stream dtype.
    HC and Sinkhorn require NVIDIA CUDA tensors in FP32, FP16, or BF16.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        num_iters: int = 20,
        init_scale: float = 0.01,
        **kwargs,
    ):
        super().__init__(hidden_size=hidden_size, **kwargs)
        if isinstance(num_streams, bool) or not isinstance(num_streams, int) or not 2 <= num_streams <= 8:
            raise ValueError('num_streams must be an integer between 2 and 8')
        if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
            raise ValueError('num_iters must be a positive integer')
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError('init_scale must be finite and positive')
        self.num_streams = num_streams
        self.num_iters = num_iters
        self.init_scale = init_scale
        if not self.is_last_sublayer:
            self.weight = nn.Parameter(torch.empty(num_streams * (num_streams + 2), num_streams * hidden_size))
            self.scale = nn.Parameter(torch.empty(3))
            self.pre_bias = nn.Parameter(torch.empty(num_streams))
            self.post_bias = nn.Parameter(torch.empty(num_streams))
            self.res_bias = nn.Parameter(torch.empty(num_streams, num_streams))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if not self.is_last_sublayer:
            nn.init.normal_(self.weight, std=self.weight.shape[-1] ** -0.5)
            nn.init.constant_(self.scale, self.init_scale)
            nn.init.constant_(self.pre_bias, -math.log(self.num_streams - 1))
            nn.init.zeros_(self.post_bias)
            nn.init.zeros_(self.res_bias)

    def initialize(self, x: torch.Tensor) -> MHCState:
        streams = x.unsqueeze(-2).expand(*x.shape[:-1], self.num_streams, self.hidden_size)
        pre, post, mixing = self.prepare_coefficients(streams)
        return MHCState(streams, pre, post, mixing)

    def prepare_coefficients(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=streams.device.type, enabled=False):
            flat = F.rms_norm(streams.float().flatten(-2), (self.num_streams * self.hidden_size,), eps=self.norm.eps)
            logits = F.linear(flat, self.weight.float())
            pre, post, mixing = logits.split((self.num_streams, self.num_streams, self.num_streams ** 2), dim=-1)
            pre = (pre * self.scale[0].float() + self.pre_bias.float()).sigmoid()
            post = 2 * (post * self.scale[1].float() + self.post_bias.float()).sigmoid()
            mixing = mixing.unflatten(-1, (self.num_streams, self.num_streams))
            mixing = sinkhorn(logits=mixing * self.scale[2].float() + self.res_bias.float(), num_iters=self.num_iters)

        return pre, post, mixing

    def pre(self, state: MHCState) -> tuple[torch.Tensor, None]:
        if self.is_output:
            return self.norm(self.get_hidden_state(state)), None

        hidden_states = fused_hc_pre(x=state.streams, pre=state.pre)
        return self.norm(hidden_states), None

    def post(
        self,
        state: MHCState,
        branch_output: torch.Tensor,
        context: None,
    ) -> MHCState:
        streams = fused_hc_post(x=state.streams, branch=branch_output, post=state.post, mixing=state.mixing)
        if self.is_last_sublayer:
            return MHCState(streams, None, None, None)

        pre, post, mixing = self.prepare_coefficients(streams)
        return MHCState(streams, pre, post, mixing)

    @staticmethod
    def get_hidden_state(state: MHCState) -> torch.Tensor:
        if state.pre is None:
            return state.streams.float().mean(dim=-2).to(state.streams.dtype)

        return fused_hc_pre(x=state.streams, pre=state.pre)
