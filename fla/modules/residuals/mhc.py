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
    """Residual streams and routing coefficients for the next branch; final state has no coefficients."""

    streams: torch.Tensor
    pre: torch.Tensor | None
    post: torch.Tensor | None
    mixing: torch.Tensor | None


class ManifoldHyperConnection(BaseResidual[MHCState, MHCState]):
    r"""Manifold-constrained mixing of residual streams with learned read and write weights.

    State retains N streams X of shape `[..., N, hidden_size]` and prepared routing coefficients.
    A normalized projection of X produces sigmoid read weights p, scaled sigmoid write weights q,
    and a mixing matrix A normalized by alternating column and row Sinkhorn steps:

    .. math::
        h_l = \sum_n p_n X_n,
        \qquad X'_i = \sum_j A_{ij} X_j + q_i v_l.

    `readout` applies the prepared read weights; `pre` additionally normalizes the branch input.
    `post` updates the streams and prepares routing for the next branch using this module's parameters.
    The output module owns the initial routing and initializes state from repeated embeddings.
    The last branch, indexed by `num_sublayers - 1`, writes its output without creating another set of routing parameters.
    Its resulting state is read by averaging streams, followed by output normalization when calling the output module.

    Routing and stream reductions use FP32; HC returns the stream dtype.
    Supported inputs are NVIDIA CUDA FP32, FP16, and BF16 tensors.

    Args:
        num_streams (int, Optional):
            Number of residual streams, between 2 and 8. Default: 4.
        num_iters (int, Optional):
            Number of Sinkhorn normalization iterations. Default: 20.
        init_scale (float, Optional):
            Initial scale for dynamic routing logits. Default: 0.01.
        sub_layer_idx (int):
            Zero-based branch index; the output residual uses `num_sublayers`.
        num_sublayers (int):
            Total number of branches, counting attention and MLP separately.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        num_iters: int = 20,
        init_scale: float = 0.01,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        *,
        sub_layer_idx: int,
        num_sublayers: int,
    ):
        super().__init__(
            hidden_size=hidden_size,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            sub_layer_idx=sub_layer_idx,
            num_sublayers=num_sublayers,
        )
        if isinstance(num_streams, bool) or not isinstance(num_streams, int) or not 2 <= num_streams <= 8:
            raise ValueError('num_streams must be an integer between 2 and 8')
        if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
            raise ValueError('num_iters must be a positive integer')
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError('init_scale must be finite and positive')
        self.num_streams = num_streams
        self.num_iters = num_iters
        self.init_scale = init_scale
        self.prepare_next = self.sub_layer_idx != self.num_sublayers - 1
        if self.prepare_next:
            self.weight = nn.Parameter(torch.empty(num_streams * (num_streams + 2), num_streams * hidden_size))
            self.scale = nn.Parameter(torch.empty(3))
            self.pre_bias = nn.Parameter(torch.empty(num_streams))
            self.post_bias = nn.Parameter(torch.empty(num_streams))
            self.res_bias = nn.Parameter(torch.empty(num_streams, num_streams))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.prepare_next:
            nn.init.normal_(self.weight, std=self.weight.shape[-1] ** -0.5)
            nn.init.constant_(self.scale, self.init_scale)
            nn.init.constant_(self.pre_bias, -math.log(self.num_streams - 1))
            nn.init.zeros_(self.post_bias)
            nn.init.zeros_(self.res_bias)

    def initialize(self, x: torch.Tensor) -> MHCState:
        if not self.prepare_next:
            raise ValueError('The last branch has no initial routing; initialize state with the output residual')
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

    def pre(self, state: MHCState) -> tuple[torch.Tensor, MHCState]:
        return self.norm(self.readout(state)), state

    def post(self, branch_output: torch.Tensor, context: MHCState) -> MHCState:
        streams = fused_hc_post(x=context.streams, branch=branch_output, post=context.post, mixing=context.mixing)
        if not self.prepare_next:
            return MHCState(streams, None, None, None)
        pre, post, mixing = self.prepare_coefficients(streams)
        return MHCState(streams, pre, post, mixing)

    def readout(self, state: MHCState) -> torch.Tensor:
        if state.pre is None:
            return state.streams.float().mean(dim=-2).to(state.streams.dtype)
        return fused_hc_pre(x=state.streams, pre=state.pre)
