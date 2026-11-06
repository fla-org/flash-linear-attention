# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import NamedTuple

import torch
from torch import nn

from fla.modules.residuals.base import BaseResidual
from fla.ops.attnres import fused_attnres


class AttentionResidualState(NamedTuple):
    residuals: list[torch.Tensor]
    num_branches: int


class AttentionResidual(BaseResidual[AttentionResidualState, AttentionResidualState]):
    """Read completed blocks and the partial sum using this sublayer's learned query.

    State contains the residual sources and the number of completed branches, which determines block boundaries.
    Each `post` returns new state without mutating the incoming history.
    Gradient checkpointing requires `use_reentrant=False`.
    """

    supports_reentrant_checkpointing = False

    def __init__(self, hidden_size: int, block_size: int = 1, norm_eps: float = 1e-6, fuse_norm: bool = True):
        super().__init__(hidden_size=hidden_size, norm_eps=norm_eps, fuse_norm=fuse_norm)
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        self.block_size = block_size
        self.query = nn.Linear(hidden_size, 1, bias=False)
        self.query._is_attnres_proj = True
        self.key_norm = nn.RMSNorm(hidden_size, eps=self.norm.eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.zeros_(self.query.weight)
        self.key_norm.reset_parameters()

    def initialize(self, x: torch.Tensor) -> AttentionResidualState:
        return AttentionResidualState([x], 0)

    def pre(self, state: AttentionResidualState) -> tuple[torch.Tensor, AttentionResidualState]:
        if state.num_branches == 0:
            return self.norm(state.residuals[0]), state
        return self(state), state

    def post(self, branch_output: torch.Tensor, context: AttentionResidualState) -> AttentionResidualState:
        residuals, num_branches = context
        # all sources passed to fused_attnres must share the first branch's dtype.
        if num_branches == 0:
            residuals = [residuals[0].to(branch_output.dtype)]
        if num_branches % self.block_size == 0:
            residuals = [*residuals, branch_output]
        else:
            residuals = [*residuals[:-1], residuals[-1] + branch_output]
        return AttentionResidualState(residuals, num_branches + 1)

    def get_hidden_state(self, state: AttentionResidualState) -> torch.Tensor:
        return state.residuals[-1]

    def forward(self, state: AttentionResidualState) -> torch.Tensor:
        return fused_attnres(
            query=self.query.weight,
            residuals=state.residuals,
            rms_weight=self.key_norm.weight,
            output_rms_weight=self.norm.weight,
            rms_eps=self.norm.eps,
        )
