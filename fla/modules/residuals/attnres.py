# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from torch import nn

from fla.modules.residuals.base import BaseResidual
from fla.ops.attnres import fused_attnres


class AttentionResidual(BaseResidual):
    """Read completed blocks and the partial sum using this sublayer's learned query."""

    def __init__(self, hidden_size: int, block_size: int = 1, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
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

    def initialize(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x]

    def pre(self, state: list[torch.Tensor]) -> tuple[torch.Tensor, None]:
        if self.sub_layer_idx == 0 and not self.is_output:
            return self.norm(state[0]), None
        output = fused_attnres(
            query=self.query.weight,
            residuals=state,
            rms_weight=self.key_norm.weight,
            output_rms_weight=self.norm.weight,
            rms_eps=self.norm.eps,
        )
        return output, None

    def post(self, state: list[torch.Tensor], branch_output: torch.Tensor, context=None) -> list[torch.Tensor]:
        # all sources passed to fused_attnres must share the first branch's dtype.
        if self.sub_layer_idx == 0:
            state = [state[0].to(branch_output.dtype)]
        if self.sub_layer_idx % self.block_size == 0:
            return [*state, branch_output]
        return [*state[:-1], state[-1] + branch_output]

    @staticmethod
    def get_hidden_state(state: list[torch.Tensor]) -> torch.Tensor:
        return state[-1]
