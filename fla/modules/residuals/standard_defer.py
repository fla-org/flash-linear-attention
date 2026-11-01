# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import NamedTuple

import torch

from fla.modules.residuals.base import BaseResidual


class ResidualState(NamedTuple):
    """Accumulated residual and pending branch output; `None` marks the initial state."""

    x: torch.Tensor
    branch_output: torch.Tensor | None


class StandardDeferredResidual(BaseResidual[ResidualState, torch.Tensor]):
    """Defer residual addition to fuse it with the next sublayer's normalization.

    The fused path returns the branch dtype and retains the accumulated residual dtype in context.
    The output module adds the final branch before normalization, matching the model's final readout.
    """

    def initialize(self, x: torch.Tensor) -> ResidualState:
        return ResidualState(x, None)

    def pre(self, state: ResidualState) -> tuple[torch.Tensor, torch.Tensor]:
        x, branch_output = state

        if self.is_output:
            x = self.get_hidden_state(state)
            return self.norm(x), x

        if self.sub_layer_idx == 0:
            return self.norm(x), x

        assert branch_output is not None
        if self.fuse_norm:
            normed_x, x = self.norm(branch_output, x, prenorm=True)
        else:
            x = x + branch_output
            normed_x = self.norm(x)
        return normed_x, x

    def post(self, state: ResidualState, branch_output: torch.Tensor, context: torch.Tensor) -> ResidualState:
        return ResidualState(context, branch_output)

    @staticmethod
    def get_hidden_state(state: ResidualState) -> torch.Tensor:
        if state.branch_output is None:
            return state.x
        return state.x + state.branch_output
