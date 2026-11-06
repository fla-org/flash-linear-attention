# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import NamedTuple

import torch

from fla.modules.residuals.base import BaseResidual


class StandardResidualState(NamedTuple):
    residual: torch.Tensor
    branch_output: torch.Tensor | None


class StandardResidual(BaseResidual[StandardResidualState, torch.Tensor]):
    """Add each branch output to the residual, optionally fusing the addition with the next normalization.

    Unfused addition follows PyTorch dtype promotion.
    Fused addition retains the residual dtype and returns the normalized input in the branch dtype.
    The final readout always adds before normalizing.

    Args:
        fuse_residual (bool, Optional):
            Keep the branch output for addition during the next `pre` call. Default: `False`.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        fuse_residual: bool = False,
    ):
        super().__init__(hidden_size=hidden_size, norm_eps=norm_eps, fuse_norm=fuse_norm)
        self.fuse_residual = fuse_residual

    def initialize(self, x: torch.Tensor) -> StandardResidualState:
        return StandardResidualState(x, None)

    def pre(self, state: StandardResidualState) -> tuple[torch.Tensor, torch.Tensor]:
        residual, branch_output = state
        if branch_output is None:
            return self.norm(residual), residual
        if self.fuse_norm:
            return self.norm(branch_output, residual, prenorm=True)
        residual = residual + branch_output
        return self.norm(residual), residual

    def post(self, branch_output: torch.Tensor, context: torch.Tensor) -> StandardResidualState:
        if self.fuse_residual:
            return StandardResidualState(context, branch_output)
        return StandardResidualState(context + branch_output, None)

    def get_hidden_state(self, state: StandardResidualState) -> torch.Tensor:
        if state.branch_output is None:
            return state.residual
        return state.residual + state.branch_output
