# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

from fla.modules.norm import RMSNorm
from fla.modules.residuals.base import BaseResidual

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class StandardResidualState(NamedTuple):
    residual: torch.Tensor
    branch_output: torch.Tensor | None

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.residual if self.branch_output is None else self.residual + self.branch_output


class StandardResidual(BaseResidual[StandardResidualState, torch.Tensor]):
    r"""Standard residual accumulation, optionally fusing addition with the next normalization.

    All depth mixing weights are M_{i->l} = 1, so state only needs the running sum h_l.
    With v_l = f_l(h_l) as the branch output, the update is

    .. math::
        h_{l+1} = h_l + v_l.

    State stores two summands `(residual, branch_output)` of h_l; `None` means the sum is already materialized.
    `pre` normalizes their sum while returning h_l as the context for `post`.
    `post` stores `(h_l + v_l, None)` when fusion is disabled, or `(h_l, v_l)` when enabled.
    The latter lets the next `pre` fuse addition and normalization without changing the residual rule.

    Unfused addition follows PyTorch dtype promotion.
    Fused addition retains the residual dtype and returns the normalized input in the branch dtype.
    The final output always adds before normalizing.

    Args:
        fuse_residual (bool, Optional):
            Keep the branch output for addition during the next `pre` call when `fuse_norm=True`. Default: `False`.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        fuse_residual: bool = False,
    ):
        super().__init__(hidden_size=hidden_size)
        self.fuse_norm = fuse_norm
        self.norm = (RMSNorm if fuse_norm else nn.RMSNorm)(hidden_size, eps=norm_eps)
        # delaying addition only benefits the fused add-norm path.
        self.fuse_residual = fuse_residual and fuse_norm

    @classmethod
    def from_config(cls, config: 'PretrainedConfig', **kwargs) -> 'StandardResidual':
        kwargs.setdefault('fuse_residual', True)
        return cls(hidden_size=config.hidden_size, norm_eps=config.norm_eps, fuse_norm=config.fuse_norm, **kwargs)

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()

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

    def forward(self, state: StandardResidualState) -> torch.Tensor:
        return self.norm(state.hidden_states)
