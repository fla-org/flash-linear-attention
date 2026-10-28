# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.modules.residuals.base import BaseResidual


class StandardResidual(BaseResidual[torch.Tensor, None]):
    """Keep the accumulated residual as a Tensor and add each branch output immediately.

    `pre` normalizes the current state and returns no context; `post` adds the branch output to that state.
    Addition follows PyTorch dtype promotion rules, with no pending branch stored between sublayers.
    The output module normalizes the final accumulated Tensor.
    """

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def pre(self, state: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.norm(state), None

    def post(self, state: torch.Tensor, branch_output: torch.Tensor, context: None = None) -> torch.Tensor:
        return state + branch_output

    @staticmethod
    def get_hidden_state(state: torch.Tensor) -> torch.Tensor:
        return state
