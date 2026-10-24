# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from fla.modules.norm import RMSNorm


class BaseResidual(nn.Module, ABC):
    """Own one sublayer's normalization and residual parameters.

    `pre` reads the incoming state and returns a normalized branch input plus per-call context.
    `post` writes the branch output into a new state. State and context remain explicit inputs,
    retain autograd connections, and are never cached on the module or updated in place.
    An output module performs only the final readout and normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        sub_layer_idx: int = 0,
        is_output: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sub_layer_idx = sub_layer_idx
        self.is_output = is_output
        self.norm = (RMSNorm if fuse_norm else nn.RMSNorm)(hidden_size, eps=norm_eps)

    def reset_parameters(self) -> None:
        """Initialize owned parameters; custom implementations may override this method."""
        self.norm.reset_parameters()

    def initialize(self, x: torch.Tensor) -> Any:
        return x

    @abstractmethod
    def pre(self, state: Any) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError

    @abstractmethod
    def post(self, state: Any, branch_output: torch.Tensor, context: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_hidden_state(state: Any) -> torch.Tensor:
        return state
