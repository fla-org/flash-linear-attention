# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from torch import nn

from fla.modules.norm import RMSNorm

StateT = TypeVar('StateT')
ContextT = TypeVar('ContextT')


class BaseResidual(nn.Module, ABC, Generic[StateT, ContextT]):
    """Own normalization, learned residual parameters, and residual state transitions.

    `pre` reads the incoming state and returns a normalized branch input plus per-call context.
    `post` writes the branch output into a new state. State and context remain explicit inputs,
    retain autograd connections, and are never cached on the module or updated in place.
    `StateT` ties the initialized state to the inputs and outputs of subsequent state transitions and reads.
    `ContextT` ties each `pre` context to the corresponding `post` input; use `None` when no context is needed.
    The output module initializes the model state and performs the final readout and normalization.
    Callers identify this separate output module with `is_output=True`.
    Reentrant checkpointing requires flattening state inputs and outputs at the checkpoint boundary.
    Subclasses can disable it with `supports_reentrant_checkpointing = False`.
    """

    supports_reentrant_checkpointing = True

    def __init__(
        self,
        hidden_size: int,
        sub_layer_idx: int,
        num_sublayers: int,
        is_output: bool = False,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sub_layer_idx = sub_layer_idx
        self.num_sublayers = num_sublayers
        self.is_last_sublayer = sub_layer_idx == num_sublayers - 1
        self.is_output = is_output

        self.fuse_norm = fuse_norm
        self.norm = (RMSNorm if fuse_norm else nn.RMSNorm)(hidden_size, eps=norm_eps)

    def reset_parameters(self) -> None:
        """Initialize owned parameters; custom implementations may override this method."""
        self.norm.reset_parameters()

    @abstractmethod
    def initialize(self, x: torch.Tensor) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def pre(self, state: StateT) -> tuple[torch.Tensor, ContextT]:
        raise NotImplementedError

    @abstractmethod
    def post(self, state: StateT, branch_output: torch.Tensor, context: ContextT) -> StateT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_hidden_state(state: StateT) -> torch.Tensor:
        raise NotImplementedError
