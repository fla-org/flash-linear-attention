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

StateT = TypeVar('StateT')


class BaseResidual(nn.Module, ABC, Generic[StateT]):
    """Prepare branch inputs and carry implementation-specific residual state between sublayers.

    Implementations own their parameters and state layout. Initialization and updates return a branch input
    and history; `get_hidden_state` exposes a representation for intermediate model hidden states.
    Histories and their reported representations retain autograd connections without in-place updates.

    Parameter initialization belongs to `reset_parameters`, independently of history initialization.
    Model integrations call it instead of initializing residual children with their generic weight policy.
    Custom initial values must be reproducible by this method, including after construction on the meta device.
    """

    def reset_parameters(self) -> None:
        """Initialize child modules using their own reset methods.

        Override to initialize parameters or buffers owned directly by the residual, or to customize child defaults.
        Call `super().reset_parameters()` first when retaining the other child defaults, and call this method after
        constructing all fields for standalone use. Use `torch.nn.init` functions on the original tensors so
        checkpoint loading can protect already loaded weights; avoid `.data`, tensor views and in-place arithmetic.
        """
        def reset(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                for child in module.children():
                    reset(child)

        for child in self.children():
            reset(child)

    @abstractmethod
    def initialize(self, x: torch.Tensor) -> tuple[torch.Tensor, StateT]:
        """Create the first sublayer's history and prepare its branch input from model embeddings."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, branch_output: torch.Tensor, history: StateT) -> tuple[torch.Tensor, StateT]:
        """Return the next branch input (or final model output) and a new history without mutating the input history."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_hidden_state(history: StateT) -> torch.Tensor:
        """Expose a `[..., hidden_size]` representation without copying, detaching, or mutating history."""
        raise NotImplementedError
