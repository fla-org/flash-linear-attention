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
    r"""Own residual parameters, residual state transitions, and branch/output normalization.

    Following Attention Residuals, each attention or MLP is a layer l, h_l is its input before normalization,
    and f_l(h_l) includes the branch's PreNorm. The caller owns the attention/MLP computation and layer order.

    `initialize(h_1)` creates residual state from embeddings.
    `readout(state)` returns h_l; `pre(state)` returns Norm_l(h_l) and the context needed for writeback.
    The caller computes the branch output f_l(h_l), which `post(branch_output, context)` writes into new state.
    Calling the output residual module performs its final readout and output normalization.
    State and context retain autograd connections and must not be mutated or cached on the module.
    """

    supports_reentrant_checkpointing = True

    def __init__(
        self,
        hidden_size: int,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
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
    def post(self, branch_output: torch.Tensor, context: ContextT) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def readout(self, state: StateT) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, state: StateT) -> torch.Tensor:
        """Normalize the final readout; implementations may fuse aggregation and normalization."""
        return self.norm(self.readout(state))
