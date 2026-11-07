# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

import torch
from torch import nn
from typing_extensions import Self

from fla.modules.norm import RMSNorm


class ResidualState(Protocol):
    """Current model-visible tensor and the residual history retained between layers."""

    @property
    def hidden_states(self) -> torch.Tensor:
        """Current sum exposed by the model, before residual readout and normalization."""
        ...

    def with_hidden_states(self, hidden_states: torch.Tensor) -> Self:
        """Replace the current tensor while retaining completed residual history."""
        ...


StateT = TypeVar('StateT', bound=ResidualState)
ContextT = TypeVar('ContextT')


class BaseResidual(nn.Module, ABC, Generic[StateT, ContextT]):
    r"""Residual aggregation, normalization, and state updates.

    Let v_0 = h_1 be the embedding and v_i = f_i(h_i) be a branch output.
    The depth mixing weights M_{i->l} determine the input h_l to branch l:

    .. math::
        h_l = \sum_{i=0}^{l-1} M_{i\to l} v_i,
        \qquad h_{l+1} = \sum_{i=0}^{l} M_{i\to l+1} v_i.

    Each implementation defines these weights and the state needed to evaluate the sums.
    State may retain individual outputs, block sums, or a recurrent summary.
    The module owns three operations:

    * `readout(state)` computes h_l without branch normalization.
    * `pre(state)` returns the normalized h_l and the context needed to update state.
    * `post(branch_output, context)` incorporates v_l into the state used by the next branch.

    The model executes its branch between `pre` and `post` to produce v_l = f_l(h_l).
    Here f_l includes the normalization performed by `pre`; the model owns the remaining branch computation.
    Context carries the old state and any update coefficients needed after that computation.
    The next residual module reads the updated state using its own mixing weights.

    `initialize(x)` creates the initial state from embeddings.
    Calling the output residual module applies output normalization to its readout.
    State and context retain autograd connections and must not be mutated or cached on the module.
    See Section 6.2 of `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.
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
