# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from torch import nn

from fla.modules.layernorm import RMSNorm
from fla.modules.residuals.base import BaseResidual


class StandardResidual(BaseResidual[torch.Tensor]):
    """Add the branch output to tensor history and normalize the next sublayer input.

    The first residual owns the input norm. Each residual's output norm belongs to the next
    sublayer, or to the model output at the end.
    """

    def __init__(
        self,
        hidden_size: int,
        sub_layer_idx: int = 0,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop('num_sublayers', None)
        if kwargs:
            raise TypeError(f'Unexpected StandardResidual arguments: {sorted(kwargs)}')
        if isinstance(sub_layer_idx, bool) or not isinstance(sub_layer_idx, int) or sub_layer_idx < 0:
            raise ValueError('sub_layer_idx must be a non-negative integer')
        self.hidden_size = hidden_size
        self.sub_layer_idx = sub_layer_idx
        self.fuse_norm = fuse_norm
        norm_cls = RMSNorm if fuse_norm else nn.RMSNorm
        self.norm = norm_cls(hidden_size, eps=norm_eps)
        self.input_norm = norm_cls(hidden_size, eps=norm_eps) if sub_layer_idx == 0 else None

    def initialize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sub_layer_idx != 0:
            raise ValueError('Only the first sublayer can initialize residual history')
        return self.input_norm(x), x

    def forward(self, branch_output: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_norm:
            normed_x, history = self.norm(branch_output, residual=history, prenorm=True)
        else:
            history = branch_output + history
            normed_x = self.norm(history)

        return normed_x, history

    @staticmethod
    def get_hidden_state(history: torch.Tensor) -> torch.Tensor:
        return history
