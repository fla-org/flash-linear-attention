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
from fla.ops.attnres import fused_attnres


class AttentionResidual(BaseResidual[list[torch.Tensor]]):
    """Keep completed blocks and the current partial sum in an immutable-by-convention list.

    The query, key norm and output norm prepare the *next* sublayer. Boundaries instead refer to
    the current sublayer whose output is being added. The existing AttnRes op fuses the output norm.
    Initial embedding history is cast to the first branch output dtype. Later branches must share this dtype.
    """

    def __init__(self, hidden_size: int, sub_layer_idx: int = 0, block_size: int = 1, **kwargs):
        super().__init__()
        if isinstance(sub_layer_idx, bool) or not isinstance(sub_layer_idx, int) or sub_layer_idx < 0:
            raise ValueError('sub_layer_idx must be a non-negative integer')
        self.hidden_size = hidden_size
        self.sub_layer_idx = sub_layer_idx
        norm_eps = kwargs.pop('norm_eps', 1e-6)
        norm_cls = RMSNorm if kwargs.pop('fuse_norm', True) else nn.RMSNorm
        kwargs.pop('num_sublayers', None)
        if kwargs:
            raise TypeError(f'Unexpected AttentionResidual arguments: {sorted(kwargs)}')
        self.norm = norm_cls(hidden_size, eps=norm_eps)
        self.input_norm = norm_cls(hidden_size, eps=norm_eps) if sub_layer_idx == 0 else None
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        self.is_boundary = sub_layer_idx % block_size == 0
        self.query = nn.Linear(hidden_size, 1, bias=False)
        self.query._is_attnres_proj = True
        nn.init.zeros_(self.query.weight)
        self.key_norm = nn.RMSNorm(hidden_size, eps=self.norm.eps)
        if sub_layer_idx == 0:
            # retain the single-source query parameters for official checkpoint compatibility.
            self.input_query = nn.Linear(hidden_size, 1, bias=False)
            self.input_query._is_attnres_proj = True
            nn.init.zeros_(self.input_query.weight)
            self.input_key_norm = nn.RMSNorm(hidden_size, eps=self.norm.eps)

    def reset_parameters(self) -> None:
        for name in ('norm', 'input_norm', 'key_norm', 'input_key_norm'):
            norm = getattr(self, name, None)
            if norm is not None:
                norm.reset_parameters()
        nn.init.zeros_(self.query.weight)
        if self.sub_layer_idx == 0:
            nn.init.zeros_(self.input_query.weight)

    def initialize(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.sub_layer_idx != 0:
            raise ValueError('Only the first sublayer can initialize residual history')
        history = [x]
        return self.input_norm(x), history

    def forward(
        self, branch_output: torch.Tensor, history: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # the fused kernel requires all sources to share a dtype.
        if self.sub_layer_idx == 0:
            history = [history[0].to(branch_output.dtype)]

        if self.is_boundary:
            history = [*history, branch_output]
        else:
            history = [*history[:-1], history[-1] + branch_output]
        output = fused_attnres(
            query=self.query.weight,
            residuals=history,
            rms_weight=self.key_norm.weight,
            output_rms_weight=self.norm.weight,
            rms_eps=self.key_norm.eps,
        )
        return output, history

    @staticmethod
    def get_hidden_state(history: list[torch.Tensor]) -> torch.Tensor:
        return history[-1]
