# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import NamedTuple

import torch
from torch import nn

from fla.modules.residuals.base import BaseResidual
from fla.ops.attnres import fused_attnres


class AttentionResidualState(NamedTuple):
    blocks: list[torch.Tensor]
    num_branches: int


class AttentionResidual(BaseResidual[AttentionResidualState, AttentionResidualState]):
    r"""Attention over completed residual blocks and the current block's partial sum.

    Following Attention Residuals, b_0 = h_1 is the embedding, b_n is the sum of branch outputs in block n,
    and b_n^i is its partial sum after i branches. State stores these sources and the completed branch count.
    Each attention or MLP counts as one branch; block_size=1 gives Full AttnRes.

    For the sources v_j available to layer l, the learned query w_l and key normalization define

    .. math::
        \alpha_{j\to l}
        = \operatorname{softmax}_j\!\left(w_l^\top\operatorname{RMSNorm}(v_j)\right),
        \qquad h_l = \sum_j \alpha_{j\to l} v_j.

    `readout` returns h_l; `pre` additionally applies the branch norm, separate from the key norm above.
    At a new block, `post` starts its sum with f_l(h_l); within the block, it applies

    .. math::
        b_n^i = b_n^{i-1} + f_l(h_l).

    Completed blocks remain unchanged. The output residual uses its own query to aggregate all sources,
    followed by output normalization. Gradient checkpointing requires `use_reentrant=False`.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.
    """

    supports_reentrant_checkpointing = False

    def __init__(self, hidden_size: int, block_size: int = 1, norm_eps: float = 1e-6, fuse_norm: bool = True):
        super().__init__(hidden_size=hidden_size, norm_eps=norm_eps, fuse_norm=fuse_norm)
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        self.block_size = block_size
        self.query = nn.Linear(hidden_size, 1, bias=False)
        self.query._is_attnres_proj = True
        self.key_norm = nn.RMSNorm(hidden_size, eps=self.norm.eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.zeros_(self.query.weight)
        self.key_norm.reset_parameters()

    def initialize(self, x: torch.Tensor) -> AttentionResidualState:
        return AttentionResidualState([x], 0)

    def pre(self, state: AttentionResidualState) -> tuple[torch.Tensor, AttentionResidualState]:
        if state.num_branches == 0:
            return self.norm(state.blocks[0]), state
        return self(state), state

    def post(self, branch_output: torch.Tensor, context: AttentionResidualState) -> AttentionResidualState:
        blocks, num_branches = context
        # all sources passed to fused_attnres must share the first branch's dtype.
        if num_branches == 0:
            blocks = [blocks[0].to(branch_output.dtype)]
        if num_branches % self.block_size == 0:
            blocks = [*blocks, branch_output]
        else:
            blocks = [*blocks[:-1], blocks[-1] + branch_output]
        return AttentionResidualState(blocks, num_branches + 1)

    def readout(self, state: AttentionResidualState) -> torch.Tensor:
        if state.num_branches == 0:
            return state.blocks[0]
        return fused_attnres(
            query=self.query.weight.squeeze(0),
            residuals=state.blocks,
            rms_weight=self.key_norm.weight,
            rms_eps=self.norm.eps,
        )

    def forward(self, state: AttentionResidualState) -> torch.Tensor:
        return fused_attnres(
            query=self.query.weight.squeeze(0),
            residuals=state.blocks,
            rms_weight=self.key_norm.weight,
            output_rms_weight=self.norm.weight,
            rms_eps=self.norm.eps,
        )
