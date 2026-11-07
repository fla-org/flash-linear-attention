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
    r"""Residual aggregation with softmax weights over branch outputs or block sums.

    In Full AttnRes, state retains the embedding v_0 = h_1 and all preceding branch outputs v_i = f_i(h_i).
    The learned query w_l defines the depth mixing weights:

    .. math::
        M_{i\to l} = \alpha_{i\to l}
        = \operatorname{softmax}_i\!\left(w_l^\top\operatorname{RMSNorm}(v_i)\right),
        \qquad h_l = \sum_{i=0}^{l-1} \alpha_{i\to l} v_i.

    `readout` computes h_l; `pre` additionally applies the branch norm and returns the existing state as context.
    Key normalization only affects the attention weights; the values v_i remain unnormalized.
    `post` appends the new branch output v_l, making it available to the next residual module's query.

    Block AttnRes stores b_0 = h_1, completed block sums b_n, and the nonempty current partial sum b_n^j.
    Each attention branch and each MLP branch contributes one output; `block_size` counts these outputs.
    The same attention rule operates over these stored sources.
    All outputs in a source block share its mixing weight: M_{i->l} = alpha_{n->l} for i in block n.
    With v_l as the j-th output in block n, `post` updates the partial sum:

    .. math::
        b_n = \sum_{i\in\mathcal{B}_n} v_i,
        \qquad b_n^j = b_n^{j-1} + v_l, \quad b_n^0 = 0.

    At a block boundary, `post` starts a new block with v_l; completed blocks remain unchanged.
    State also records the completed branch count to determine these boundaries.
    `block_size=1` gives Full AttnRes. The output residual aggregates all sources with its own query,
    then applies output normalization. Gradient checkpointing requires `use_reentrant=False`.
    See Sections 3 and 6.2 of `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.
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
