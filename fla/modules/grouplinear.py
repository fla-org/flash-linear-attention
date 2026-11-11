# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Adapted from https://github.com/NX-AI/xlstm/blob/main/xlstm/components/linear_headwise.py

import torch
import torch.nn as nn


class GroupLinear(nn.Module):
    r"""
    Apply independent linear projections to equal-sized groups of input features.

    For each group :math:`g`, the projection is

    .. math::
        y_g = x_g W_g^{\mathsf{T}} + b_g, \qquad
        W_g \in \mathbb{R}^{(D_{\mathrm{out}} / G) \times (D_{\mathrm{in}} / G)}.

    Concatenating the group outputs gives a linear layer with a block-diagonal weight matrix,
    using :math:`D_{\mathrm{in}} D_{\mathrm{out}} / G` weights and no mixing between groups.
    This follows the headwise projections used in [xLSTM](https://arxiv.org/abs/2405.04517),
    Appendix A.4, for the mLSTM query and key projections.

    Args:
        in_features (int):
            Total input feature dimension. Must be divisible by `groups`.
        out_features (int):
            Total output feature dimension. Must be divisible by `groups`.
        groups (int, Optional):
            Number of independent projection groups. Default: 1.
        bias (bool, Optional):
            Whether to learn an additive bias. Default: `True`.

    Shape:
        Input: `[..., in_features]`. Output: `[..., out_features]`.
        Weight: `[out_features, in_features // groups]`. Bias: `[out_features]`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError(f'`groups` must be positive, got {groups}.')
        if in_features % groups != 0:
            raise ValueError(f'`in_features` ({in_features}) must be divisible by `groups` ({groups}).')
        if out_features % groups != 0:
            raise ValueError(f'`out_features` ({out_features}) must be divisible by `groups` ({groups}).')

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features // groups))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'groups={self.groups}, bias={self.bias is not None}'
        )

    def reset_parameters(self) -> None:
        # xLSTM small initialization: std = sqrt(2 / (5 * in_features_per_group)).
        nn.init.kaiming_normal_(self.weight, a=2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unflatten(-1, (self.groups, self.in_features // self.groups))
        weight = self.weight.view(self.groups, self.out_features // self.groups, self.in_features // self.groups)
        y = torch.einsum('...gi,goi->...go', x, weight).flatten(-2)
        if self.bias is not None:
            y = y + self.bias
        return y
