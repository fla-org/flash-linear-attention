# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from torch import nn


class GroupedLinear(nn.Module):
    """Group/head-wise affine projection layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError(f"`groups` must be positive, got {groups}.")
        if in_features % groups != 0:
            raise ValueError(
                f"`in_features` ({in_features}) must be divisible by `groups` ({groups}).",
            )
        if out_features % groups != 0:
            raise ValueError(
                f"`out_features` ({out_features}) must be divisible by `groups` ({groups}).",
            )

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // groups),
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def exra_repr(self):
        s = f"{self.in_features}, {self.out_features}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return s

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, a=2)  # small init (GPT-NeoX)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.groups, -1)
        w = self.weight.view(self.groups, self.out_features // self.groups, -1)
        s = torch.einsum("...hd,hod->...ho", x, w)
        s = s.reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            s = s + self.bias
        return s
