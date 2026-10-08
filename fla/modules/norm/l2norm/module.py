# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn

from fla.modules.norm.l2norm.ops import l2norm


class L2Norm(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x=x, eps=self.eps, output_dtype=self.output_dtype)
