# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankProjection(nn.Module):
    """Simple low-rank linear projection: up(down(x)).

    Drop-in replacement for nn.Linear(d_in, d_out, bias=False).
    Param count: d_in·R + R·d_out  (vs d_in·d_out for full-rank).
    The owning SwiLA layer initializes ``down`` for unit-scale features and
    chooses an ``up`` scale appropriate to pi, beta, or alpha.
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SwiGLUProjection(nn.Module):
    """Low-rank SwiGLU projection: out(silu(gate(x)) * feat(x)).

    Drop-in replacement for nn.Linear(d_in, d_out, bias=False).
    Param count: 2·d_in·R + R·d_out  (vs d_in·d_out for full-rank).
    The owning SwiLA layer separately calibrates the input branches and terminal
    output because SwiGLU changes the intermediate feature variance.
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_in, rank, bias=False)
        self.feat = nn.Linear(d_in, rank, bias=False)
        self.out = nn.Linear(rank, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.feat(x))
