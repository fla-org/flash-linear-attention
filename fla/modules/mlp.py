# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from fla.modules.activations import ACT2FN, swiglu, swiglu_linear

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class GatedMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True
    ) -> GatedMLP:
        super().__init__()

        self.hidden_size = hidden_size
        self.fuse_swiglu = fuse_swiglu
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[Any]
    ) -> torch.Tensor:
        gate, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
        else:
            return self.down_proj(swiglu(gate, y))
