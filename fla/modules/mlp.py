# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.modules.activations import swiglu, swiglu_linear

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

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
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if self.fuse_swiglu:
            self.swiglu_linear = SwiGLULinear()

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[Any]
    ) -> torch.Tensor:
        gate, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return self.swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
        else:
            return self.down_proj(swiglu(gate, y))


class SwiGLULinear(nn.Module):

    def forward(self, x, y, weight, bias):
        return swiglu_linear(x, y, weight, bias)


class SwiGLULinearParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.desired_input_layouts = (Shard(-1),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        x, y, weight, bias = inputs
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, device_mesh, input_layouts, run_check=False)
        if x.placements != desired_input_layouts:
            x = x.redistribute(placements=desired_input_layouts, async_op=True)

        if not isinstance(y, DTensor):
            y = DTensor.from_local(y, device_mesh, input_layouts, run_check=False)
        if y.placements != desired_input_layouts:
            y = y.redistribute(placements=desired_input_layouts, async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, (Shard(1),))

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, (Replicate(),))

        return x, y, weight, bias

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            output_fn=partial(self._prepare_output_fn, self.output_layouts, self.use_local_output)
        )
