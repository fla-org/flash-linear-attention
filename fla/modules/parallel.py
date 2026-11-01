# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None


class PrepareModuleWeight(ParallelStyle):
    """Wrap parameters that already match their layout on a one-dimensional mesh.

    Replicated values must match across ranks; sharded values must already be local slices.
    """

    def __init__(self, *, layouts: Placement | None = None):
        super().__init__()
        self.layouts = Replicate() if layouts is None else layouts

    def _replicate_module_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ):
        for p_name, param in module.named_parameters(recurse=False):
            replicated_param = nn.Parameter(
                DTensor.from_local(param, device_mesh, [self.layouts], run_check=False),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(p_name, replicated_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._replicate_module_fn,
            input_fn=None,
            output_fn=None,
        )
