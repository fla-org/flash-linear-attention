# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None


class PrepareModuleWeight(ParallelStyle):
    def __init__(self, *, layouts: Placement | None = None):
        super().__init__()
        self.layouts = layouts

    def _replicate_module_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ):
        for p_name, param in module.named_parameters():
            replicated_param = nn.Parameter(
                DTensor.from_local(param, device_mesh, [self.layouts], run_check=False),
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
