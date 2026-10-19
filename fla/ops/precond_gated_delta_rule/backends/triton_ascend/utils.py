# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def _require_npu_device(*tensors: torch.Tensor | None) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "npu":
            return False, "input device is not NPU"
    return True, None
