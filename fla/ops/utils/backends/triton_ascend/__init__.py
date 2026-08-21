# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend NPU backend for utils ops."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend


class TritonAscendUtilsBackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def solve_tril_verifier(self, A: torch.Tensor, *args, **kwargs) -> tuple[bool, str | None]:
        from fla.utils import IS_NPU
        if not IS_NPU:
            return False, "not running on NPU"
        if A.device.type != 'npu':
            return False, f"input device is not NPU, got {A.device.type}"
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, f"unsupported dtype for NPU solve_tril: {A.dtype}"
        if A.shape[-1] not in (16, 32, 64):
            return False, f"solve_tril requires BT in (16, 32, 64), got {A.shape[-1]}"
        return True, None

    def solve_tril(self, *args, **kwargs):
        from fla.ops.utils.backends.triton_ascend.solve_tril import solve_tril_npu
        return solve_tril_npu(*args, **kwargs)

    def chunk_global_cumsum_verifier(self, *args, **kwargs):
        return True, None

    def chunk_global_cumsum(self, *args, **kwargs):
        from fla.ops.utils.backends.triton_ascend.cumsum import chunk_global_cumsum_npu
        return chunk_global_cumsum_npu(*args, **kwargs)

    def chunk_local_cumsum_verifier(self, *args, **kwargs):
        return True, None

    def chunk_local_cumsum(self, *args, **kwargs):
        from fla.ops.utils.backends.triton_ascend.cumsum import chunk_local_cumsum_npu
        return chunk_local_cumsum_npu(*args, **kwargs)
