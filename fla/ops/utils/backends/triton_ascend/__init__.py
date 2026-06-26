# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend NPU backend for utils ops."""

from __future__ import annotations

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
