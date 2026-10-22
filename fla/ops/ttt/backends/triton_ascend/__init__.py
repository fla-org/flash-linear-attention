# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for TTT ops."""

from __future__ import annotations

from fla.backends import BaseBackend, register


@register('ttt')
class TritonAscendTTTBackend(BaseBackend):
    """Ascend NPU backend for TTT linear ops."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def fused_chunk_ttt_linear(self, *args, **kwargs):
        from fla.ops.ttt.backends.triton_ascend.fused_chunk import fused_chunk_ttt_linear_npu
        return fused_chunk_ttt_linear_npu(*args, **kwargs)


__all__ = ['TritonAscendTTTBackend']
