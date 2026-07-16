# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for GLA ops."""

from __future__ import annotations

from fla.ops.backends import BaseBackend


class TritonAscendGLABackend(BaseBackend):
    """Ascend NPU backend for GLA chunk forward output kernels."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def chunk_gla_fwd_o_gk_verifier(self, *args, **kwargs):
        return True, None

    def chunk_gla_fwd_o_gk(self, *args, **kwargs):
        from fla.ops.gla.backends.triton_ascend.chunk_fwd_o import chunk_gla_fwd_o_gk_npu
        return chunk_gla_fwd_o_gk_npu(*args, **kwargs)
