# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for Parallax ops."""

from __future__ import annotations

from fla.backends import BaseBackend, register


@register('parallax')
class TritonAscendParallaxBackend(BaseBackend):
    """Ascend NPU backend for Parallax single-token decode."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def parallax_decode_one_step(self, *args, **kwargs):
        from fla.ops.parallax.backends.triton_ascend.decode import parallax_decode_one_step_npu
        return parallax_decode_one_step_npu(*args, **kwargs)


__all__ = ['TritonAscendParallaxBackend']
