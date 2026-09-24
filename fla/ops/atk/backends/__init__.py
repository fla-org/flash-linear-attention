# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""ATK backends."""

from fla.ops.atk.backends.triton_ascend import TritonAscendATKBackend
from fla.ops.backends import BackendRegistry, dispatch

atk_registry = BackendRegistry("atk")
atk_registry.register(TritonAscendATKBackend())


__all__ = ['atk_registry', 'dispatch']
