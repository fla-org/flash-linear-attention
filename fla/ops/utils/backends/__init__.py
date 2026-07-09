# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Utils op backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.utils.backends.triton_ascend import TritonAscendUtilsBackend

utils_registry = BackendRegistry('utils')
utils_registry.register(TritonAscendUtilsBackend())

__all__ = ['dispatch', 'utils_registry']
