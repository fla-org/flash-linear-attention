# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Based op backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.based.backends.triton_ascend import TritonAscendBasedBackend

based_registry = BackendRegistry("based")
based_registry.register(TritonAscendBasedBackend())

__all__ = ['based_registry', 'dispatch']
