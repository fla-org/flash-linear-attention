# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""RWKV6 backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.rwkv6.backends.tilelang import RWKV6TileLangBackend

rwkv6_registry = BackendRegistry("rwkv6")
rwkv6_registry.register(RWKV6TileLangBackend())


__all__ = ['dispatch', 'rwkv6_registry']
