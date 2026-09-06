# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BackendRegistry
from fla.ops.common.backends.tilelang import TileLangBackend

attn_registry = BackendRegistry("attn")
attn_registry.register(TileLangBackend())
dispatch = attn_registry.dispatch

__all__ = ['attn_registry', 'dispatch']
