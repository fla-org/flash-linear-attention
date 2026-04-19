# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Common backends for shared operations like chunk_gated_delta_rule_fwd_h."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.common.backends.intracard import IntraCardCPBackend
from fla.ops.common.backends.tilelang import TileLangBackend

common_registry = BackendRegistry("common")


common_registry.register(IntraCardCPBackend())
common_registry.register(TileLangBackend())


__all__ = ['common_registry', 'dispatch']
