# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.generalized_delta_rule.dplr.backends.tilelang import DPLRTileLangBackend

dplr_registry = BackendRegistry("generalized_delta_rule.dplr")
dplr_registry.register(DPLRTileLangBackend())


__all__ = ['dispatch', 'dplr_registry']
