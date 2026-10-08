# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR backends."""

from fla.backends import _registry_for
from fla.ops.generalized_delta_rule.dplr.backends.tilelang import DPLRTileLangBackend

dplr_registry = _registry_for('generalized_delta_rule.dplr')
dispatch = dplr_registry.dispatch

__all__ = ['DPLRTileLangBackend', 'dispatch', 'dplr_registry']
