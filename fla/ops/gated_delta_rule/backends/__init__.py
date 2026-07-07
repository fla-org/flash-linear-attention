# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN backends for gated_delta_rule ops."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.gated_delta_rule.backends.triton_ascend import TritonAscendGDNBackend

gated_delta_rule_registry = BackendRegistry("gated_delta_rule")

gated_delta_rule_registry.register(TritonAscendGDNBackend())

__all__ = ['dispatch', 'gated_delta_rule_registry']
