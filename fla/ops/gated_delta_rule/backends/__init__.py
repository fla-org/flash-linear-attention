# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDR backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend
from fla.ops.gated_delta_rule.backends.triton_ascend import TritonAscendGDNBackend

gdr_registry = BackendRegistry("gated_delta_rule")

gdr_registry.register(TritonAscendGDNBackend())
gdr_registry.register(FlashQLABackend())


__all__ = ['dispatch', 'gdr_registry']
