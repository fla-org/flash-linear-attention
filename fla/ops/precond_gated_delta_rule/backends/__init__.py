# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDR backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.precond_gated_delta_rule.backends.triton_ascend import TritonAscendPrecondGDNBackend

pgdr_registry = BackendRegistry("precond_gated_delta_rule")

pgdr_registry.register(TritonAscendPrecondGDNBackend())


__all__ = ['dispatch', 'pgdr_registry']
