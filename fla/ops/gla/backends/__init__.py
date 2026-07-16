# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GLA backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.gla.backends.triton_ascend import TritonAscendGLABackend

gla_registry = BackendRegistry('gla')
gla_registry.register(TritonAscendGLABackend())


__all__ = ['dispatch', 'gla_registry']
