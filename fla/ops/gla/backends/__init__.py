# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GLA backends."""

from fla.backends import _registry_for
from fla.ops.gla.backends.triton_ascend import TritonAscendGLABackend

gla_registry = _registry_for('gla')
dispatch = gla_registry.dispatch

__all__ = ['TritonAscendGLABackend', 'dispatch', 'gla_registry']
