# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""AttnRes backends."""

from fla.ops.attnres.backends.gluon import AttnResGluonBackend
from fla.ops.backends import BackendRegistry, dispatch

attnres_registry = BackendRegistry("attnres")
attnres_registry.register(AttnResGluonBackend())


__all__ = ['attnres_registry', 'dispatch']
