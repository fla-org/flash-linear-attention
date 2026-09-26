# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""AttnRes backends."""

from fla.ops.attnres.backends.triton_ascend import TritonAscendAttnResBackend
from fla.ops.backends import BackendRegistry, dispatch

attnres_registry = BackendRegistry("attnres")

attnres_registry.register(TritonAscendAttnResBackend())

# gluon.py imports triton.experimental at module load; without this guard, import fails
# on NPU and attnres backends never register (including triton_ascend above).
try:
    from fla.ops.attnres.backends.gluon import AttnResGluonBackend

    attnres_registry.register(AttnResGluonBackend())
except ImportError:
    pass


__all__ = ['attnres_registry', 'dispatch']
