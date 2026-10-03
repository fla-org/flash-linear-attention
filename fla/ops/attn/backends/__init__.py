# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.ops.attn.backends.gluon import AttnGluonBackend
from fla.ops.backends import BackendRegistry
from fla.ops.common.backends.tilelang import TileLangBackend

registry = BackendRegistry("attn")
registry.register(AttnGluonBackend())
registry.register(TileLangBackend())
