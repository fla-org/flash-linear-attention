# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GDN-2 backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.gdn2.backends.fused_infer import GDN2FusedInferBackend

gdn2_registry = BackendRegistry("gdn2")
gdn2_registry.register(GDN2FusedInferBackend())

__all__ = ["dispatch", "gdn2_registry"]
