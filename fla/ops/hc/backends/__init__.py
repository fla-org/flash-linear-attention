# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from functools import partial

from fla.backends import GluonBackend, TileLangBackend, register
from fla.ops.hc.fused import _hc_post, _hc_pre, _sinkhorn, hc_post, hc_pre, sinkhorn


class HCTileLangBackend(TileLangBackend):
    default_enable = False


for backend_type, backend_class in (('gluon', GluonBackend), ('tilelang', HCTileLangBackend)):
    for entry, implementation in ((hc_pre, _hc_pre), (hc_post, _hc_post), (sinkhorn, _sinkhorn)):
        register(entry, backend=backend_class, env_var=f'FLA_HC_{backend_type.upper()}')(
            partial(implementation, backend=backend_type),
        )
