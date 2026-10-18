# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from functools import partial

from fla.backends import GluonBackend, TileLangBackend, register
from fla.ops.hyper_connections.fused import _post, _pre, _sinkhorn, hyper_connection_post, hyper_connection_pre, sinkhorn


class HyperConnectionTileLangBackend(TileLangBackend):
    default_enable = False


for backend_type, backend_class in (('gluon', GluonBackend), ('tilelang', HyperConnectionTileLangBackend)):
    for entry, implementation in ((hyper_connection_pre, _pre), (hyper_connection_post, _post), (sinkhorn, _sinkhorn)):
        register(
            entry,
            backend=backend_class,
            env_var=f'FLA_HYPER_CONNECTIONS_{backend_type.upper()}',
        )(partial(implementation, backend=backend_type))
