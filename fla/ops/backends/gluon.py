# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

from fla.ops.backends import BaseBackend


class GluonBackend(BaseBackend):
    """Enable Gluon through either the global switch or the operator's switch."""

    backend_type = 'gluon'
    package_name = 'triton.experimental.gluon'
    env_var = 'FLA_GLUON'
    default_enable = False

    @classmethod
    def is_enabled(cls) -> bool:
        return os.environ.get('FLA_GLUON', '0') != '0' or super().is_enabled()
