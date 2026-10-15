# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register
from fla.ops.utils.backends.ascendc.cumsum import chunk_local_cumsum_verifier


@register('utils')
class AscendCUtilsBackend(BaseBackend):
    backend_type = 'ascendc'
    env_var = 'FLA_ASCENDC'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls):
        from fla.utils import IS_NPU
        from fla.utils.ascendc import is_available
        return IS_NPU and is_available()

    chunk_local_cumsum_verifier = staticmethod(chunk_local_cumsum_verifier)

    def chunk_local_cumsum(self, *args, **kwargs):
        from fla.ops.utils.backends.ascendc.cumsum import chunk_local_cumsum
        return chunk_local_cumsum(*args, **kwargs)
