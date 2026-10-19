# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Optional Ascend wheel implementations for KDA operations."""

import logging
import os
from functools import cache

from fla.backends import BaseBackend, register
from fla.utils import IS_NPU, find_spec_cached

logger = logging.getLogger(__name__)


@cache
def _warn_missing_wheel() -> None:
    logger.warning(
        'AscendC was enabled but fla_npu is not installed; trying other backends. '
        'From the FLA source checkout, install exactly one wheel matching your NPU: '
        'A2 (910B): python -m pip install -e ".[npu-ascendc-a2]"; '
        'A3 (910_93): python -m pip install -e ".[npu-ascendc-a3]"; '
        'A5 (950): python -m pip install -e ".[npu-ascendc-a5]". '
        'Then restart Python. The current KDA adapter supports A5 only.'
    )


def is_ascendc_available(env_var: str) -> bool:
    if not IS_NPU:
        return False
    if find_spec_cached('fla_npu') is not None:
        return True
    if os.environ.get('FLA_ASCENDC', '0') != '0' or os.environ.get(env_var, '0') != '0':
        _warn_missing_wheel()
    return False


@register('kda', env_var='FLA_KDA_ASCENDC_OPS')
class AscendCKDAOpsBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls) -> bool:
        return is_ascendc_available('FLA_KDA_ASCENDC_OPS')

    def kda_gate_chunk_cumsum_verifier(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.gate import kda_gate_chunk_cumsum_verifier
        return kda_gate_chunk_cumsum_verifier(*args, **kwargs)

    def kda_gate_chunk_cumsum(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.gate import kda_gate_chunk_cumsum
        return kda_gate_chunk_cumsum(*args, **kwargs)

    def chunk_kda_bwd_intra_verifier(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.chunk_intra import chunk_kda_bwd_intra_verifier
        return chunk_kda_bwd_intra_verifier(*args, **kwargs)

    def chunk_kda_bwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.chunk_intra import chunk_kda_bwd_intra
        return chunk_kda_bwd_intra(*args, **kwargs)
