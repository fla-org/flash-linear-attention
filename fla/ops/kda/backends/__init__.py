# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KDA backends."""

from fla.backends import TritonAscendBackend
from fla.ops.kda.backends.flash_kda import FlashKDABackend  # noqa: F401
from fla.ops.kda.backends.tilelang import KDATileLangBackend  # noqa: F401

if TritonAscendBackend.is_available():
    from fla.ops.kda.backends.triton_ascend import (  # noqa: F401
        chunk_bwd,
        chunk_intra,
        chunk_intra_token_parallel,
        fused_recurrent,
        gate,
        wy_fast,
    )
