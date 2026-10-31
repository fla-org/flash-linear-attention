# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from .chunk import chunk_kda
from .fused_recurrent import fused_recurrent_kda

__all__ = [
    "chunk_kda",
    "fused_recurrent_kda",
]


from fla.backends import TileLangBackend, TritonAscendBackend
from fla.ops.kda.backends.flash_kda import FlashKDABackend  # noqa: F401

if TileLangBackend.is_available():
    from fla.ops.kda.backends import tilelang  # noqa: F401

if TritonAscendBackend.is_available():
    from fla.ops.kda.backends.triton_ascend import (  # noqa: F401
        chunk_bwd,
        chunk_intra,
        chunk_intra_token_parallel,
        fused_recurrent,
        gate,
        wy_fast,
    )
