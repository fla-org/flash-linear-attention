# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from .chunk import chunk_gdn2
from .fused_recurrent import fused_recurrent_gdn2
from .naive import naive_chunk_gdn2, naive_recurrent_gdn2

__all__ = [
    "chunk_gdn2",
    "fused_recurrent_gdn2",
    "naive_chunk_gdn2",
    "naive_recurrent_gdn2",
]


from fla.backends import TritonAscendBackend

if TritonAscendBackend.is_available():
    from fla.ops.gdn2.backends.triton_ascend import chunk_bwd, chunk_intra, fused_recurrent  # noqa: F401
