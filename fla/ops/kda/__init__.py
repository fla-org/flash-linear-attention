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


from fla.backends import register
from fla.ops.kda.backends import KDATileLangBackend, flash_kda  # noqa: F401

register(chunk_kda)(KDATileLangBackend)
