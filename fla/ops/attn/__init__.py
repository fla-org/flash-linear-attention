# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TileLangBackend
from fla.ops.attn.naive import naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn

__all__ = [
    'naive_parallel_attn',
    'parallel_attn',
]


if TileLangBackend.is_available():
    from fla.ops.attn.backends import tilelang  # noqa: F401
