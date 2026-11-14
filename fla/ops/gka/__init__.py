# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend

from .chunk import chunk_gka
from .fused_recurrent import fused_recurrent_gka
from .naive import naive_recurrent_gka, naive_recurrent_gka_chebyshev

__all__ = ['chunk_gka', 'fused_recurrent_gka', 'naive_recurrent_gka', 'naive_recurrent_gka_chebyshev']

if TritonAscendBackend.is_available():
    from fla.ops.gka.backends import triton_ascend  # noqa: F401
