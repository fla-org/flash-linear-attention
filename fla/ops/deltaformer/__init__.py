# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from .naive import naive_deltaformer_attn
from .parallel import deltaformer_attn

__all__ = [
    'deltaformer_attn',
    'naive_deltaformer_attn',
]
