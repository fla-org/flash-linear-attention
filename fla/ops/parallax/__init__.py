# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Parallax (parameterized local linear attention), contributed by
# Yifei Zuo et al. (https://arxiv.org/abs/2605.29157).

from .naive import naive_parallax
from .parallel import parallel_parallax

__all__ = [
    'naive_parallax',
    'parallel_parallax',
]
