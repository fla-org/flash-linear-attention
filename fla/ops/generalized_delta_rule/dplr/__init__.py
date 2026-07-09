# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from .chunk import chunk_dplr_delta_rule
from .fused_recurrent import fused_recurrent_dplr_delta_rule

__all__ = [
    'chunk_dplr_delta_rule',
    'fused_recurrent_dplr_delta_rule',
]
