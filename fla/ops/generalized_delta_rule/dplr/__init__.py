# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.ops.generalized_delta_rule.dplr.chunk import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.fused_recurrent import fused_recurrent_dplr_delta_rule

__all__ = [
    'chunk_dplr_delta_rule',
    'fused_recurrent_dplr_delta_rule',
]


from fla.ops.generalized_delta_rule.dplr.backends.tilelang import DPLRTileLangBackend  # noqa: F401
