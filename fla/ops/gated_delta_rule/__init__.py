# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule, chunk_gdn
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule, fused_recurrent_gdn
from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule, naive_recurrent_gated_delta_rule

__all__ = [
    "chunk_gated_delta_rule", "chunk_gdn",
    "fused_recurrent_gated_delta_rule", "fused_recurrent_gdn",
    "naive_chunk_gated_delta_rule",
    "naive_recurrent_gated_delta_rule",
]


from fla.ops.gated_delta_rule.backends.flash_qla import FlashQLABackend  # noqa: F401

if TritonAscendBackend.is_available():
    from fla.ops.gated_delta_rule.backends import triton_ascend  # noqa: F401
