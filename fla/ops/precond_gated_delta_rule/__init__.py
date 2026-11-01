# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.ops.precond_gated_delta_rule.chunk import chunk_precond_gated_delta_rule
from fla.ops.precond_gated_delta_rule.fused_recurrent import fused_recurrent_precond_gated_delta_rule
from fla.ops.precond_gated_delta_rule.naive import naive_recurrent_precond_gated_delta_rule

__all__ = [
    "chunk_precond_gated_delta_rule",
    "fused_recurrent_precond_gated_delta_rule",
    "naive_recurrent_precond_gated_delta_rule",
]


if TritonAscendBackend.is_available():
    from fla.ops.precond_gated_delta_rule.backends import triton_ascend  # noqa: F401
