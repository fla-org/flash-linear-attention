# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton backend helpers for delta-rule ops."""

from fla.ops.delta_rule.backends.triton.chunk_bwd import (
    chunk_delta_rule_wy_dqkw_fused_triton,
)

__all__ = ["chunk_delta_rule_wy_dqkw_fused_triton"]
