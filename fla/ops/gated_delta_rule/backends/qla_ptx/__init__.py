# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)


"""Experimental fixed-specialization B300 runner; never registered for dispatch."""

from fla.ops.gated_delta_rule.backends.qla_ptx.kernel import run

__all__ = ["run"]
