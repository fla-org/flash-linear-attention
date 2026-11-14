# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""SwiLA projections and explicit diagnostic, initialization and checkpoint APIs."""

from fla.modules.swila.diagnostics import (
    SwiLADiagnosticCapture,
    SwiLADiagnostics,
    SwiLAGradNorms,
    capture_swila_diagnostics,
    find_active_swila_diagnostics,
    get_expert_metrics,
)
from fla.modules.swila.initialization_checks import check_swila_initialization
from fla.modules.swila.projections import LowRankProjection, SwiGLUProjection
from fla.ops.swila.checkpoint_context import swila_checkpoint_context_fn

__all__ = [
    'LowRankProjection',
    'SwiGLUProjection',
    'SwiLADiagnosticCapture',
    'SwiLADiagnostics',
    'SwiLAGradNorms',
    'capture_swila_diagnostics',
    'check_swila_initialization',
    'find_active_swila_diagnostics',
    'get_expert_metrics',
    'swila_checkpoint_context_fn',
]
