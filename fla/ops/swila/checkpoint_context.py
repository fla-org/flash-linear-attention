# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""Checkpoint phase markers for SwiLA-specific checkpoint-aware kernels."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

CheckpointPhase = Literal["normal", "prepass", "recompute"]

_SWILA_CHECKPOINT_PHASE: ContextVar[CheckpointPhase] = ContextVar(
    "swila_checkpoint_phase",
    default="normal",
)


@contextmanager
def _checkpoint_phase(phase: CheckpointPhase) -> Iterator[None]:
    token = _SWILA_CHECKPOINT_PHASE.set(phase)
    try:
        yield
    finally:
        _SWILA_CHECKPOINT_PHASE.reset(token)


def get_swila_checkpoint_phase() -> CheckpointPhase:
    return _SWILA_CHECKPOINT_PHASE.get()


def swila_checkpoint_context_fn():
    return _checkpoint_phase("prepass"), _checkpoint_phase("recompute")
