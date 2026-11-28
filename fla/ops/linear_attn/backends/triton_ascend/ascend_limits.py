# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright © 2026 Huawei Technologies Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for Ascend on-chip (UB) budget constants.

Every hand-written cost model in this package used to carry its own copy of the
budget (192 KiB here, 48 KiB there, "1024 elements" elsewhere), which made it
impossible to answer "what does this kernel assume about the hardware?" and made
re-tuning for another SoC a guessing game.  All of them now read from this
module.

Values are physical-bytes constants for the target SoC family, deliberately
conservative:

* ``UB_BYTES`` — per-core Unified Buffer.  Ascend 910B (A2) and 950PR (A5) are
  both in the 192-216 KiB range; 192 KiB is the smaller of the two, so tiling
  derived from it is safe on both.
* ``DH_UB_BUDGET_BYTES`` — the tighter working set the ``chunk_bwd_dh`` kernel
  can afford once ``dht`` seeds an extra fp32 ``[BK, BV]`` tile.  Exceeding it
  does not fail at compile time on Ascend; it manifests as a bulk MTE write past
  the allocation and an aicore exception (507015) at run time.
* ``MAX_VECTOR_TILE_ELEMS`` — element cap for 2-D ``tl.cumsum`` tiles.  The scan
  temporary costs ~96 B/element regardless of dtype, so ``[64, 64]`` needs
  ~386 KiB and aborts ``PlanMemory`` on a 192 KiB core.

Every value can be overridden with an environment variable, which is the escape
hatch for running on a SoC whose UB is larger or smaller than assumed::

    OKLP_UB_BYTES=221184 OKLP_DH_UB_BUDGET_BYTES=49152 python -m pytest tests -q
"""

from __future__ import annotations

import os

#: Per-core Unified Buffer size in bytes (192 KiB: the smaller of A2 / A5).
UB_BYTES: int = int(os.environ.get("OKLP_UB_BYTES", 192 * 1024))

#: Fraction of the UB a kernel is allowed to plan for.
SAFETY_MARGIN: float = float(os.environ.get("OKLP_UB_SAFETY_MARGIN", 0.80))

#: Working-set cap for the ``chunk_bwd_dh`` kernel when ``dht`` is present.
DH_UB_BUDGET_BYTES: int = int(os.environ.get("OKLP_DH_UB_BUDGET_BYTES", 48 * 1024))

#: Maximum element count for a 2-D ``tl.cumsum`` tile (scan temporary ~96 B/elem).
MAX_VECTOR_TILE_ELEMS: int = int(os.environ.get("OKLP_MAX_CUMSUM_TILE_ELEMS", 1024))


def ub_budget_bytes() -> int:
    """Bytes a hand-written cost model may plan for (UB size × safety margin)."""
    return int(UB_BYTES * SAFETY_MARGIN)


__all__ = [
    "DH_UB_BUDGET_BYTES",
    "MAX_VECTOR_TILE_ELEMS",
    "SAFETY_MARGIN",
    "UB_BYTES",
    "ub_budget_bytes",
]
