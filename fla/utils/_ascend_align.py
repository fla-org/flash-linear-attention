# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend leftover-tile masking.

``MASK_LEFTOVER`` zeros partial last-T tiles and varlen lanes. Boundary checks
on Ascend do not clear those lanes, and a padded last chunk can spill into the
next batch. Last-dim leftover is left to the kernel's own boundary checks.
"""

from __future__ import annotations


def npu_leftover_mask(
    *,
    T: int | None = None,
    BT: int | None = None,
    varlen: bool = False,
    K: int | None = None,
    BK: int | None = None,
    V: int | None = None,
    BV: int | None = None,
) -> bool:
    """True when a kernel must zero leftover T lanes (partial last tile / varlen).

    ``K`` / ``BK`` / ``V`` / ``BV`` are accepted for call-site compatibility and ignored.
    """
    del K, BK, V, BV
    if varlen:
        return True
    return T is not None and BT is not None and int(T) % int(BT) != 0
