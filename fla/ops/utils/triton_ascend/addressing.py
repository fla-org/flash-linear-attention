# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared Triton-Ascend pointer and token-row helpers."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def mload(ptr, mask, MASK: tl.constexpr):
    """Mask only when ``MASK`` is true. Aligned launches DCE the predicate and keep a static DMA."""
    if MASK:
        return tl.load(ptr, mask=mask, other=0)
    return tl.load(ptr)


@triton.jit
def mstore(ptr, val, mask, MASK: tl.constexpr):
    if MASK:
        tl.store(ptr, val, mask=mask)
    else:
        tl.store(ptr, val)


@triton.jit
def chunk_token_rows(i_t, T, BT: tl.constexpr):
    """In-chunk token offsets with an int64 chunk base.

    Row lanes stay int32; the chunk origin is promoted before multiplying by ``BT``.
    """
    rows = tl.arange(0, BT)
    row_base = tl.cast(i_t, tl.int64) * BT
    limit = tl.cast(T, tl.int32) - tl.cast(row_base, tl.int32)
    return row_base, rows, rows < limit


@triton.jit
def token_rows(origin, T, WIDTH: tl.constexpr):
    """int32 row lanes from a scalar token origin. See ``chunk_token_rows``."""
    rows = tl.arange(0, WIDTH)
    base = tl.cast(origin, tl.int64)
    limit = tl.cast(T, tl.int32) - tl.cast(origin, tl.int32)
    return base, rows, rows < limit


@triton.jit
def clamp_token_rows(rows, T):
    """Point tail rows at a real token.

    Masked loads of ``row >= T`` still form that address. Past the allocation
    the read is NaN, and a NaN lane on the reduction axis poisons ``final_state``.
    Clamping keeps the DMA inside the sequence; the predicate still drops the lane.
    The clamp is an index vector of length BT, not another data tile.
    """
    last = tl.maximum(tl.cast(T, tl.int64) - 1, 0)
    return tl.minimum(rows, last)


@triton.jit
def add_token_rows(ptr, row_base, rows, stride, SPLIT: tl.constexpr):
    """Token-axis pointer offset.

    SPLIT adds the int64 chunk base to the pointer first, so the row vector
    stays int32. Masked launches keep one int64 offset: splitting it there
    mismatches the tail predicate and drops those rows.
    """
    if SPLIT:
        return (ptr + row_base * stride) + rows * stride
    return ptr + rows * stride
