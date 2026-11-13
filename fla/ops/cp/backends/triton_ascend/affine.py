# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Bound local affine-state transforms within Ascend's on-chip matrix buffers."""

import triton
import triton.language as tl
from triton.language.extra.cann.extension import extract_slice, insert_slice


def cp_state_requires_tiling(key_dim: int, *, backward: bool = False) -> bool:
    """Keep row tiles for wide transforms, backward tails and tails fitting one row tile."""
    key_width = max(16, triton.next_power_of_2(key_dim))
    # A single row tile has no row-loop overhead to remove in the forward path.
    return key_width > 128 or (key_width != key_dim and (backward or key_width == 16))


def cp_state_column_block_size(key_dim: int) -> int:
    """Budget 16 KiB for partial-key and wide state tiles; preserve aligned small tiles."""
    key_width = max(16, triton.next_power_of_2(key_dim))
    if key_width == key_dim and key_width <= 128:
        return 64
    return min(64, 4096 // key_width)


@triton.jit
def chunk_cp_affine_update(left, right, decay, state, BK: tl.constexpr, USE_BG: tl.constexpr):
    """Apply the chunk transform in row tiles without materializing its full K-by-K matrix."""
    tl.static_assert(BK >= 16, "CP affine row tiling requires BK >= 16")
    tl.static_assert(BK <= 256, "CP affine row tiling requires BK <= 256")
    BR: tl.constexpr = 16
    columns = tl.arange(0, BK)
    result = tl.full(state.shape, 0.0, tl.float32)
    for block in range(BK // BR):
        rows = block * BR + tl.arange(0, BR)
        left_rows = extract_slice(left, (0, block * BR), (left.shape[0], BR), (1, 1))
        product = tl.dot(tl.trans(left_rows), right)
        diagonal = tl.gather(decay, rows, axis=0)
        diagonal = tl.where(rows[:, None] == columns[None, :], diagonal[:, None], 0.0)
        step = diagonal + product if USE_BG else diagonal - product
        updated = tl.dot(step, state, input_precision="ieee")
        result = insert_slice(result, updated, (block * BR, 0), (BR, state.shape[1]), (1, 1))
    return result
