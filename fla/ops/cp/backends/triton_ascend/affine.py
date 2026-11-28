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
    """Select 16-row affine updates instead of forming the full padded key-by-key transform.

    The padded key width BK is the next power of two, with a minimum of 16.
    Row tiling limits the intermediate transform to [16, BK] rather than [BK, BK].
    Use it for BK > 128 in both directions and for any partially filled key tile in backward.
    Forward keeps the full transform for partial tiles with BK of 32, 64 or 128 to avoid row-loop overhead.
    Partial tiles with BK = 16 use tiling in both directions because they need only one row tile.
    This selects the row-update path; cp_state_column_block_size controls column width separately.

    Args:
        key_dim (int):
            Actual key dimension K before padding, rather than the padded kernel width BK.
        backward (bool, Optional):
            Select the backward policy, which also tiles partially filled key dimensions. Default: `False`.

    Returns:
        bool:
            Whether to use chunk_cp_affine_update for the affine transform update.
    """
    key_width = max(16, triton.next_power_of_2(key_dim))
    # A single row tile has no row-loop overhead to remove in the forward path.
    return key_width > 128 or (key_width != key_dim and (backward or key_width == 16))


def cp_state_column_block_size(key_dim: int) -> int:
    """Choose a column-width cap for FP32 state tiles independently of row tiling.

    The padded key width BK is the next power of two, with a minimum of 16.
    For partial-key or wide transforms, keep each [BK, column_width] FP32 tile within 16 KiB:
    4096 elements / BK rows, capped at 64 columns. Other live buffers are outside this per-tile budget.
    Preserve 64 columns when K = BK <= 128; this exception can exceed 16 KiB per tile at K = 128.
    For example, K = 100 gives BK = 128 and 32 columns, while K = 256 gives 16 columns.
    Callers decide when to apply this cap and may reduce it further for their own buffer budgets.
    A column cap can apply even when cp_state_requires_tiling selects the full row transform.

    Args:
        key_dim (int):
            Actual key dimension K before padding, rather than the padded kernel width BK.

    Returns:
        int:
            Column-width cap for a state tile; this does not change the logical key or value dimensions.
    """
    key_width = max(16, triton.next_power_of_2(key_dim))
    if key_width == key_dim and key_width <= 128:
        return 64
    return min(64, 4096 // key_width)


@triton.jit
def chunk_cp_affine_update(left, right, decay, state, BK: tl.constexpr, USE_BG: tl.constexpr):
    """Left-compose one chunk's linear state transform using 16-row intermediate tiles.

    Compute (diag(decay) +/- left.T @ right) @ state, with + for DPLR and - for GDN/KDA.
    Callers swap left/right for the backward transform, so the same update serves both directions.
    Here state is a column block of the accumulated CP transform, not the additive local-state contribution.

    Each iteration forms only [16, BK] transform rows and multiplies them by the full input state block.
    The returned [BK, C] block assembles these independent row results; state is not updated in place.
    This avoids materializing the full [BK, BK] chunk transform while retaining the FP32 accumulated state.
    Callers zero invalid token/key lanes in the operands and mask stores to the logical output dimensions.

    Args:
        left (tl.tensor):
            Chunk operand of shape [BT, BK], where BT is the token tile size and BK includes key padding.
            Forward uses the gated key-like operand; backward uses the WY operand w.
        right (tl.tensor):
            Matching operand of shape [BT, BK], with the same dtype as left.
            Forward uses w; backward uses the gated key-like operand.
        decay (tl.tensor):
            FP32 diagonal factors of shape [BK], already expanded for scalar gates or set to one without decay.
        state (tl.tensor):
            FP32 accumulated transform block of shape [BK, C], where C is the current column tile width.
        BK (tl.constexpr):
            Padded power-of-two key width, from 16 through 256; each iteration produces 16 output rows.
        USE_BG (tl.constexpr):
            Use addition for the DPLR bg/w transform when `True`, or subtraction for GDN/KDA when `False`.

    Returns:
        tl.tensor:
            Updated FP32 transform block of shape [BK, C], computed with IEEE precision for the state product.
    """
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
