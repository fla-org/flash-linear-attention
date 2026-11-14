# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Linderman Lab
# Adapted from https://github.com/lindermanlab/switching-linear-attention (MIT).

"""
Tier 2 Atom Operations: Softmax Family

Numerically stable softmax, log-softmax, and their backward passes.
All use the transpose workaround for Triton axis bugs.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_2d(logits, mask_col):
    """Numerically stable softmax over axis=1 of [R, C] with masking.

    Invalid columns (mask_col=False) are set to -inf before softmax
    and zeroed in the output.

    Args:
        logits: [R, C] raw logits.
        mask_col: [C] boolean mask for valid columns.
    Returns:
        [R, C] softmax probabilities (0 for invalid columns).
    """
    # Mask invalid columns to -inf
    masked = tl.where(mask_col[None, :], logits, float('-inf'))

    # Max for numerical stability (workaround: transpose + axis=0)
    masked_t = tl.trans(masked)                             # [C, R]
    max_t = tl.max(masked_t, axis=0, keep_dims=True)       # [1, R]
    max_val = tl.trans(max_t)                               # [R, 1]
    # Guard all-masked rows (max=-inf) to avoid -inf - (-inf) = NaN
    max_val = tl.where(max_val == float('-inf'), 0.0, max_val)

    # Exp of shifted logits, zeroing invalid columns
    shifted = masked - max_val                              # [R, C]
    exp_val = tl.where(mask_col[None, :], tl.exp(shifted), 0.0)  # [R, C]

    # Sum (workaround: transpose + axis=0)
    exp_t = tl.trans(exp_val)                               # [C, R]
    sum_t = tl.sum(exp_t, axis=0, keep_dims=True)           # [1, R]
    sum_val = tl.trans(sum_t)                               # [R, 1]
    sum_val = tl.maximum(sum_val, 1e-8)                     # Prevent 0/0

    return exp_val / sum_val


@triton.jit
def softmax_backward_2d(r, d_r):
    """Backward through softmax: d_logits = r * (d_r - sum(r * d_r)).

    Args:
        r: [R, C] softmax output from forward.
        d_r: [R, C] gradient w.r.t. softmax output.
    Returns:
        [R, C] gradient w.r.t. logits.
    """
    # sum(r * d_r) over axis=1 (workaround: transpose + axis=0)
    rd_t = tl.trans(r * d_r)                              # [C, R]
    sum_rd_t = tl.sum(rd_t, axis=0, keep_dims=True)       # [1, R]
    sum_rd = tl.trans(sum_rd_t)                            # [R, 1]

    return r * (d_r - sum_rd)


@triton.jit
def log_softmax_2d(logits, mask_col):
    """Log-softmax over axis=1 of [R, C] with masking.

    Args:
        logits: [R, C] raw logits.
        mask_col: [C] boolean mask for valid columns.
    Returns:
        [R, C] log-softmax result. Invalid columns set to -inf.
    """
    # Mask invalid columns to -inf
    masked = tl.where(mask_col[None, :], logits, float('-inf'))

    # Max (workaround: transpose + axis=0)
    masked_t = tl.trans(masked)
    max_t = tl.max(masked_t, axis=0)               # [R]
    max_safe = tl.where(max_t == float('-inf'), 0.0, max_t)

    # Shift, exp, sum
    shifted = masked - max_safe[:, None]             # [R, C]
    exp_val = tl.where(mask_col[None, :], tl.exp(shifted), 0.0)
    exp_t = tl.trans(exp_val)
    sum_val = tl.maximum(tl.sum(exp_t, axis=0), 1e-8)  # [R]

    # log_softmax = shifted - log(sum)
    result = shifted - tl.log(sum_val)[:, None]
    return tl.where(mask_col[None, :], result, float('-inf'))
