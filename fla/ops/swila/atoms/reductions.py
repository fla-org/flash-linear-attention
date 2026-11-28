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
Tier 1 Atom Operations: Reduction Primitives

These exist to work around Triton bugs with reductions on non-zero axes:
- tl.sum(2D, axis=1) produces wrong results at small block sizes
- tl.max(2D, axis=1) produces wrong results at small block sizes
- tl.sum(3D, axis=2) produces wrong results

The workaround is: transpose so the target axis becomes axis=0, then reduce.

All functions are @triton.jit and get inlined by the compiler with zero overhead.
"""

import triton
import triton.language as tl


@triton.jit
def sum_2d_axis1(x):
    """Sum over axis=1 of 2D tensor: [R, C] -> [R].

    Workaround for tl.sum(2D, axis=1) bug. Uses transpose + axis=0.

    Args:
        x: [R, C] float32 tensor. Invalid columns should be zeroed by caller.
    Returns:
        [R] float32 tensor with sum over columns.
    """
    x_t = tl.trans(x)          # [C, R]
    return tl.sum(x_t, axis=0)  # [R]


@triton.jit
def dot_3d_axis2(h, vec):
    """Dot product: sum over last axis of [A, B, C] * [C] -> [A, B].

    Computes h @ vec by element-wise multiply then sum over axis=2.
    Workaround for tl.sum(3D, axis=2) bug. Uses trans(2,0,1) + axis=0.

    Args:
        h: [A, B, C] float32 tensor (e.g. state [BV, BE, BK]).
        vec: [C] float32 tensor (e.g. key or query [BK]).
    Returns:
        [A, B] float32 tensor.
    """
    product = h * vec[None, None, :]       # [A, B, C]
    product_t = tl.trans(product, 2, 0, 1)  # [C, A, B]
    return tl.sum(product_t, axis=0)        # [A, B]
