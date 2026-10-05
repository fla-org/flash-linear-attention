# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


@triton.jit
def make_block_ptr(base, shape, strides, offsets, block_shape: tl.constexpr, order: tl.constexpr):
    """Keep Ascend's local block offsets int32 while base-address arithmetic stays int64."""
    offsets_i32 = ()
    for i in tl.static_range(len(offsets)):
        offsets_i32 += (tl.cast(offsets[i], tl.int32),)
    return tl.make_block_ptr(
        base=base,
        shape=shape,
        strides=strides,
        offsets=offsets_i32,
        block_shape=block_shape,
        order=order,
    )
