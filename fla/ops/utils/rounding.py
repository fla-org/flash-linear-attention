# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


# algorithm reference: https://github.com/pytorch/ao/blob/a724e78/torchao/optim/quant_utils.py#L121
@triton.jit
def fp32_to_bf16_sr(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    """Round FP32 values to BF16 using the low 16 bits of each random integer.

    Uniform random bits give an unbiased result when both neighboring BF16 values are finite.
    Signed zeros and infinities are preserved; subnormals follow the same rounding rule.
    NaNs are quieted and overflow may produce infinity.
    The caller supplies int32 or uint32 random bits, for example with `tl.randint(seed, offsets)`.
    """
    tl.static_assert(x.dtype == tl.float32, "x must be float32")
    tl.static_assert(rand.dtype == tl.int32 or rand.dtype == tl.uint32, "rand must be int32 or uint32")
    bits = x.to(tl.uint32, bitcast=True)
    upper = bits >> 16
    increment = (rand.to(tl.uint32) & 0xFFFF) < (bits & 0xFFFF)
    rounded = upper + increment.to(tl.uint32)
    # a NaN payload confined to the discarded bits must not become infinity.
    rounded = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, upper | 0x40, rounded)
    # bitcasting avoids flushing BF16 subnormals during a floating-point conversion.
    return rounded.to(tl.uint16).to(tl.bfloat16, bitcast=True)
