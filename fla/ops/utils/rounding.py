# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


# algorithm reference: https://github.com/pytorch/ao/blob/a724e78/torchao/optim/quant_utils.py#L121
@triton.jit(do_not_specialize=['T'])
def fp32_to_bf16_sr_kernel(x, rand, y, seed, T, BT: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    b_x = tl.load(x + o_t, mask=o_t < T, other=0)
    if rand is None:
        b_r = tl.randint(tl.load(seed), o_t)
    else:
        b_r = tl.load(rand + o_t, mask=o_t < T, other=0)
    b_x = b_x.to(tl.uint32, bitcast=True)
    b_r = b_r.to(tl.uint32) & 0xFFFF
    b_y = (b_x >> 16) + (b_r < (b_x & 0xFFFF)).to(tl.uint32)
    # preserve NaNs whose payload is confined to the discarded bits.
    b_y = tl.where((b_x & 0x7FFFFFFF) > 0x7F800000, (b_x >> 16) | 0x40, b_y)
    b_y = b_y.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(y + o_t, b_y, mask=o_t < T)


@input_guard
def fp32_to_bf16_sr(x: torch.Tensor, rand: torch.Tensor | None = None) -> torch.Tensor:
    """Convert an FP32 tensor to BF16 with stochastic rounding. Backward is unsupported.

    Signed zeros and infinities are preserved, subnormals are rounded, and NaNs are quieted.
    Overflow may produce infinity; the result is unbiased only between finite BF16 neighbors.

    Args:
        x (torch.Tensor):
            FP32 input tensor.
        rand (torch.Tensor, Optional):
            Int32 or uint32 random bits with the same shape and device as `x`. Only the low 16 bits are used.
            If `None`, generate Philox random bits using a seed drawn from the device's PyTorch RNG. Default: `None`.

    Returns:
        torch.Tensor:
            BF16 tensor with the same shape and device as `x`.
    """
    if x.dtype != torch.float32:
        raise TypeError("x must be float32")
    if rand is not None:
        if rand.dtype not in (torch.int32, torch.uint32):
            raise TypeError("rand must be int32 or uint32")
        if rand.shape != x.shape or rand.device != x.device:
            raise ValueError("rand must have the same shape and device as x")
    y = torch.empty_like(x, dtype=torch.bfloat16)
    T = x.numel()
    seed = torch.randint(0, 2**32, (), dtype=torch.int64, device=x.device) if rand is None else None
    fp32_to_bf16_sr_kernel[(triton.cdiv(T, 256),)](x=x, rand=rand, y=y, seed=seed, T=T, BT=256)
    return y
