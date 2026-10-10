# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.softplus import softplus, softplus2
from fla.utils import assert_close, device


@triton.jit
def softplus_kernel(X, Y, N: tl.constexpr, BASE2: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    if BASE2:
        y = softplus2(x)
    else:
        y = softplus(x)
    tl.store(Y + offsets, y, offsets < N)


@pytest.mark.parametrize('base2', [False, True], ids=['softplus', 'softplus2'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=['fp32', 'fp16', 'bf16'])
def test_softplus_compile(base2: bool, dtype: torch.dtype):
    x = torch.cat([
        torch.linspace(-25, 25, 1025, device=device),
        torch.tensor([-100, 14.9, 15, 15.1, 19.9, 20, 20.1, 100], device=device),
    ]).to(dtype)

    def apply_softplus(x):
        y = torch.empty_like(x, dtype=torch.float32)
        softplus_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), base2, 256)
        return y

    ref = F.softplus(x.float() * math.log(2)) / math.log(2) if base2 else F.softplus(x.float())
    eager = apply_softplus(x)
    compiled = torch.compile(apply_softplus, fullgraph=True)(x)

    assert_close('softplus reference', ref, eager, 1e-6)
    assert_close('softplus compiled', eager, compiled, 0.0, err_atol=0.0)
