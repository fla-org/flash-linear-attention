# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import triton
import triton.language as tl

from fla.ops.utils.rounding import stochastic_rounding_bf16
from fla.utils import device


@triton.jit(do_not_specialize=['N'])
def rounding_fwd_kernel(x, rand, y, N, seed, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets, mask=offsets < N, other=0)
    if rand is None:
        random_bits = tl.randint(seed, offsets)
    else:
        random_bits = tl.load(rand + offsets, mask=offsets < N, other=0)
    tl.store(y + offsets, stochastic_rounding_bf16(x=values, rand=random_bits), mask=offsets < N)


def _rounding(x: torch.Tensor, rand: torch.Tensor | None = None, seed: int = 42, block_size: int = 256) -> torch.Tensor:
    x = x.to(device)
    rand = rand.to(device) if rand is not None else None
    y = torch.empty_like(x, dtype=torch.bfloat16)
    grid = (triton.cdiv(x.numel(), block_size),)
    rounding_fwd_kernel[grid](x=x, rand=rand, y=y, N=x.numel(), seed=seed, BLOCK_SIZE=block_size)
    return y.cpu()


@pytest.mark.parametrize('N', [pytest.param(N, id=f'N{N}') for N in [1, 1025, 65537]])
def test_stochastic_rounding_bf16(N: int):
    torch.manual_seed(42)
    bits = torch.randint(0, 0x7F7F0001, (N,), dtype=torch.int64)
    bits |= torch.randint(0, 2, (N,), dtype=torch.int64) << 31
    x = bits.to(torch.int32).view(torch.float32)
    rand = torch.randint(-(2**31), 2**31, (N,), dtype=torch.int32)

    # fp64 distances give an independent oracle, including the subnormal range.
    magnitude = x.double().abs()
    nearest = magnitude.to(torch.bfloat16)
    toward_zero = torch.where(
        nearest.double() > magnitude,
        torch.nextafter(nearest, torch.zeros_like(nearest)),
        nearest,
    )
    away_zero = torch.nextafter(toward_zero, torch.full_like(toward_zero, float('inf')))
    probability = (magnitude - toward_zero.double()) / (away_zero.double() - toward_zero.double())
    ref = torch.where((rand.to(torch.int64) & 0xFFFF).double() / 65536 < probability, away_zero, toward_zero)
    ref = torch.copysign(ref, x.to(torch.bfloat16))
    tri = _rounding(x=x, rand=rand)
    torch.testing.assert_close(tri.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)


def test_stochastic_rounding_bf16_exact():
    bits = torch.arange(65536, dtype=torch.int32)
    bits = bits[(bits & 0x7FFF) <= 0x7F80]
    x = (bits << 16).view(torch.float32)
    rand = torch.full_like(bits, 0xFFFF)
    tri = _rounding(x=x, rand=rand)
    torch.testing.assert_close(tri.view(torch.int16), bits.to(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize(
    ('bits', 'fraction'),
    [
        pytest.param(bits, fraction, id=f'bits{bits:08x}-p{fraction}')
        for bits, fraction in [
            (0x3F804000, 0.25),
            (0xBFC0C000, 0.75),
            (0x3FFF8000, 0.5),
            (0x00000001, 1 / 65536),
            (0x007FFFFF, 65535 / 65536),
        ]
    ],
)
def test_stochastic_rounding_bf16_probability(bits: int, fraction: float):
    x = torch.full((65536,), bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    rand = torch.arange(65536, dtype=torch.int32)
    tri = _rounding(x=x, rand=rand)
    away_zero = (tri.view(torch.int16).to(torch.int32) & 0xFFFF) == (bits >> 16) + 1
    assert away_zero.sum().item() == int(fraction * 65536)
    torch.testing.assert_close(tri.double().mean(), x[0].double(), rtol=0, atol=0)


def test_stochastic_rounding_bf16_special_values():
    bits = torch.tensor(
        [0x7F800001, 0xFF800001, 0x7FC12345, 0xFFFFFFFF, 0x7F7FFFFF, 0xFF7FFFFF],
        dtype=torch.int64,
    )
    x = bits.to(torch.int32).view(torch.float32)
    rand = torch.zeros(x.shape, dtype=torch.int32)
    tri = _rounding(x=x, rand=rand)
    assert torch.isnan(tri[:4]).all()
    assert ((tri[:4].view(torch.int16).to(torch.int32) & 0x40) != 0).all()
    torch.testing.assert_close(torch.signbit(tri), torch.signbit(x), rtol=0, atol=0)
    torch.testing.assert_close(tri[4:].float(), torch.tensor([float('inf'), -float('inf')]), rtol=0, atol=0)


def test_stochastic_rounding_bf16_reproducibility():
    torch.manual_seed(42)
    x = torch.randn(65537)
    ref = _rounding(x=x, seed=42, block_size=256)
    repeat = _rounding(x=x, seed=42, block_size=256)
    retiled = _rounding(x=x, seed=42, block_size=1024)
    torch.testing.assert_close(ref.view(torch.int16), repeat.view(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(ref.view(torch.int16), retiled.view(torch.int16), rtol=0, atol=0)
