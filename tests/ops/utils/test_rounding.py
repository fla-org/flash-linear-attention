# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils import fp32_to_bf16_sr
from fla.utils import device


@pytest.mark.parametrize(
    ('T', 'D'),
    [pytest.param(T, D, id=f'T{T}-D{D}') for T, D in [(0, 1), (1, 1), (33, 31), (65537, 1)]],
)
def test_fp32_to_bf16_sr(T: int, D: int):
    torch.manual_seed(42)
    bits = torch.randint(0, 0x7F7F0001, (T, D), dtype=torch.int64)
    bits |= torch.randint(0, 2, (T, D), dtype=torch.int64) << 31
    x = bits.to(torch.int32).view(torch.float32).t()
    rand = torch.randint(-(2**31), 2**31, (T, D), dtype=torch.int32).t()

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
    tri = fp32_to_bf16_sr(x=x.to(device), rand=rand.to(device)).cpu()
    torch.testing.assert_close(tri.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)


def test_fp32_to_bf16_sr_exact():
    bits = torch.arange(65536, dtype=torch.int32)
    bits = bits[(bits & 0x7FFF) <= 0x7F80]
    x = (bits << 16).view(torch.float32)
    rand = torch.full_like(bits, 0xFFFF)
    tri = fp32_to_bf16_sr(x=x.to(device), rand=rand.to(device)).cpu()
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
def test_fp32_to_bf16_sr_probability(bits: int, fraction: float):
    x = torch.full((65536,), bits, dtype=torch.int64).to(torch.int32).view(torch.float32)
    rand = torch.arange(65536, dtype=torch.int32)
    tri = fp32_to_bf16_sr(x=x.to(device), rand=rand.to(device)).cpu()
    away_zero = (tri.view(torch.int16).to(torch.int32) & 0xFFFF) == (bits >> 16) + 1
    assert away_zero.sum().item() == int(fraction * 65536)
    torch.testing.assert_close(tri.double().mean(), x[0].double(), rtol=0, atol=0)


def test_fp32_to_bf16_sr_special_values():
    bits = torch.tensor(
        [0x7F800001, 0xFF800001, 0x7FC12345, 0xFFFFFFFF, 0x7F7FFFFF, 0xFF7FFFFF],
        dtype=torch.int64,
    )
    x = bits.to(torch.int32).view(torch.float32)
    rand = torch.zeros(x.shape, dtype=torch.int32)
    tri = fp32_to_bf16_sr(x=x.to(device), rand=rand.to(device)).cpu()
    assert torch.isnan(tri[:4]).all()
    assert ((tri[:4].view(torch.int16).to(torch.int32) & 0x40) != 0).all()
    torch.testing.assert_close(torch.signbit(tri), torch.signbit(x), rtol=0, atol=0)
    torch.testing.assert_close(tri[4:].float(), torch.tensor([float('inf'), -float('inf')]), rtol=0, atol=0)


def test_fp32_to_bf16_sr_reproducibility():
    torch.manual_seed(42)
    x = torch.full((65536,), 1.001953125, device=device)
    ref = fp32_to_bf16_sr(x=x)
    torch.manual_seed(42)
    repeat = fp32_to_bf16_sr(x=x)
    torch.testing.assert_close(ref.view(torch.int16), repeat.view(torch.int16), rtol=0, atol=0)
    # a six-sigma binomial bound checks the internally generated rounding probabilities.
    assert abs((ref == 1.0078125).sum().item() - 16384) < 6 * (65536 * 0.25 * 0.75)**0.5
