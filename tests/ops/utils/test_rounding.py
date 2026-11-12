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
    [pytest.param(T, D, id=f'T{T}-D{D}') for T, D in [(33, 31), (65537, 1)]],
)
def test_fp32_to_bf16_sr(T: int, D: int):
    torch.manual_seed(42)
    x = torch.randn(T, D)
    values = torch.tensor([0., -0., 1., -1., float('inf'), -float('inf'), 2**-149, 2**-126, torch.finfo(torch.float32).max])
    x.view(-1)[:values.numel()] = values
    # a NaN whose payload would be lost by truncation.
    x.view(torch.int32).view(-1)[values.numel()] = 0x7F800001
    rand = torch.randint(-(2**31), 2**31, (T, D), dtype=torch.int32)
    rand.view(-1)[:values.numel()] = 0
    x, rand = x.t(), rand.t()

    # reference: https://github.com/pytorch/ao/blob/a724e78/torchao/optim/quant_utils.py#L121
    bits = x.view(torch.int32)
    truncated = bits & 0xFFFF0000
    ref = torch.where((rand & 0xFFFF) < (bits & 0xFFFF), truncated + 0x10000, truncated)
    ref = ref.view(torch.float32).bfloat16()
    tri = fp32_to_bf16_sr(x=x.to(device), rand=rand.to(device)).cpu()
    is_nan = x.isnan()
    torch.testing.assert_close(tri[~is_nan].view(torch.int16), ref[~is_nan].view(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(tri.isnan(), is_nan, rtol=0, atol=0)
    assert ((tri[is_nan].view(torch.int16) & 0x40) != 0).all()


def test_fp32_to_bf16_sr_rng():
    torch.manual_seed(42)
    x = torch.full((65536,), 1.001953125, device=device)
    ref = fp32_to_bf16_sr(x=x)
    torch.manual_seed(42)
    repeat = fp32_to_bf16_sr(x=x)
    torch.testing.assert_close(ref.view(torch.int16), repeat.view(torch.int16), rtol=0, atol=0)
    # a six-sigma binomial bound checks the internally generated rounding probabilities.
    assert abs((ref == 1.0078125).sum().item() - 16384) < 6 * (65536 * 0.25 * 0.75)**0.5
