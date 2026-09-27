# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2_norm
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (1, 65, 2, 512, torch.float16),
            (1, 65, 2, 513, torch.bfloat16),
            (1, 33, 2, 513, torch.float32),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
            (4, 1024, 5, 1024, torch.float16),
            (4, 1024, 5, 1024, torch.bfloat16),
            (5, 1024, 6, 2048, torch.float16),
            (5, 1024, 6, 2048, torch.bfloat16),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2_norm(x)
    ref_dx = torch.autograd.grad(ref.sum(), x)[0]
    tri_dx = torch.autograd.grad(tri.sum(), x)[0]

    assert_close('y', ref, tri, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)


@pytest.mark.parametrize('D', [128, 513], ids=['tiled', 'row'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
def test_l2norm_output_dtype(D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(2, 63, D, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(x.shape, device=device)
    tri = l2_norm(x=x, output_dtype=torch.float32)
    ref = x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
    tri_dx = torch.autograd.grad(tri, x, do)[0]
    ref_dx = torch.autograd.grad(ref, x, do)[0]

    assert tri.dtype == torch.float32
    assert tri_dx.dtype == dtype
    assert_close('y', ref, tri, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)
