# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.utils import mean_pooling
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'C', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-C{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 16, 30, torch.float),
            (2, 500, 4, 32, 60, torch.float),
            (2, 1000, 5, 64, 128, torch.float),
            (3, 1024, 6, 64, 500, torch.float),
            (4, 2048, 8, 128, 1024, torch.float),
        ]
    ],
)
def test_mean_pooling(
    B: int,
    T: int,
    H: int,
    C: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, dtype=dtype).to(device)
    x.requires_grad = True
    ref = torch.cat([x[:, i:i+C, :].float().mean(1, True) for i in range(0, T, C)], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    assert_close('mean_pooling', ref, tri, 1e-3)
    assert_close('mean_pooling', ref_dx, tri_dx, 1e-3)


@pytest.mark.parametrize(
    ('H', 'C', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-C{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 32, 60, [0, 15], torch.float),
            (3, 64, 100, [0, 256, 500, 1000], torch.float),
            (4, 64, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 128, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 128, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_mean_pooling_varlen(
    H: int,
    C: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    x = torch.randn(1, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    ref = torch.cat([
        torch.cat([x[:, i:min(end, i+C), :].float().mean(1, True) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C, cu_seqlens=cu_seqlens)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)
    torch.testing.assert_close(ref_dx, tri_dx.to(ref_dx.dtype), rtol=1.6e-2, atol=3e-5)
