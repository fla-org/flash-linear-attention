# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum
from fla.utils import assert_close, device


def reversed_cumsum(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    c = x.cumsum(dim)
    y = x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
    return y.to(dtype)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 30, torch.float),
            (2, 500, 4, 60, torch.float),
            (2, 1000, 5, 128, torch.float),
            (3, 1024, 6, 500, torch.float),
            (4, 2048, 8, 1024, torch.float),
        ]
    ],
)
def test_global_cumsum(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    s = torch.randn(B, T, H, dtype=dtype).to(device)
    ref = s.float().cumsum(1).to(dtype)
    tri = chunk_global_cumsum(s)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = s.float().cumsum(1).to(dtype)
    tri = chunk_global_cumsum(s)
    assert_close('global_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 60, [0, 15], torch.float),
            (3, 100, [0, 256, 500, 1000], torch.float),
            (4, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_global_cumsum_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        s[:, start:end].float().cumsum(1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
    ], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        s[:, start:end].float().cumsum(1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
    ], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    assert_close('global_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 30, torch.float),
            (2, 500, 4, 60, torch.float),
            (2, 1000, 5, 128, torch.float),
            (3, 1024, 6, 500, torch.float),
            (4, 2048, 8, 1024, torch.float),
        ]
    ],
)
def test_global_reversed_cumsum(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    s = torch.randn(B, T, H, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(1)).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(1)).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True)
    assert_close('global_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 60, [0, 15], torch.float),
            (3, 100, [0, 256, 500, 1000], torch.float),
            (4, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_global_reversed_cumsum_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1)
                    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    assert_close('global_reversed_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1)
                    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    assert_close('global_reversed_cumsum', ref, tri, 1e-3)


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
def test_local_cumsum(
    B: int,
    T: int,
    H: int,
    C: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    s = torch.randn(B, T, H, dtype=dtype).to(device)
    ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C)
    assert_close('local_cumsum', ref, tri, 1e-3)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C)
    assert_close('local_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('H', 'C', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-C{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 16, 60, [0, 15], torch.float),
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
def test_local_cumsum_varlen(
    H: int,
    C: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    assert_close('local_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    assert_close('local_cumsum', ref, tri, 1e-3)
