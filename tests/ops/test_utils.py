# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum
from fla.utils import device


def reversed_cumsum(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    c = x.cumsum(dim)
    y = x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
    return y.to(dtype)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [3, 15, 500, 1024])
@pytest.mark.parametrize("D", [50, 64, 100, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
def test_global_cumsum(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    head_first: bool
):
    torch.manual_seed(42)
    s = torch.randn(B, H, T, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, dtype=dtype).to(device)
    ref = s.float().cumsum(2 if head_first else 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, head_first=head_first)
    torch.testing.assert_close(ref, tri)

    s = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = s.float().cumsum(2 if head_first else 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, head_first=head_first)
    torch.testing.assert_close(ref, tri)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [15, 64, 500, 1024])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [50, 64, 100, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_global_cumsum_varlen(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([s[:, start:end].float().cumsum(1) for start, end in zip(offsets[:-1], offsets[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([s[:, start:end].float().cumsum(1) for start, end in zip(offsets[:-1], offsets[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [3, 15, 500, 1024])
@pytest.mark.parametrize("D", [50, 64, 100, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
def test_global_reversed_cumsum(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    head_first: bool
):
    torch.manual_seed(42)
    s = torch.randn(B, H, T, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(2 if head_first else 1)).to(dtype)
    tri = chunk_global_cumsum(s, dtype, reverse=True, head_first=head_first)
    torch.testing.assert_close(ref, tri)

    s = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(2 if head_first else 1)).to(dtype)
    tri = chunk_global_cumsum(s, dtype, reverse=True, head_first=head_first)
    torch.testing.assert_close(ref, tri)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [15, 64, 500, 1024])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [50, 64, 100, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_global_reversed_cumsum_varlen(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1) for start, end in zip(offsets[:-1], offsets[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, reverse=True, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1) for start, end in zip(offsets[:-1], offsets[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, reverse=True, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [150, 500, 1024])
@pytest.mark.parametrize("C", [32, 64])
@pytest.mark.parametrize("D", [50, 64, 100, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
def test_local_cumsum(
    B: int,
    H: int,
    T: int,
    C: int,
    D: int,
    dtype: torch.dtype,
    head_first: bool
):
    torch.manual_seed(42)
    s = torch.randn(B, H, T, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, dtype=dtype).to(device)
    if head_first:
        ref = torch.cat([s[:, :, i:i+C].float().cumsum(2) for i in range(0, T, C)], 2)
    else:
        ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, head_first=head_first)
    torch.testing.assert_close(ref, tri)

    s = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    if head_first:
        ref = torch.cat([s[:, :, i:i+C].float().cumsum(2) for i in range(0, T, C)], 2)
    else:
        ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, head_first=head_first)
    torch.testing.assert_close(ref, tri)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [15, 64, 500, 1024])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("C", [32, 64])
@pytest.mark.parametrize("D", [50, 64, 100, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_local_cumsum_varlen(
    B: int,
    T: int,
    H: int,
    C: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(offsets[:-1], offsets[1:])
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(offsets[:-1], offsets[1:])
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri)
