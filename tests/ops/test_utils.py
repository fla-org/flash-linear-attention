# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum, mean_pooling
from fla.utils import device

compiled_mode = os.getenv("COMPILER_MODE") == "1"
if compiled_mode:
    test_b_list = [1]
    test_t_list = [64]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
else:
    test_b_list = [2]
    test_t_list = [1, 7, 15, 63, 286, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [32, 64, 100, 256]
test_h_list = [2]


def reversed_cumsum(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    c = x.cumsum(dim)
    y = x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
    return y.to(dtype)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
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
    torch.testing.assert_close(ref, tri.to(ref.dtype))

    s = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = s.float().cumsum(2 if head_first else 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, head_first=head_first)
    torch.testing.assert_close(ref, tri.to(ref.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
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
    torch.testing.assert_close(ref, tri.to(ref.dtype))

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([s[:, start:end].float().cumsum(1) for start, end in zip(offsets[:-1], offsets[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri.to(ref.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
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
    torch.testing.assert_close(ref, tri.to(ref.dtype))

    s = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(2 if head_first else 1)).to(dtype)
    tri = chunk_global_cumsum(s, dtype, reverse=True, head_first=head_first)
    torch.testing.assert_close(ref, tri.to(ref.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
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
    torch.testing.assert_close(ref, tri.to(ref.dtype))

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1) for start, end in zip(offsets[:-1], offsets[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, dtype, reverse=True, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri.to(ref.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("C", [32, 64])
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
    torch.testing.assert_close(ref, tri.to(ref.dtype))

    s = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    if head_first:
        ref = torch.cat([s[:, :, i:i+C].float().cumsum(2) for i in range(0, T, C)], 2)
    else:
        ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, head_first=head_first)
    torch.testing.assert_close(ref, tri.to(ref.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("C", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
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
    torch.testing.assert_close(ref, tri.to(ref.dtype))

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(offsets[:-1], offsets[1:])
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, offsets=offsets, head_first=False)
    torch.testing.assert_close(ref, tri.to(ref.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("C", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_mean_pooling(
    B: int,
    H: int,
    T: int,
    C: int,
    D: int,
    dtype: torch.dtype,
    head_first: bool
):
    torch.manual_seed(42)
    x = torch.randn(B, H, T, D, dtype=dtype).to(device) if head_first else torch.randn(B, T, H, D, dtype=dtype).to(device)
    x.requires_grad = True
    if head_first:
        ref = torch.cat([x[:, :, i:i+C].float().mean(2, True) for i in range(0, T, C)], 2).to(dtype)
    else:
        ref = torch.cat([x[:, i:i+C, :].float().mean(1, True) for i in range(0, T, C)], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C, head_first=head_first)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    torch.testing.assert_close(ref, tri.to(ref.dtype))
    torch.testing.assert_close(ref_dx, tri_dx.to(ref_dx.dtype))


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("C", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
def test_mean_pooling_varlen(
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

    x = torch.randn(1, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    ref = torch.cat([
        torch.cat([x[:, i:min(end, i+C), :].float().mean(1, True) for i in range(start, end, C)], 1)
        for start, end in zip(offsets[:-1], offsets[1:])
    ], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C, cu_seqlens=offsets, head_first=False)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    torch.testing.assert_close(ref, tri.to(ref.dtype))
    torch.testing.assert_close(ref_dx, tri_dx.to(ref_dx.dtype))
