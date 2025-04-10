# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum, mean_pooling
from fla.ops.utils.testing import COMPILER_MODE
from fla.utils import device

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [64, 32, 100, 256]
test_h_list = [2]


def reversed_cumsum(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    c = x.cumsum(dim)
    y = x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
    return y.to(dtype)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
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
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = s.float().cumsum(1).to(dtype)
    tri = chunk_global_cumsum(s)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_global_cumsum_varlen(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([s[:, start:end].float().cumsum(1) for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([s[:, start:end].float().cumsum(1) for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
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
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(1)).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_global_reversed_cumsum_varlen(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1) for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1) for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('C', [32, 64])
@pytest.mark.parametrize('dtype', [torch.float16])
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
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_varlen_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('C', [32, 64])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
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
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('C', [32, 64])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
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

    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)
    torch.testing.assert_close(ref_dx, tri_dx.to(ref_dx.dtype), rtol=1.6e-2, atol=3e-5)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('C', [32, 64])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
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
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:B-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    ref = torch.cat([
        torch.cat([x[:, i:min(end, i+C), :].float().mean(1, True) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C, cu_seqlens=cu_seqlens)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)
    torch.testing.assert_close(ref_dx, tri_dx.to(ref_dx.dtype), rtol=1.6e-2, atol=3e-5)
