# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.utils.cumsum import chunk_global_cumsum, chunk_local_cumsum
from fla.ops.utils.testing import COMPILER_MODE, assert_close
from fla.utils import device, device_platform

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


def rev_cumsum(s, dim=-1):
    return torch.flip(torch.cumsum(torch.flip(s, dims=[dim]), dim), dims=[dim])


def cumsum_local_reference(
    s: torch.Tensor,
    reverse: bool = False,
    chunk_size: int = 128
):
    o = torch.zeros_like(s)
    T = s.size(1)
    fn = torch.cumsum if not reverse else rev_cumsum
    for i in range(0, T, chunk_size):
        s_chunk = s[:, i:i+chunk_size]
        o[:, i:i+chunk_size] = fn(s_chunk.float(), dim=1).to(o)

    return o


def cumsum_global_reference(
    s: torch.Tensor,
    reverse: bool = False,
):
    fn = torch.cumsum if not reverse else rev_cumsum
    return fn(s.float(), dim=1).to(s)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_cumsum_local_vector(
        B: int,
        T: int,
        H: int,
        D: int,
        dtype: torch.dtype,
        reverse: bool,
        chunk_size: int
):
    s = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    ref = cumsum_local_reference(s, reverse=reverse, chunk_size=chunk_size)
    tri = chunk_local_cumsum(s, reverse=reverse, chunk_size=chunk_size)
    assert_close("local cumsum vector", ref, tri, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_cumsum_local_scalar(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype,
    reverse: bool,
    chunk_size: int
):
    s = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    ref = cumsum_local_reference(s, reverse=reverse, chunk_size=chunk_size)
    tri = chunk_local_cumsum(s, reverse=reverse, chunk_size=chunk_size)
    assert_close("local cumsum scalar", ref, tri, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure"
)
def test_cumsum_global_vector(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    reverse: bool,
):
    s = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    ref = cumsum_global_reference(s, reverse=reverse)
    tri = chunk_global_cumsum(s, reverse=reverse)
    assert_close("global cumsum vector", ref, tri, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_cumsum_global_scalar(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype,
    reverse: bool,
):
    s = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    ref = cumsum_global_reference(s, reverse=reverse)
    tri = chunk_global_cumsum(s, reverse=reverse)
    assert_close("global cumsum scalar", ref, tri, 0.001 if dtype == torch.float else 0.003)
