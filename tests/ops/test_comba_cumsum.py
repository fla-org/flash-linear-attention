# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.comba.cumsum import chunk_comba_cumsum_scalar_fwd, chunk_comba_cumsum_scalar_bwd
from fla.utils import COMPILER_MODE, assert_close, device, device_platform

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


def cumsum_comba_local_fwd_reference(s, reverse=False, chunk_size=128):
    o_0 = torch.zeros_like(s)
    o_1 = torch.zeros_like(s)
    T = s.size(1)
    fn = torch.cumsum
    for i in range(0, T, chunk_size):
        s_chunk = s[:, i:i+chunk_size]
        o_1[:, i:i+chunk_size] = fn(s_chunk.float(), dim=1).to(o_1)
        o_0[:, i:i+chunk_size] = o_1[:, i:i+chunk_size] - s_chunk

    return o_0, o_1



@pytest.mark.parametrize("B", [32])
@pytest.mark.parametrize("T", [256, 1024, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("dtype", [torch.float, torch.float16])
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_cumsum_local_scalar_fwd(B, T, H, dtype, chunk_size):
    s = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    ref_0, ref_1 = cumsum_comba_local_fwd_reference(s, chunk_size=chunk_size)
    tri_0, tri_1 = chunk_comba_cumsum_scalar_fwd(s, chunk_size=chunk_size)
    assert_close("local cumsum scalar", ref_0, tri_0, 0.001 if dtype == torch.float else 0.003)
    assert_close("local cumsum scalar", ref_1, tri_1, 0.001 if dtype == torch.float else 0.003)
