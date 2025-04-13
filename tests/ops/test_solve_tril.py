# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.ops.utils.testing import COMPILER_MODE, assert_close
from fla.utils import device, device_platform

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_t_varlen_list = [[0, 64, 128, 256, 512]]
else:
    test_b_list = [2]
    test_t_list = [128, 200, 300, 500]
    test_t_varlen_list = [[0, 63, 286, 300, 512], [0, 127, 246, 521, 1000], [0, 255, 492, 1042, 2000]]
test_h_list = [2]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('chunk_size', [16, 32, 64])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Pytorch Failure'
)
def test_solve_tril(B, T, H, chunk_size):
    # do not randomly intiialize A otherwise the inverse is not stable
    k = F.normalize(torch.randn((B, H, T, 64), dtype=torch.float32, device=device), dim=-1)
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, 64)
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)
    Ai = solve_tril(A.reshape(B, H, -1, chunk_size)[:, :, :T, :].transpose(1, 2)).transpose(1, 2)

    Ai_ref = torch.inverse(A + torch.eye(A.shape[-1], device=A.device)[None, None, None, ...])
    Ai_ref = Ai_ref.reshape(B, H, -1, chunk_size)[:, :, :T, :]
    assert_close('solve_tril', Ai, Ai_ref, 0.0001)


@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('cu_seqlens', test_t_varlen_list)
@pytest.mark.parametrize('chunk_size', [64, 32, 16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Pytorch Failure'
)
def test_solve_tril_varlen(H, cu_seqlens, chunk_size):
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    # Construct the input. otherwise inverse's condition number might be too large to measure the error
    k = F.normalize(torch.randn((1, T, H, 64), dtype=torch.bfloat16, device=device), dim=-1)
    beta = torch.randn((1, T, H), dtype=torch.bfloat16, device=device).sigmoid()
    A, _ = chunk_scaled_dot_kkt_fwd(k, beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    Ai = solve_tril(A, cu_seqlens=cu_seqlens)

    Ai_ref = torch.zeros_like(Ai)
    for i in range(len(cu_seqlens) - 1):
        for j in range(cu_seqlens[i], cu_seqlens[i+1], chunk_size):
            actual_size = min(chunk_size, cu_seqlens[i+1] - j)
            Ai_ref[:, j:j+actual_size, :, :actual_size] = torch.inverse(
                A[:, j:j+actual_size, :, :actual_size].transpose(1, 2) +
                torch.eye(actual_size, device=A.device, dtype=A.dtype)[None, None, ...]
            ).transpose(1, 2)
    assert_close('solve_tril_varlen', Ai, Ai_ref, 0.0001)
