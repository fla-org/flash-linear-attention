
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-chunk_size{}".format(*test))
        for test in [
            (1, 63, 1, 16),
            (2, 500, 4, 32),
            (2, 1000, 5, 64),
            (3, 1024, 6, 64),
            (4, 2048, 8, 64),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Pytorch Failure',
)
def test_solve_tril(B, T, H, chunk_size):
    # do not randomly intiialize A otherwise the inverse is not stable
    k = F.normalize(torch.randn((B, H, T, 64), dtype=torch.float32, device=device), dim=-1)
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, 64)
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)

    ref = torch.inverse(A + torch.eye(A.shape[-1], device=A.device)[None, None, None, ...])
    ref = ref.reshape(B, H, -1, chunk_size)[:, :, :T, :]

    tri = solve_tril(A.reshape(B, H, -1, chunk_size)[:, :, :T, :].transpose(1, 2)).transpose(1, 2)

    assert_close('solve_tril', ref, tri, 0.0001)


@pytest.mark.parametrize(
    ('H', 'D', 'chunk_size', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-D{}-chunk_size{}-cu_seqlens{}".format(*test))
        for test in [
            (4, 64, 16, [0, 15]),
            (4, 64, 32, [0, 256, 500, 1000]),
            (4, 100, 64, [0, 15, 100, 300, 1200, 2000]),
            (4, 64, 16, [0, 1, 100, 300, 1200, 2048]),
            (4, 128, 32, [0, 200, 512, 1200, 2048]),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Pytorch Failure',
)
def test_solve_tril_varlen(
    H: int,
    D: int,
    chunk_size: int,
    cu_seqlens: list[int],
):
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    # Construct the input. otherwise inverse's condition number might be too large to measure the error
    k = F.normalize(torch.randn((1, T, H, D), dtype=torch.bfloat16, device=device), dim=-1)
    beta = torch.randn((1, T, H), dtype=torch.bfloat16, device=device).sigmoid()
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)

    ref = torch.zeros_like(A)
    for i in range(len(cu_seqlens) - 1):
        for j in range(cu_seqlens[i], cu_seqlens[i+1], chunk_size):
            actual_size = min(chunk_size, cu_seqlens[i+1] - j)
            ref[:, j:j+actual_size, :, :actual_size] = torch.inverse(
                A[:, j:j+actual_size, :, :actual_size].transpose(1, 2) +
                torch.eye(actual_size, device=A.device, dtype=A.dtype)[None, None, ...],
            ).transpose(1, 2)

    tri = solve_tril(A, cu_seqlens=cu_seqlens)
    assert_close('solve_tril_varlen', ref, tri, 0.0001)
