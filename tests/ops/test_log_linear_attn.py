import os

import pytest
import torch

from fla.ops.log_linear_attn import chunk_log_linear_attn
from fla.ops.log_linear_attn.naive import naive_log_linear_attn
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'L', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 1024, 4, 60, 10, torch.float32),
            (2, 1024, 8, 128, 10, torch.float32),
            (4, 2048, 8, 64, 11, torch.float32)
        ]
    ]
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure'
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    L: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dt = torch.nn.functional.softplus(
        torch.randn(B, T, H, dtype=torch.float32, device=device) - 4
    )
    a = -torch.exp(torch.rand(H, dtype=torch.float32, device=device))
    q = torch.randn(B, T, 1, D, dtype=dtype, device=device)
    k = torch.randn(B, T, 1, D, dtype=dtype, device=device)
    l = torch.randn(B, T, H, L, dtype=dtype, device=device)
    v = (x * dt.unsqueeze(-1)).to(dtype=dtype)
    g = a * dt

    out, _ = chunk_log_linear_attn(q, k, v, g, l)

    ref = naive_log_linear_attn(q, k, v, g, l)

    assert_close('o', ref, out, 0.004)

