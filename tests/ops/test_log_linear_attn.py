import os
from typing import List

import pytest
import numpy as np
import torch

from fla.ops.log_linear_attn import chunk_log_linear_attn
from fla.ops.log_linear_attn.naive import naive_log_linear_attn
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [(2, 1024, 8, 128, torch.float32), (4, 2048, 8, 64, torch.float32)]
    ],
)
@pytest.mark.skipif(device_platform == "intel", reason="Intel Triton Failure")
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    L = int(np.log2(T) + 1)
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
    assert_close("o", ref, out, 0.004)


@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, [0, 15], torch.float32),
            (4, 64, [0, 256, 500, 1000], torch.float32),
            (4, 128, [0, 15, 100, 300, 1200, 2000], torch.float32),
        ]
    ],
)
@pytest.mark.skipif(device_platform == "intel", reason="Intel Triton Failure")
def test_chunk_varlen(
    H: int,
    D: int,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1].item()

    L = int(np.ceil(np.log2(T)) + 1)
    x = torch.randn(1, T, H, D, dtype=dtype, device=device)
    dt = torch.nn.functional.softplus(
        torch.randn(1, T, H, dtype=torch.float32, device=device) - 4
    )
    a = -torch.exp(torch.rand(H, dtype=torch.float32, device=device))
    q = torch.randn(1, T, 1, D, dtype=dtype, device=device)
    k = torch.randn(1, T, 1, D, dtype=dtype, device=device)
    l = torch.randn(1, T, H, L, dtype=dtype, device=device)
    v = (x * dt.unsqueeze(-1)).to(dtype=dtype)
    g = a * dt

    out, _ = chunk_log_linear_attn(q, k, v, g, l, cu_seqlens=cu_seqlens)

    o = []
    for i in range(cu_seqlens.shape[0] - 1):
        bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
        v_s = v[:, bos:eos]
        g_s = g[:, bos:eos]
        k_s = k[:, bos:eos]
        q_s = q[:, bos:eos]
        l_s = l[:, bos:eos]

        o.append(naive_log_linear_attn(q_s, k_s, v_s, g_s, l_s))
    ref = torch.cat(o, dim=1)

    assert_close("o", ref, out, 0.004)
