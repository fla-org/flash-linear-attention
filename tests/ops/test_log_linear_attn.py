import os
from typing import List

import numpy as np
import pytest
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
    level_scales = torch.randn(B, T, H, L, dtype=dtype, device=device)
    v = (x * dt.unsqueeze(-1)).to(dtype=dtype)
    g = a * dt

    out, _ = chunk_log_linear_attn(q, k, v, g, level_scales)

    ref = naive_log_linear_attn(q, k, v, g, level_scales)

    assert_close("o", ref, out, 0.004)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [(2, 512, 8, 64, torch.float32), (2, 1024, 8, 128, torch.float32)]
    ],
)
@pytest.mark.skipif(device_platform == "intel", reason="Intel Triton Failure")
def test_chunk_bwd(
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
    level_scales = torch.randn(B, T, H, L, dtype=dtype, device=device)
    v = (x * dt.unsqueeze(-1)).to(dtype=dtype)
    g = a * dt
    do = torch.randn_like(v)
    q, k, v, g, level_scales = map(lambda x: x.to(device).requires_grad_(), (q, k, v, g, level_scales))

    out, _ = chunk_log_linear_attn(q, k, v, g, level_scales)
    (out * do).sum().backward()
    tri_dq, tri_dk, tri_dv, tri_dg, tri_dl = q.grad, k.grad, v.grad, g.grad, level_scales.grad
    q.grad = k.grad = v.grad = g.grad = level_scales.grad = None

    ref = naive_log_linear_attn(q, k, v, g, level_scales)
    (ref * do).sum().backward()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dl = q.grad, k.grad, v.grad, g.grad, level_scales.grad

    assert_close("o", ref, out, 0.004)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("dl", ref_dl, tri_dl, 0.015)


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
    level_scales = torch.randn(1, T, H, L, dtype=dtype, device=device)
    v = (x * dt.unsqueeze(-1)).to(dtype=dtype)
    g = a * dt

    out, _ = chunk_log_linear_attn(q, k, v, g, level_scales, cu_seqlens=cu_seqlens)

    o = []
    for i in range(cu_seqlens.shape[0] - 1):
        bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
        v_s = v[:, bos:eos]
        g_s = g[:, bos:eos]
        k_s = k[:, bos:eos]
        q_s = q[:, bos:eos]
        level_scales_s = level_scales[:, bos:eos]

        o.append(naive_log_linear_attn(q_s, k_s, v_s, g_s, level_scales_s))
    ref = torch.cat(o, dim=1)

    assert_close("o", ref, out, 0.004)
