# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import numpy as np
import pytest
import torch
import triton

from fla.ops.log_linear_attn import chunk_log_linear_attn
from fla.ops.log_linear_attn.chunk import chunkwise_bwd_kernel_dkg
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
        torch.randn(B, T, H, dtype=torch.float32, device=device) - 4,
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
        torch.randn(B, T, H, dtype=torch.float32, device=device) - 4,
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
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1].item()

    L = int(np.ceil(np.log2(T)) + 1)
    x = torch.randn(1, T, H, D, dtype=dtype, device=device)
    dt = torch.nn.functional.softplus(
        torch.randn(1, T, H, dtype=torch.float32, device=device) - 4,
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


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 70, 4, 64, torch.float32),
            (1, 200, 4, 64, torch.float32),
            (2, 320, 4, 64, torch.float32),  # divisible by the chunk size: control
        ]
    ],
)
@pytest.mark.skipif(device_platform == "intel", reason="Intel Triton Failure")
def test_chunk_partial_last_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    """The rows of a last chunk shorter than the chunk size must not read `level_scales`.

    `level_scales` is a view into a longer buffer whose tail is NaN, so any read past its end
    lands on a NaN and shows up in the gradients.
    """
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    L = int(np.ceil(np.log2(T))) + 1
    x = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dt = torch.nn.functional.softplus(
        torch.randn(B, T, H, dtype=torch.float32, device=device) - 4,
    )
    a = -torch.exp(torch.rand(H, dtype=torch.float32, device=device))
    q = torch.randn(B, T, 1, D, dtype=dtype, device=device)
    k = torch.randn(B, T, 1, D, dtype=dtype, device=device)
    v = (x * dt.unsqueeze(-1)).to(dtype=dtype)
    g = a * dt
    do = torch.randn_like(v)

    # level_scales sits at the front of a buffer whose tail holds NaN
    numel = B * T * H * L
    buffer = torch.full((numel + 64 * H * L,), float("nan"), dtype=dtype, device=device)
    level_scales = buffer[:numel].view(B, T, H, L)
    level_scales.copy_(torch.randn(B, T, H, L, dtype=dtype, device=device))

    q, k, v, g, level_scales = map(lambda x: x.requires_grad_(), (q, k, v, g, level_scales))

    out, _ = chunk_log_linear_attn(q, k, v, g, level_scales)
    (out * do).sum().backward()

    assert torch.isfinite(out).all(), "forward output is not finite"
    for name, tensor in zip(
        ("dq", "dk", "dv", "dg", "dl"), (q, k, v, g, level_scales),
    ):
        assert torch.isfinite(tensor.grad).all(), (
            f"{name} is not finite: {(~torch.isfinite(tensor.grad)).sum().item()} of "
            f"{tensor.grad.numel()} elements read past the end of level_scales"
        )


@pytest.mark.parametrize(
    ("T", "H", "D"),
    [
        pytest.param(*test, id="T{}-H{}-D{}".format(*test))
        for test in [(70, 4, 64), (200, 4, 64), (128, 4, 64)]
    ],
)
@pytest.mark.skipif(device_platform == "intel", reason="Intel Triton Failure")
def test_chunkwise_bwd_dkg_last_chunk_fold(T: int, H: int, D: int):
    """The chunk-scalar gate gradient must land on the chunk's last LIVE row.

    Driven directly because the wrapper hands the kernel a zero `dg_last` for the only chunk
    that can be partial, which hides the misplaced deposit.  `dh` is zeroed so the fold is the
    kernel's only contribution to `dg`.
    """
    torch.manual_seed(42)
    B, BT = 1, 64
    NT = triton.cdiv(T, BT)

    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.1
    dg = torch.zeros(B, T, H, dtype=torch.float32, device=device)
    dg_last = torch.zeros(B, NT, H, dtype=torch.float32, device=device)
    dg_last[:, NT - 1, :] = 1.0
    dh = torch.zeros(B, NT, H, D, D, dtype=torch.float32, device=device)
    k = torch.zeros(B, T, 1, D, dtype=torch.float32, device=device)
    v = torch.zeros(B, T, H, D, dtype=torch.float32, device=device)
    dk = torch.zeros(B, T, H, D, dtype=torch.float32, device=device)

    chunkwise_bwd_kernel_dkg[(NT, B * H)](
        dh=dh, k=k, v=v, g=g, dg_last=dg_last, dk=dk, dg=dg, cu_seqlens=None,
        T=T, H=H, K=D, V=D, L=int(np.ceil(np.log2(T))) + 1, BT=BT, NT=NT,
    )

    ref = torch.zeros_like(dg)
    ref[:, T - 1] = dg_last[:, NT - 1] * torch.exp(g[:, T - 1])
    assert_close("dg", ref, dg, 0.002)
