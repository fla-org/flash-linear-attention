import math

import pytest
import torch

from fla.ops.stickbreaking_attn import naive_stickbreaking_attn, parallel_stickbreaking_attn
from fla.utils import assert_close, device, is_intel_alchemist


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 128, 2, 64, torch.float16),
            (1, 256, 4, 64, torch.float16),
            (2, 512, 4, 64, torch.float16),
            (4, 1024, 4, 128, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_stickbreaking_attn(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)

    do = torch.randn((B, T, H, D), dtype=dtype, device=device)
    dr = torch.randn((B, T, H), dtype=dtype, device=device)

    scale = 1.0 / math.sqrt(D)

    # Reference (naive)
    ref_o, ref_rem = naive_stickbreaking_attn(q, k, v, scale)
    (ref_o * do).sum().backward(retain_graph=True)
    (ref_rem * dr).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # Triton fused
    tri_o, tri_rem = parallel_stickbreaking_attn(q, k, v, scale=scale)
    (tri_o * do).sum().backward(retain_graph=True)
    (tri_rem * dr).sum().backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    # Compare
    assert_close(" o", ref_o, tri_o, 0.008)
    assert_close("rem", ref_rem, tri_rem, 0.02)
    assert_close("dq", ref_dq, tri_dq, 0.02)
    assert_close("dk", ref_dk, tri_dk, 0.02)
    assert_close("dv", ref_dv, tri_dv, 0.02)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, [0, 63], torch.float16),
            (4, 64, [0, 256, 500, 1000], torch.float16),
            (4, 128, [0, 15, 100, 300, 1200, 2000], torch.float16),
            (2, 128, [0, 100, 123, 300, 500, 800, 1000, 1500, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_stickbreaking_attn_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    num_chunks = len(cu_seqlens) - 1
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)

    do = torch.randn_like(q)
    dr = torch.randn((1, T, H), dtype=dtype, device=device)

    scale = 1.0 / math.sqrt(D)

    ref_os = []
    ref_rems = []
    for idx in range(num_chunks):
        start, end = cu_seqlens[idx], cu_seqlens[idx + 1]
        ref_o_chunk, ref_rem_chunk = naive_stickbreaking_attn(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            scale,
        )
        ref_os.append(ref_o_chunk)
        ref_rems.append(ref_rem_chunk)

    ref_o = torch.cat(ref_os, dim=1)
    ref_rem = torch.cat(ref_rems, dim=1)

    (ref_o * do).sum().backward(retain_graph=True)
    (ref_rem * dr).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri_o, tri_rem = parallel_stickbreaking_attn(q, k, v, scale=scale, cu_seqlens=cu_seqlens_tensor)
    (tri_o * do).sum().backward(retain_graph=True)
    (tri_rem * dr).sum().backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close("o", ref_o, tri_o, 0.008)
    assert_close("rem", ref_rem, tri_rem, 0.02)
    assert_close("dq", ref_dq, tri_dq, 0.02)
    assert_close("dk", ref_dk, tri_dk, 0.02)
    assert_close("dv", ref_dv, tri_dv, 0.02)
