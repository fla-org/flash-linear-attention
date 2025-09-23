# -*- coding: utf-8 -*-

import math

import pytest
import torch

from fla.ops import sb_attn, sb_attn_naive
from fla.utils import assert_close, device, is_intel_alchemist


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 64, 2, 64, torch.float32),
            (1, 128, 4, 64, torch.bfloat16),
        ]
    ]
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM."
)
def test_stickbreaking_attn(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)

    do = torch.randn((B, T, H, D), dtype=dtype, device=device)
    dr = torch.randn((B, T, H), dtype=dtype, device=device)

    inv_temp = 1.0 / math.sqrt(D)

    # Reference (naive)
    ref_o, ref_rem = sb_attn_naive(q, k, v, inv_temp, attend_current=False)
    (ref_o * do).sum().backward(retain_graph=True)
    (ref_rem * dr).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # Triton fused
    tri_o, tri_rem = sb_attn(q, k, v, inv_temp=inv_temp, attend_current=False)
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
