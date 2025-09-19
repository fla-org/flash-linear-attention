# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.deltaformer import delta_pre_attn
from fla.ops.deltaformer.naive import delta_pre_attn_naive
from fla.utils import assert_close, device, is_intel_alchemist


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 128, 2, 64, torch.float32),
            # Test with bfloat16
            (1, 256, 4, 64, torch.bfloat16),
            (2, 512, 4, 64, torch.bfloat16),
            (2, 1024, 4, 128, torch.bfloat16)
        ]
    ]
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM."
)
def test_delta_pre_attn(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype
):
    """
    Test DeltaFormer pre-attention by comparing fused implementation with naive reference.
    """
    torch.manual_seed(42)

    # Generate test inputs
    q = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(True)
    beta = torch.randn((B, H, T), dtype=dtype, device=device).sigmoid().requires_grad_(True)

    # Output gradient for backward pass testing
    do = torch.randn((B, H, T, D), dtype=dtype, device=device)

    # Test with beta parameter
    ref = delta_pre_attn_naive(q, k, v, beta)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dbeta, beta.grad = beta.grad.clone(), None

    # Test fused implementation
    tri = delta_pre_attn(q, k, v, beta)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dbeta, beta.grad = beta.grad.clone(), None

    # Compare outputs and gradients
    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 0.008)
