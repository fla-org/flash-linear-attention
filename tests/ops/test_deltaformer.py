# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.deltaformer import deltaformer_attn
from fla.ops.deltaformer.naive import naive_deltaformer_attn
from fla.utils import assert_close, device, is_intel_alchemist


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 128, 2, 64, torch.float16),
            # Test with bfloat16
            (1, 256, 4, 64, torch.bfloat16),
            (2, 512, 4, 64, torch.bfloat16),
            (4, 1024, 4, 128, torch.bfloat16)
        ]
    ]
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM."
)
def test_deltaformer_attn(
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

    q = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(True)
    beta = torch.randn((B, H, T), dtype=dtype, device=device).sigmoid().requires_grad_(True)

    do = torch.randn((B, H, T, D), dtype=dtype, device=device)

    ref = naive_deltaformer_attn(q, k, v, beta)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dbeta, beta.grad = beta.grad.clone(), None

    tri = deltaformer_attn(q, k, v, beta)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dbeta, beta.grad = beta.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 0.008)

    if B > 1:
        q_packed = torch.cat([q[i] for i in range(B)], dim=1).unsqueeze(0).detach().clone().requires_grad_(True)
        k_packed = torch.cat([k[i] for i in range(B)], dim=1).unsqueeze(0).detach().clone().requires_grad_(True)
        v_packed = torch.cat([v[i] for i in range(B)], dim=1).unsqueeze(0).detach().clone().requires_grad_(True)
        beta_packed = torch.cat([beta[i] for i in range(B)], dim=1).unsqueeze(0).detach().clone().requires_grad_(True)

        cu_seqlens = torch.arange(0, B*T + 1, T, dtype=torch.int32, device=device)

        grad_output_packed = torch.cat([do[i] for i in range(B)], dim=1).unsqueeze(0)

        tri_varlen = deltaformer_attn(q_packed, k_packed, v_packed, beta_packed, cu_seqlens=cu_seqlens)

        tri_varlen.backward(grad_output_packed)
        tri_varlen_dq = q_packed.grad
        tri_varlen_dk = k_packed.grad
        tri_varlen_dv = v_packed.grad
        tri_varlen_dbeta = beta_packed.grad

        tri_varlen_reshaped = torch.stack([tri_varlen[0, :, i*T:(i+1)*T, :] for i in range(B)], dim=0)
        tri_varlen_dq_reshaped = torch.stack([tri_varlen_dq[0, :, i*T:(i+1)*T, :] for i in range(B)], dim=0)
        tri_varlen_dk_reshaped = torch.stack([tri_varlen_dk[0, :, i*T:(i+1)*T, :] for i in range(B)], dim=0)
        tri_varlen_dv_reshaped = torch.stack([tri_varlen_dv[0, :, i*T:(i+1)*T, :] for i in range(B)], dim=0)
        tri_varlen_dbeta_reshaped = torch.stack([tri_varlen_dbeta[0, :, i*T:(i+1)*T] for i in range(B)], dim=0)

        assert_close("varlen_o", ref, tri_varlen_reshaped, 0.005)
        assert_close("varlen_dq", ref_dq, tri_varlen_dq_reshaped, 0.008)
        assert_close("varlen_dk", ref_dk, tri_varlen_dk_reshaped, 0.008)
        assert_close("varlen_dv", ref_dv, tri_varlen_dv_reshaped, 0.008)
        assert_close("varlen_dbeta", ref_dbeta, tri_varlen_dbeta_reshaped, 0.008)
