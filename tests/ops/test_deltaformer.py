# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch

from fla.ops.deltaformer import deltaformer_attn
from fla.ops.deltaformer.naive import naive_deltaformer_attn
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


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
    IS_INTEL_ALCHEMIST,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_deltaformer_attn(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    """
    Test DeltaFormer pre-attention by comparing fused implementation with naive reference.
    """
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    beta = torch.randn((B, T, H), dtype=dtype, device=device).sigmoid().requires_grad_(True)

    do = torch.randn((B, T, H, D), dtype=dtype, device=device)

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

    assert_close('o', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.008)


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
    IS_INTEL_ALCHEMIST,
    reason="Skipping test on Intel Alchemist due to known issues with SRAM.",
)
def test_deltaformer_attn_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    beta = torch.randn((1, T, H), dtype=dtype, device=device).sigmoid().requires_grad_()

    do = torch.randn_like(q)

    refs = []
    for i in range(N):
        ref = naive_deltaformer_attn(
            q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
        )
        refs.append(ref)
    ref = torch.cat(refs, dim=1)

    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dbeta, beta.grad = beta.grad.clone(), None

    tri = deltaformer_attn(q, k, v, beta, cu_seqlens=cu_seqlens)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dbeta, beta.grad = beta.grad.clone(), None

    assert_close('o', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.008)
