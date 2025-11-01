
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6
from fla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6
from fla.utils import assert_close, device, device_platform


@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 15, 2, 60, 1.0, torch.float16),
            (3, 60, 3, 64, 0.1, torch.float16),
            (3, 64, 2, 64, 1, torch.float16),
            (4, 500, 3, 256, 1, torch.float16),
            (4, 1000, 4, 64, 10, torch.float16),
            (4, 2048, 4, 64, 1, torch.float16),
            (4, 2048, 4, 256, 1, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    w = F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device=device)) / gate_logit_normalizer

    u = torch.randn(H, D, dtype=dtype, device=device).requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=dtype, device=device).requires_grad_()
    w = w.requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ref, _ = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=False,
    )

    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    # triton implementation
    tri, tri_ht = chunk_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((tri * do).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dw', ref_dw, tri_dw, 0.005)
    assert_close('du', ref_du, tri_du, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, [0, 15], torch.float16),
            (4, 64, [0, 256, 500, 1000], torch.float16),
            (4, 100, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_chunk_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    w = F.logsigmoid(torch.randn((1, T, H, D), dtype=dtype, device=device)).requires_grad_(True)
    u = torch.randn(H, D, dtype=dtype, device=device).requires_grad_(True)
    h0 = torch.randn((N, H, D, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ref, _ = fused_recurrent_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_rwkv6(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        u.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None
    assert_close('o', ref, tri, 0.004)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dw', ref_dw, tri_dw, 0.005)
    assert_close('du', ref_du, tri_du, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)
