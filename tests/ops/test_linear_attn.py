

import pytest
import torch

from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.linear_attn.naive import naive_recurrent_linear_attn
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 64, 1, 64, None, torch.float),
            (2, 512, 4, 60, None, torch.float),
            (3, 1024, 8, 128, 1., torch.float),
            (3, 1024, 8, 128, 0.1, torch.float),
            (3, 1024, 8, 128, None, torch.float),
            (2, 2048, 8, 256, None, torch.float16),
            (2, 2048, 4, 256, None, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=True, normalize=False)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=True, normalize=False)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)
    assert_close('dq', ref_dq, tri_dq, 0.001)
    assert_close('dk', ref_dk, tri_dk, 0.001)
    assert_close('dv', ref_dv, tri_dv, 0.001)
    assert_close('dh0', ref_dh0, tri_dh0, 0.001)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float16),
            (2, 500, 3, 60, torch.float16),
            (2, 1000, 3, 128, torch.float16),
            (3, 1000, 4, 64, torch.float16),
            (2, 2048, 4, 256, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = fused_recurrent_linear_attn(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_linear_attn(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)
    assert_close('dq', ref_dq, tri_dq, 0.001)
    assert_close('dk', ref_dk, tri_dk, 0.001)
    assert_close('dv', ref_dv, tri_dv, 0.001)
    assert_close('dh0', ref_dh0, tri_dh0, 0.001)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float16),
            (2, 500, 3, 60, torch.float16),
            (2, 1000, 3, 128, torch.float16),
            (3, 1000, 4, 64, torch.float16),
            (2, 2048, 4, 256, torch.float16),
        ]
    ],
)
def test_fused_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = fused_recurrent_linear_attn(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_chunk_linear_attn(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        normalize=False,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.001)
    assert_close('ht', ref_ht, tri_ht, 0.001)
    assert_close('dq', ref_dq, tri_dq, 0.001)
    assert_close('dk', ref_dk, tri_dk, 0.001)
    assert_close('dv', ref_dv, tri_dv, 0.001)
    assert_close('dh0', ref_dh0, tri_dh0, 0.001)
