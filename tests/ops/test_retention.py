
import os

import pytest
import torch

from fla.ops.retention import chunk_retention, fused_chunk_retention, fused_recurrent_retention, parallel_retention
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'expand_ratio', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-expand_ratio{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float16),
            (2, 500, 3, 60, 1, torch.float16),
            (2, 1000, 3, 100, 1, torch.float16),
            (2, 1000, 3, 128, 2, torch.float16),
            (3, 1024, 4, 256, 2, torch.float16),
            (4, 2048, 4, 64, 2, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    V = K * expand_ratio

    q = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, K, V), dtype=dtype, device=device).requires_grad_()

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ref, ref_ht = fused_recurrent_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, tri_ht = chunk_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('H', 'K', 'expand_ratio', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-expand_ratio{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 1, [0, 15], torch.float16),
            (4, 64, 2, [0, 256, 500, 1000], torch.float16),
            (4, 100, 2, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen(
    H: int,
    K: int,
    expand_ratio: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    V = K * expand_ratio

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((N, H, K, V), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = fused_recurrent_retention(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_retention(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'expand_ratio', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-expand_ratio{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float16),
            (2, 500, 3, 60, 1, torch.float16),
            (2, 1000, 3, 100, 1, torch.float16),
            (2, 1000, 3, 128, 2, torch.float16),
            (3, 1024, 4, 256, 2, torch.float16),
            (4, 2048, 4, 64, 2, torch.float16),
        ]
    ],
)
def test_fused_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    V = K * expand_ratio

    q = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, K, V), dtype=dtype, device=device).requires_grad_()

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ref, ref_ht = fused_recurrent_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, tri_ht = fused_chunk_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('H', 'K', 'expand_ratio', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-expand_ratio{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 1, [0, 15], torch.float16),
            (4, 64, 2, [0, 256, 500, 1000], torch.float16),
            (4, 100, 2, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_fused_chunk_varlen(
    H: int,
    K: int,
    expand_ratio: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    V = K * expand_ratio

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((N, H, K, V), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = fused_recurrent_retention(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_chunk_retention(
        q=q,
        k=k,
        v=v,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'expand_ratio', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-expand_ratio{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float16),
            (2, 500, 4, 60, 1, torch.float16),
            (2, 1024, 8, 128, 1, torch.float16),
            (3, 1024, 8, 128, 2, torch.float16),
            (3, 1024, 8, 256, 2, torch.float16),
            (4, 2048, 8, 64, 2, torch.float16),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    V = K * expand_ratio

    q = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, _ = fused_recurrent_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, _ = parallel_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
