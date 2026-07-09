# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

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
    ('B', 'T', 'H', 'K', 'dtype', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-{}-chunk{}".format(*test))
        for chunk_size in [16, 32, 64]
        for test in [
            (1, 64, 2, 64, torch.float16, chunk_size),
        ]
    ],
)
def test_chunk_with_chunk_size(
    B: int,
    T: int,
    H: int,
    K: int,
    dtype: torch.dtype,
    chunk_size: int,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, K, dtype=dtype, device=device)
    h0 = torch.randn(B, H, K, K, dtype=dtype, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run_ref():
        q_, k_, v_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, h0))
        o, ht = fused_recurrent_retention(q_, k_, v_, initial_state=h0_, output_final_state=True)
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, h0_.grad

    def run_tri(chunk_size: int):
        q_, k_, v_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, h0))
        o, ht = chunk_retention(q_, k_, v_, initial_state=h0_, output_final_state=True, chunk_size=chunk_size)
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, h0_.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dh0 = run_ref()
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dh0 = run_tri(chunk_size)

    assert_close(f'o@{chunk_size}', ref_o, tri_o, 0.005)
    assert_close(f'ht@{chunk_size}', ref_ht, tri_ht, 0.005)
    assert_close(f'dq@{chunk_size}', ref_dq, tri_dq, 0.005)
    assert_close(f'dk@{chunk_size}', ref_dk, tri_dk, 0.005)
    assert_close(f'dv@{chunk_size}', ref_dv, tri_dv, 0.005)
    assert_close(f'dh0@{chunk_size}', ref_dh0, tri_dh0, 0.005)


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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'expand_ratio', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-expand_ratio{}-{}".format(*test))
        for test in [
            (2, 256, 4, 64, 1, torch.float16),
            (2, 1024, 4, 128, 2, torch.float16),
        ]
    ],
)
def test_fused_recurrent_state_v_first(
    B: int,
    T: int,
    H: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    V = K * expand_ratio

    q = torch.randn((B, T, H, K), dtype=dtype, device=device)
    k = torch.randn((B, T, H, K), dtype=dtype, device=device)
    v = torch.randn((B, T, H, V), dtype=dtype, device=device)
    h0 = torch.randn((B, H, K, V), dtype=dtype, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run(state_v_first: bool):
        q_, k_, v_ = (x.detach().clone().requires_grad_() for x in (q, k, v))
        h0_in = h0.transpose(-1, -2).contiguous() if state_v_first else h0.clone()
        dht_in = dht.transpose(-1, -2).contiguous() if state_v_first else dht
        h0_in = h0_in.requires_grad_()
        out, ht = fused_recurrent_retention(
            q_, k_, v_,
            initial_state=h0_in,
            output_final_state=True,
            state_v_first=state_v_first,
        )
        ((out * do).sum() + (ht * dht_in).sum()).backward()
        return out, ht, q_.grad, k_.grad, v_.grad, h0_in.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dh0 = run(False)
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dh0 = run(True)

    assert tri_ht.shape == (B, H, V, K)
    assert_close('o', ref_o, tri_o, 0.005)
    assert_close('ht', ref_ht, tri_ht.transpose(-1, -2), 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0.transpose(-1, -2), 0.005)

    # the legacy `transpose_state_layout` kwarg maps to `state_v_first` with a warning,
    # and passing both names at once is rejected
    with pytest.warns(DeprecationWarning):
        fused_recurrent_retention(q, k, v, transpose_state_layout=True)
    with pytest.raises(ValueError):
        fused_recurrent_retention(q, k, v, state_v_first=True, transpose_state_layout=True)
