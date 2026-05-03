# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.linear_attn.naive import naive_chunk_linear_attn, naive_recurrent_linear_attn
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
    ('B', 'T', 'H', 'D', 'normalize', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-norm{}-{}".format(*test))
        for test in [
            (1, 128, 2, 64, False, torch.float),
            (2, 256, 4, 60, False, torch.float),
            (1, 128, 2, 64, True, torch.float),
            (2, 256, 4, 60, True, torch.float),
        ]
    ],
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    normalize: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)

    ref, _ = naive_recurrent_linear_attn(q, k, v, normalize=normalize)
    tri = naive_chunk_linear_attn(q, k, v, normalize=normalize)

    assert_close('o', ref, tri, 1e-3)


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


@pytest.mark.parametrize(
    ('fn', 'B', 'T', 'split', 'H', 'D'),
    [
        pytest.param(fn, 2, 256, 128, 4, 64, id=f"{name}-split128")
        for fn, name in [
            (fused_recurrent_linear_attn, 'fused_recurrent'),
            (fused_chunk_linear_attn, 'fused_chunk'),
            (chunk_linear_attn, 'chunk'),
        ]
    ],
)
def test_normalize_split_resume(fn, B: int, T: int, split: int, H: int, D: int):
    """Splitting at `split` and resuming via the returned (kv_state, z_state)
    must reproduce the single-call output when normalize=True."""
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=torch.float32, device=device)
    k = torch.randn((B, T, H, D), dtype=torch.float32, device=device)
    v = torch.randn((B, T, H, D), dtype=torch.float32, device=device)

    o_full, _ = fn(q=q, k=k, v=v, output_final_state=True, normalize=True)

    o_a, state_a = fn(
        q=q[:, :split], k=k[:, :split], v=v[:, :split],
        output_final_state=True, normalize=True,
    )
    o_b, _ = fn(
        q=q[:, split:], k=k[:, split:], v=v[:, split:],
        initial_state=state_a, output_final_state=True, normalize=True,
    )
    o_split = torch.cat([o_a, o_b], dim=1)

    assert_close('o', o_full, o_split, 0.002)


@pytest.mark.parametrize(
    ('fn', 'B', 'T', 'H', 'D'),
    [
        pytest.param(fn, 2, 256, 4, 64, id=f"{name}-normgrad")
        for fn, name in [
            (fused_recurrent_linear_attn, 'fused_recurrent'),
            (fused_chunk_linear_attn, 'fused_chunk'),
            (chunk_linear_attn, 'chunk'),
        ]
    ],
)
def test_normalize_grad(fn, B: int, T: int, H: int, D: int):
    """Gradient parity for `normalize=True` against the naive recurrent reference.

    Regression test for the missing `dk` denominator-path contribution: pre-fix,
    `chunk_global_cumsum` had no autograd so the chain `k -> k_cum -> denom` was
    severed and `dk` collapsed to the numerator-only contribution.

    Uses strictly-positive q, k ( normalize=True is designed for);
    """
    torch.manual_seed(42)
    eps = 0.1
    q = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    k = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref, _ = naive_recurrent_linear_attn(q, k, v, normalize=True)
    (ref * do).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, _ = fn(q=q, k=k, v=v, normalize=True)
    (tri * do).sum().backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('fn', 'B', 'T', 'split', 'H', 'D'),
    [
        pytest.param(fn, 2, 256, 128, 4, 64, id=f"{name}-zinitgrad")
        for fn, name in [
            (fused_recurrent_linear_attn, 'fused_recurrent'),
            (fused_chunk_linear_attn, 'fused_chunk'),
            (chunk_linear_attn, 'chunk'),
        ]
    ],
)
def test_normalize_zinit_grad(fn, B: int, T: int, split: int, H: int, D: int):
    """Gradient parity when chaining via `(kv_state, z_state)`: splitting at
    `split` and resuming with the prior `z_state` as `z_init` must produce the
    same q/k/v gradients as a single-call run."""
    torch.manual_seed(42)
    eps = 0.1
    q = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    k = (torch.randn((B, T, H, D), dtype=torch.float32, device=device).abs() + eps).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)

    o_full, _ = fn(q=q, k=k, v=v, normalize=True)
    (o_full * do).sum().backward()
    full_dq, q.grad = q.grad.clone(), None
    full_dk, k.grad = k.grad.clone(), None
    full_dv, v.grad = v.grad.clone(), None

    o_a, state_a = fn(
        q=q[:, :split], k=k[:, :split], v=v[:, :split],
        output_final_state=True, normalize=True,
    )
    o_b, _ = fn(
        q=q[:, split:], k=k[:, split:], v=v[:, split:],
        initial_state=state_a, normalize=True,
    )
    o_split = torch.cat([o_a, o_b], dim=1)
    (o_split * do).sum().backward()
    split_dq, q.grad = q.grad.clone(), None
    split_dk, k.grad = k.grad.clone(), None
    split_dv, v.grad = v.grad.clone(), None

    assert_close('o', o_full, o_split, 0.002)
    assert_close('dq', full_dq, split_dq, 0.005)
    assert_close('dk', full_dk, split_dk, 0.005)
    assert_close('dv', full_dv, split_dv, 0.005)
