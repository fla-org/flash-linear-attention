
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.ttt import chunk_ttt_linear, fused_chunk_ttt_linear
from fla.ops.ttt.naive import chunk_ttt_linear_ref
from fla.utils import assert_close, check_shared_mem, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float16),
            (2, 100, 4, 60, 0.1, torch.float16),
            (2, 1024, 3, 128, 0.1, torch.float16),
            (2, 1024, 4, 128, 1, torch.float16),
            (3, 2000, 4, 128, 0.1, torch.float16),
            (4, 2048, 8, 64, 0.1, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    if D > 64 and check_shared_mem('hopper') is False:
        pytest.skip(reason="Current CI do not support this config")
    if T > 1000:
        pytest.skip(reason="Current CI do not support this config")
    eta_base = 5e-3
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    w = torch.randn(H, D, dtype=dtype)
    b = torch.randn(H, D, dtype=dtype)
    eta = torch.randn(B, T, H, 1, dtype=dtype) * eta_base
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    hb0 = torch.randn(B, H, 1, D, dtype=torch.float32)

    q, k, v, w, b, eta, h0, hb0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, w, b, eta, h0, hb0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)
    dhbt = torch.rand_like(hb0)

    tri, tri_ht, tri_hbt = chunk_ttt_linear(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        eta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        initial_state_bias=hb0.clone(),
    )
    ((tri * do).sum() + (tri_ht * dht).sum() + (tri_hbt * dhbt).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dw, tri_db, tri_deta, \
        tri_dh0, tri_dhb0 = q.grad, k.grad, v.grad, w.grad, b.grad, eta.grad, h0.grad, hb0.grad
    q.grad = k.grad = v.grad = w.grad = b.grad = eta.grad = h0.grad = hb0.grad = None

    ref, ref_ht, ref_hbt = chunk_ttt_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        eta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        initial_state_bias=hb0.clone(),
    )
    ((ref * do).sum() + (ref_ht * dht).sum() + (ref_hbt * dhbt).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dw, ref_db, ref_deta, \
        ref_dh0, ref_dhb0 = q.grad, k.grad, v.grad, w.grad, b.grad, eta.grad, h0.grad, hb0.grad

    assert_close("   o", ref, tri, 0.005)
    assert_close("  ht", ref_ht, tri_ht, 0.005)
    assert_close(" hbt", ref_hbt, tri_hbt, 0.005)
    assert_close("  dq", ref_dq, tri_dq, 0.005)
    assert_close("  dk", ref_dk, tri_dk, 0.010)
    assert_close("  dv", ref_dv, tri_dv, 0.007)
    assert_close("  dw", ref_dw, tri_dw, 0.006)
    assert_close("  db", ref_db, tri_db, 0.006)
    assert_close("  de", ref_deta, tri_deta, 0.030)  # because the last element of the chunk
    assert_close(" de0", ref_deta[:, :14, :, :], tri_deta[:, :14, :, :], 0.010)
    assert_close(" dh0", ref_dh0, tri_dh0, 0.007)
    assert_close("dhb0", ref_dhb0, tri_dhb0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float16),
            (2, 100, 4, 60, 0.1, torch.float16),
            (2, 1024, 3, 128, 0.1, torch.float16),
            (2, 1024, 4, 128, 1, torch.float16),
            (3, 2000, 4, 128, 0.1, torch.float16),
            (4, 2048, 8, 64, 0.1, torch.float16),
        ]
    ],
)
def test_fused_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    if D > 64 and check_shared_mem('hopper') is False:
        pytest.skip(reason="Current CI do not support this config")
    if T > 1000:
        pytest.skip(reason="Current CI do not support this config")
    eta_base = 5e-3
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    w = torch.randn(H, D, dtype=dtype)
    b = torch.randn(H, D, dtype=dtype)
    eta = torch.randn(B, T, H, 1, dtype=dtype) * eta_base
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    hb0 = torch.randn(B, H, 1, D, dtype=torch.float32)

    q, k, v, w, b, eta, h0, hb0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, w, b, eta, h0, hb0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)
    dhbt = torch.rand_like(hb0)

    tri, tri_ht, tri_hbt = fused_chunk_ttt_linear(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        eta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        initial_state_bias=hb0.clone(),
    )
    ((tri * do).sum() + (tri_ht * dht).sum() + (tri_hbt * dhbt).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dw, tri_db, tri_deta, \
        tri_dh0, tri_dhb0 = q.grad, k.grad, v.grad, w.grad, b.grad, eta.grad, h0.grad, hb0.grad
    q.grad = k.grad = v.grad = w.grad = b.grad = eta.grad = h0.grad = hb0.grad = None

    ref, ref_ht, ref_hbt = chunk_ttt_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        eta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        initial_state_bias=hb0.clone(),
    )
    ((ref * do).sum() + (ref_ht * dht).sum() + (ref_hbt * dhbt).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dw, ref_db, ref_deta, \
        ref_dh0, ref_dhb0 = q.grad, k.grad, v.grad, w.grad, b.grad, eta.grad, h0.grad, hb0.grad

    assert_close("   o", ref, tri, 0.005)
    assert_close("  ht", ref_ht, tri_ht, 0.005)
    assert_close(" hbt", ref_hbt, tri_hbt, 0.005)
    assert_close("  dq", ref_dq, tri_dq, 0.005)
    assert_close("  dk", ref_dk, tri_dk, 0.010)
    assert_close("  dv", ref_dv, tri_dv, 0.007)
    assert_close("  dw", ref_dw, tri_dw, 0.005)
    assert_close("  db", ref_db, tri_db, 0.005)
    assert_close("  de", ref_deta, tri_deta, 0.03)  # because the last element of the chunk
    assert_close(" de0", ref_deta[:, :14, :, :], tri_deta[:, :14, :, :], 0.008)
    assert_close(" dh0", ref_dh0, tri_dh0, 0.006)
    assert_close("dhb0", ref_dhb0, tri_dhb0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, [0, 15], torch.float16),
            (3, 60, [0, 111, 500], torch.float16),
            (3, 64, [0, 256, 500, 900, 1000], torch.float16),
            (4, 100, [0, 15, 100, 300, 1200, 1599, 1800, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if D > 64 and check_shared_mem('hopper') is False:
        pytest.skip(reason="Current CI do not support this config")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    eta_base = 5e-3
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    eta = torch.randn(1, T, H, 1, dtype=dtype) * eta_base
    w = torch.randn(H, D, dtype=dtype)
    b = torch.randn(H, D, dtype=dtype)
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)
    hb0 = torch.randn((N, H, 1, D), dtype=torch.float32)
    q, k, v, w, b, eta, h0, hb0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, w, b, eta, h0, hb0))

    tri, tri_ht, tri_hbt = chunk_ttt_linear(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        eta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        initial_state_bias=hb0.clone(),
        cu_seqlens=cu_seqlens,
    )

    ref = []
    ref_ht = []
    ref_hbt = []
    for i in range(N):
        ref_i, ref_ht_i, ref_hbt_i = chunk_ttt_linear_ref(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            w=w,
            b=b,
            eta=eta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            initial_state_bias=hb0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
        ref_hbt.append(ref_hbt_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)
    ref_hbt = torch.cat(ref_hbt, 0)

    assert_close("  o", ref, tri, 0.005)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close("hbt", ref_hbt, tri_hbt, 0.005)
