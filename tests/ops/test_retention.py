# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.retention import chunk_retention, fused_recurrent_retention, parallel_retention
from fla.ops.retention.naive import naive_retention
from fla.ops.utils.testing import assert_close
from fla.utils import device

compiled_mode = os.getenv("COMPILER_MODE") == "1"
if compiled_mode:
    test_b_list = [1]
    test_t_list = [64]
    test_t_varlen_list = test_t_list
    test_d_list = [32, 64, 100]
else:
    test_b_list = [2]
    test_t_list = [1, 7, 15, 63, 286, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [32, 64, 100]
test_h_list = [2]


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("K", test_d_list)
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    K: int,
    expand_ratio: int,
    head_first: bool,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    V = K * expand_ratio

    if head_first:
        q = torch.randn((B, H, T, K), dtype=dtype, device=device).requires_grad_()
        k = torch.randn((B, H, T, K), dtype=dtype, device=device).requires_grad_()
        v = torch.randn((B, H, T, V), dtype=dtype, device=device).requires_grad_()
    else:
        q = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
        k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_()
        v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_()
    h0 = torch.randn((B, H, K, V), dtype=dtype, device=device).requires_grad_()

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ref, ref_ht = fused_recurrent_retention(q, k, v, initial_state=h0, output_final_state=True, head_first=head_first)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, tri_ht = chunk_retention(q, k, v, initial_state=h0, output_final_state=True, head_first=head_first)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize("N", test_b_list)
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("K", test_d_list)
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
def test_chunk_varlen(
    N: int,
    T: int,
    H: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    V = K * expand_ratio

    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
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
        cu_seqlens=offsets,
        head_first=False
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
        cu_seqlens=offsets,
        head_first=False
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close("  o", ref, tri, 0.004)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close(" dq", ref_dq, tri_dq, 0.005)
    assert_close(" dk", ref_dk, tri_dk, 0.005)
    assert_close(" dv", ref_dv, tri_dv, 0.005)
    assert_close("dh0", ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("K", test_d_list)
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_parallel(
    B: int,
    H: int,
    T: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    V = K * expand_ratio

    q = torch.randn((B, H, T, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(v)

    ref = naive_retention(q, k, v)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri, _ = parallel_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
