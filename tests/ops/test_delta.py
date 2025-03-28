# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.utils.testing import assert_close
from fla.utils import device

compiled_mode = os.getenv("COMPILER_MODE") == "1"
if compiled_mode:
    test_b_list = [1]
    test_t_list = [64]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
else:
    test_b_list = [2]
    test_t_list = [1, 7, 15, 63, 286, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [32, 64, 100, 256]
test_h_list = [2]


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    head_first: bool
):
    if head_first:
        q = torch.randn(B, H, T, D, dtype=dtype)
        k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v = torch.randn(B, H, T, D, dtype=dtype)
        beta = torch.rand(B, H, T, dtype=dtype).sigmoid()
        h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    else:
        q = torch.randn(B, T, H, D, dtype=dtype)
        k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v = torch.randn(B, T, H, D, dtype=dtype)
        beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
        h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=head_first
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=head_first
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close("  o", ref, tri, 0.005)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close(" dq", ref_dq, tri_dq, 0.007)
    assert_close(" dk", ref_dk, tri_dk, 0.008)
    assert_close(" dv", ref_dv, tri_dv, 0.007)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.007)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", test_t_varlen_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set"
)
def test_chunk_varlen(
    N: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=offsets,
        head_first=False
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=offsets,
        head_first=False
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close("  o", ref, tri, 0.005)
    assert_close(" ht", ref_ht, tri_ht, 0.005)
    assert_close(" dq", ref_dq, tri_dq, 0.007)
    assert_close(" dk", ref_dk, tri_dk, 0.008)
    assert_close(" dv", ref_dv, tri_dv, 0.007)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.007)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_l2_in_kernel(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
):
    q = torch.randn(B, H, T, D, dtype=dtype)
    k = torch.randn(B, H, T, D, dtype=dtype)
    v = torch.randn(B, H, T, D, dtype=dtype)
    beta = torch.rand(B, H, T, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)

    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_delta_rule(
        F.normalize(q.clone(), p=2, dim=-1).to(dtype),
        F.normalize(k.clone(), p=2, dim=-1).to(dtype),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=True
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = chunk_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=True,
        use_qk_l2norm_in_kernel=True
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None
    assert_close("  o", ref, tri, 0.01)
    assert_close(" ht", ref_ht, tri_ht, 0.01)
    assert_close(" dq", ref_dq, tri_dq, 0.01)
    assert_close(" dk", ref_dk, tri_dk, 0.01)
    assert_close(" dv", ref_dv, tri_dv, 0.01)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.01)
    assert_close("dh0", ref_dh0, tri_dh0, 0.01)

    tri, tri_ht = fused_recurrent_delta_rule(
        F.normalize(q.clone().float(), p=2, dim=-1).to(dtype),
        F.normalize(k.clone().float(), p=2, dim=-1).to(dtype),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=True
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=True,
        use_qk_l2norm_in_kernel=True
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    assert_close("  o", ref, tri, 0.002)
    assert_close(" ht", ref_ht, tri_ht, 0.002)
    assert_close(" dq", ref_dq, tri_dq, 0.002)
    assert_close(" dk", ref_dk, tri_dk, 0.002)
    assert_close(" dv", ref_dv, tri_dv, 0.002)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.002)
    assert_close("dh0", ref_dh0, tri_dh0, 0.002)

    tri, tri_ht = fused_recurrent_delta_rule(
        F.normalize(q.float().clone(), p=2, dim=-1).to(dtype),
        F.normalize(k.float().clone(), p=2, dim=-1).to(dtype),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=True
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        head_first=True,
        use_qk_l2norm_in_kernel=True
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None
    assert_close("  o", ref, tri, 0.002)
    assert_close(" ht", ref_ht, tri_ht, 0.002)
    assert_close(" dq", ref_dq, tri_dq, 0.002)
    assert_close(" dk", ref_dk, tri_dk, 0.002)
    assert_close(" dv", ref_dv, tri_dv, 0.002)
    assert_close(" db", ref_dbeta, tri_dbeta, 0.002)
    assert_close("dh0", ref_dh0, tri_dh0, 0.002)
