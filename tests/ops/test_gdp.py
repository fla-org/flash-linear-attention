# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_product.chunk_ref import chunk_gated_delta_product_ref
from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import IS_INTEL_ALCHEMIST, IS_NVIDIA, assert_close, device, device_torch_lib


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'gate_logit_normalizer', 'mask_p', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-num_householder{}-gate_logit_normalizer{}-mask_p{}-l2norm{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 0.1, 1, 1, 0, False, torch.float16),
            (2, 200, 3, 60, 0.1, 1, 1, 0, False, torch.float16),
            (2, 1000, 4, 64, 0.1, 2, 0.1, 0.5, False, torch.float16),
            (2, 1024, 4, 64, 1, 2, 1, 0, True, torch.float16),
            (2, 1024, 6, 100, 1, 2, 10, 0, False, torch.float16),
            (4, 1500, 8, 128, 0.1, 3, 1, 0.5, False, torch.float16),
            (2, 2048, 8, 128, 1, 3, 1, 0, False, torch.float16),
            (2, 2048, 8, 128, 1, 3, 1, 0, True, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

    tri, tri_ht = chunk_gated_delta_product(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'num_householder', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-num_householder{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, 3, 0, [0, 63], torch.float16),
            (2, 100, 2, 0, [0, 63, 100, 500, 1000], torch.float16),
            (2, 100, 2, 0, [0, 100, 256, 512, 1500, 1500], torch.float16),
            (2, 128, 2, 0, [0, 100, 300, 800, 1500, 2000], torch.float16),
            (2, 128, 2, 0.5, [0, 31, 111, 799, 1000, 1500, 1800, 2000], torch.float16),
            (2, 128, 2, 0.5, [0, 63, 300, 800, 1000, 1399, 2048], torch.float16),
            (2, 256, 3, 0, [0, 100, 123, 300, 500, 800, 1000, 1500, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.smoke
def test_chunk_varlen(
    H: int,
    D: int,
    num_householder: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype), dim=-1, p=2)
    v = torch.randn((1, T*num_householder, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T*num_householder, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(q)
    dht = torch.rand_like(h0)
    scale = D ** -0.5

    tri, tri_ht = chunk_gated_delta_product(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    torch_ref = torch.zeros_like(ref)
    torch_ref_ht = torch.zeros_like(ref_ht)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        q_i = q[:, start:end, :, :]
        k_i = k[:, start*num_householder:end*num_householder, :, :]
        v_i = v[:, start*num_householder:end*num_householder, :, :]
        g_i = g[:, start:end, :]
        beta_i = beta[:, start*num_householder:end*num_householder, :]
        o3_i, h3_i = naive_recurrent_gated_delta_product(
            q_i, k_i, v_i, g_i, beta_i, scale=scale, cu_seqlens=None, output_final_state=True, num_householder=num_householder,
        )
        torch_ref[:, start:end, :, :] = o3_i
        torch_ref_ht[i, :, :, :] = h3_i.squeeze(0)

    ((torch_ref * do).sum() + (torch_ref_ht * dht).sum()).backward(retain_graph=True)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)


def test_naive_varlen():
    H, D, num_householder = 2, 64, 2
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    cu_seqlens = torch.LongTensor([0, 63, 100, 100, 500]).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=torch.float16), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn((1, T*num_householder, H, D), dtype=torch.float16)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float16))
    beta = torch.rand(1, T*num_householder, H, dtype=torch.float16).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float16)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device), (q, k, v, beta, g, h0))
    scale = D ** -0.5

    ref, ref_ht = naive_recurrent_gated_delta_product(
        q, k, v, g, beta, scale, cu_seqlens,
        initial_state=h0, output_final_state=True, num_householder=num_householder,
    )
    tri, tri_ht = chunk_gated_delta_product(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True, num_householder=num_householder,
        cu_seqlens=cu_seqlens,
    )

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)


@pytest.mark.skipif(not IS_NVIDIA, reason='requires CUDA graph capture')
@pytest.mark.parametrize('num_householder', [1, 3], ids=['hh1', 'hh3'])
@pytest.mark.parametrize('use_qk_l2norm_in_kernel', [False, True], ids=['normalized', 'fused_l2norm'])
def test_chunk_varlen_cuda_graph(num_householder: int, use_qk_l2norm_in_kernel: bool):
    """Replay packed forward/backward with refreshed boundaries and chunk descriptors."""
    torch.manual_seed(42)
    T, H, K, V = 256, 2, 64, 80
    dtype = torch.bfloat16
    names = ('q', 'k', 'v', 'g', 'beta', 'initial_state')

    def make_inputs():
        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T * num_householder, H, K, dtype=dtype, device=device)
        if not use_qk_l2norm_in_kernel:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        return dict(zip(names, (
            q, k,
            torch.randn(1, T * num_householder, H, V, dtype=dtype, device=device),
            F.logsigmoid(torch.randn(1, T, H, dtype=torch.float32, device=device)),
            torch.randn(1, T * num_householder, H, dtype=dtype, device=device).sigmoid(),
            torch.randn(2, H, K, V, dtype=torch.float32, device=device),
        ), strict=True))

    def descriptors(cu):
        return prepare_chunk_indices(cu, 64), prepare_chunk_indices(cu * num_householder, 64)

    leaves = {name: value.requires_grad_() for name, value in make_inputs().items()}
    cu = torch.tensor([0, 112, T], dtype=torch.long, device=device)
    chunk_indices, chunk_indices_dp = descriptors(cu)
    do = torch.randn(1, T, H, V, dtype=dtype, device=device)
    dht = torch.randn_like(leaves['initial_state'])

    def step(inputs, boundaries, grad_output, grad_state, **kwargs):
        output, state = chunk_gated_delta_product(
            **inputs, num_householder=num_householder, output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=boundaries, **kwargs,
        )
        gradients = torch.autograd.grad((output, state), tuple(inputs.values()), (grad_output, grad_state))
        return (output, state, *gradients)

    stream = device_torch_lib.Stream()
    stream.wait_stream(device_torch_lib.current_stream())
    with device_torch_lib.stream(stream):
        for _ in range(3):
            step(leaves, cu, do, dht, chunk_indices=chunk_indices, chunk_indices_dp=chunk_indices_dp)
    device_torch_lib.current_stream().wait_stream(stream)
    device_torch_lib.synchronize()
    graph = device_torch_lib.CUDAGraph()
    try:
        with device_torch_lib.graph(graph, stream=stream):
            captured = step(leaves, cu, do, dht, chunk_indices=chunk_indices, chunk_indices_dp=chunk_indices_dp)
        device_torch_lib.current_stream().wait_stream(stream)
        for cut in (112, 144, 112):
            fresh = {name: value.requires_grad_() for name, value in make_inputs().items()}
            boundaries = torch.tensor([0, cut, T], dtype=cu.dtype, device=device)
            new_indices, new_indices_dp = descriptors(boundaries)
            assert new_indices.shape == chunk_indices.shape
            assert new_indices_dp.shape == chunk_indices_dp.shape
            with torch.no_grad():
                for name in names:
                    leaves[name].copy_(fresh[name])
                cu.copy_(boundaries)
                chunk_indices.copy_(new_indices)
                chunk_indices_dp.copy_(new_indices_dp)
                do.normal_()
                dht.normal_()
            eager = step(fresh, boundaries, do, dht)
            graph.replay()
            device_torch_lib.synchronize()
            for name, expected, actual in zip(('output', 'state', *names), eager, captured, strict=True):
                assert_close(name, expected, actual, ratio=0, err_atol=0)
    finally:
        device_torch_lib.synchronize()
        graph.reset()
