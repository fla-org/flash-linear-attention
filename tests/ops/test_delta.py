# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.delta_rule.naive import delta_rule_recurrence
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, False, torch.float16),
            (2, 100, 4, 60, 0.1, False, torch.float16),
            (2, 1000, 3, 128, 0.1, False, torch.float16),
            (2, 1024, 4, 128, 1, True, torch.float16),
            (3, 2000, 4, 128, 0.1, False, torch.float16),
            (4, 2048, 8, 64, 0.1, False, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close('o', ref, tri, 0.006)
    assert_close('ht', ref_ht, tri_ht, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'dtype', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-{}-chunk{}".format(*test))
        for chunk_size in [16, 32, 64]
        for test in [
            (1, 64, 2, 32, 0.1, torch.bfloat16, chunk_size),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk_with_chunk_size(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
    chunk_size: int,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run_ref():
        q_, k_, v_, beta_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, h0))
        o, ht = fused_recurrent_delta_rule(
            q=F.normalize(q_, p=2, dim=-1),
            k=F.normalize(k_, p=2, dim=-1),
            v=v_,
            beta=beta_,
            scale=scale,
            output_final_state=True,
            initial_state=h0_,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, beta_.grad, h0_.grad

    def run_tri(chunk_size: int):
        q_, k_, v_, beta_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, beta, h0))
        o, ht = chunk_delta_rule(
            q=F.normalize(q_, p=2, dim=-1),
            k=F.normalize(k_, p=2, dim=-1),
            v=v_,
            beta=beta_,
            scale=scale,
            output_final_state=True,
            initial_state=h0_,
            chunk_size=chunk_size,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()
        return o, ht, q_.grad, k_.grad, v_.grad, beta_.grad, h0_.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = run_ref()
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = run_tri(chunk_size)

    assert_close(f'o@{chunk_size}', ref_o, tri_o, 0.006)
    assert_close(f'ht@{chunk_size}', ref_ht, tri_ht, 0.006)
    assert_close(f'dq@{chunk_size}', ref_dq, tri_dq, 0.008)
    assert_close(f'dk@{chunk_size}', ref_dk, tri_dk, 0.008)
    assert_close(f'dv@{chunk_size}', ref_dv, tri_dv, 0.008)
    assert_close(f'db@{chunk_size}', ref_dbeta, tri_dbeta, 0.008)
    assert_close(f'dh0@{chunk_size}', ref_dh0, tri_dh0, 0.008)


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
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
@pytest.mark.smoke
def test_chunk_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    beta = torch.randn(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=dtype)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    ref, ref_ht = fused_recurrent_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    tri, tri_ht = chunk_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        output_final_state=True,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'scale', 'use_qk_l2norm_in_kernel', 'has_h0', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-K{}-V{}-scale{}-l2{}-h0{}-{}".format(*test),
        )
        for test in [
            (2, 32, 2, 32, 16, 1.0, False, False, torch.float32),
            (2, 32, 3, 64, 48, 0.1, False, True, torch.float32),
            (2, 64, 4, 64, 64, 1.0, True, True, torch.float16),
            (2, 50, 3, 96, 100, 0.2, False, True, torch.float32),
            (1, 64, 2, 128, 128, 0.1, True, False, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent_headwise_beta(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    has_h0: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    beta = torch.randn(B, T, H, V, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, K, V, dtype=torch.float32) if has_h0 else None
    q, k, v, beta = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta))
    if h0 is not None:
        h0 = h0.to(device).requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0) if h0 is not None else None

    tri, tri_ht = fused_recurrent_delta_rule(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone() if h0 is not None else None,
        output_final_state=has_h0,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    loss = (tri * do).sum()
    if has_h0:
        loss = loss + (tri_ht * dht).sum()
    loss.backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta = q.grad, k.grad, v.grad, beta.grad
    tri_dh0 = h0.grad if h0 is not None else None
    q.grad = k.grad = v.grad = beta.grad = None
    if h0 is not None:
        h0.grad = None

    ref, ref_ht = delta_rule_recurrence(
        q=F.normalize(q.clone().float(), p=2, dim=-1).transpose(1, 2) * (scale * (K ** 0.5)),
        k=F.normalize(k.clone().float(), p=2, dim=-1).transpose(1, 2),
        v=v.clone().float().transpose(1, 2),
        beta=beta.clone().float().transpose(1, 2),
        initial_state=h0.clone().float() if h0 is not None else None,
        output_final_state=has_h0,
    )
    ref = ref.transpose(1, 2)
    ref_loss = (ref.to(dtype) * do).sum()
    if has_h0:
        ref_loss = ref_loss + (ref_ht * dht).sum()
    ref_loss.backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta = q.grad, k.grad, v.grad, beta.grad
    ref_dh0 = h0.grad if h0 is not None else None

    tol = 0.002 if dtype == torch.float32 else 0.008
    assert_close('o', ref, tri, tol)
    if has_h0:
        assert_close('ht', ref_ht, tri_ht, tol)
    assert_close('dq', ref_dq, tri_dq, tol)
    assert_close('dk', ref_dk, tri_dk, tol)
    assert_close('dv', ref_dv, tri_dv, tol)
    assert_close('db', ref_dbeta, tri_dbeta, tol)
    if has_h0:
        assert_close('dh0', ref_dh0, tri_dh0, tol)


@pytest.mark.parametrize(
    ('H', 'K', 'V', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-K{}-V{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, 64, [0, 15, 48], torch.float32),
            (3, 64, 96, [0, 23, 64, 100], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent_headwise_beta_varlen(
    H: int,
    K: int,
    V: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = F.normalize(torch.randn(1, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(1, T, H, V, dtype=dtype)
    beta = torch.randn(1, T, H, V, dtype=dtype).sigmoid()
    h0 = torch.randn(N, H, K, V, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    tri, tri_ht = fused_recurrent_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref_outs = []
    ref_hts = []
    for i in range(N):
        bos, eos = cu_seqlens[i], cu_seqlens[i + 1]
        o_i, ht_i = delta_rule_recurrence(
            q=q[:, bos:eos].float().transpose(1, 2),
            k=k[:, bos:eos].float().transpose(1, 2),
            v=v[:, bos:eos].float().transpose(1, 2),
            beta=beta[:, bos:eos].float().transpose(1, 2),
            initial_state=h0[i:i + 1].float(),
            output_final_state=True,
        )
        ref_outs.append(o_i.transpose(1, 2))
        ref_hts.append(ht_i)
    ref = torch.cat(ref_outs, dim=1)
    ref_ht = torch.cat(ref_hts, dim=0)
    ((ref.to(dtype) * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    tol = 0.002 if dtype == torch.float32 else 0.008
    assert_close('o', ref, tri, tol)
    assert_close('ht', ref_ht, tri_ht, tol)
    assert_close('dq', ref_dq, tri_dq, tol)
    assert_close('dk', ref_dk, tri_dk, tol)
    assert_close('dv', ref_dv, tri_dv, tol)
    assert_close('db', ref_dbeta, tri_dbeta, tol)
    assert_close('dh0', ref_dh0, tri_dh0, tol)
