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

from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float),
            (2, 1024, 4, 60, 1, torch.float),
            (2, 1024, 8, 128, 0.1, torch.float),
            (2, 1024, 8, 128, 1, torch.float),
            (2, 1024, 8, 128, 10, torch.float),
            (4, 2048, 8, 64, 1, torch.float),
            (2, 1024, 8, 128, 0.1, torch.float16),
            (2, 1024, 8, 128, 10, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = (F.logsigmoid(torch.rand((B, T, H, D), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    h0 = torch.rand(B, H, D, D, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn((B, H, D, D), dtype=dtype, device=device)

    ref, ref_ht = naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 256, 4, 64, torch.float),
            (2, 1024, 4, 128, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent_transpose_state(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.rand((B, T, H, D), dtype=dtype, device=device)
    k = torch.rand((B, T, H, D), dtype=dtype, device=device)
    v = torch.rand((B, T, H, D), dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand((B, T, H, D), dtype=dtype, device=device))
    h0 = torch.rand(B, H, D, D, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run(transpose: bool):
        q_, k_, v_, g_ = (x.detach().clone().requires_grad_() for x in (q, k, v, g))
        h0_in = h0.transpose(-1, -2).contiguous() if transpose else h0.clone()
        dht_in = dht.transpose(-1, -2).contiguous() if transpose else dht
        h0_in = h0_in.requires_grad_()
        out, ht = fused_recurrent_gla(
            q=q_, k=k_, v=v_, gk=g_,
            initial_state=h0_in,
            output_final_state=True,
            transpose_state_layout=transpose,
        )
        ((out * do).sum() + (ht * dht_in).sum()).backward()
        return out, ht, q_.grad, k_.grad, v_.grad, g_.grad, h0_in.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dg, ref_dh0 = run(False)
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dg, tri_dh0 = run(True)

    assert_close('o', ref_o, tri_o, 0.005)
    assert_close('ht', ref_ht, tri_ht.transpose(-1, -2), 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0.transpose(-1, -2), 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, [0, 15], torch.float),
            (4, 64, [0, 256, 500, 1000], torch.float),
            (4, 100, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 64, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (4, 128, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_fused_recurrent_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.rand((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.rand((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.rand((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.rand((1, T, H, D), dtype=dtype, device=device)).requires_grad_()
    h0 = torch.rand(N, H, D, D, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn((N, H, D, D), dtype=dtype, device=device)

    refs, ref_hts = [], []
    for i in range(N):
        ref, ref_ht = naive_recurrent_gla(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            gk=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        refs.append(ref)
        ref_hts.append(ref_ht)
    ref = torch.cat(refs, dim=1)
    ref_ht = torch.cat(ref_hts, dim=0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, torch.float16),
            (2, 1024, 4, 60, 1, torch.float16),
            (2, 1024, 8, 128, 0.1, torch.float16),
            (2, 1024, 8, 128, 1, torch.float16),
            (2, 1024, 8, 128, 10, torch.float16),
            (4, 2048, 8, 64, 1, torch.float16),
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
    dtype: torch.dtype,
    gate_logit_normalizer: float,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # [B, T, H, D]
    q = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.rand((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = (F.logsigmoid(torch.rand((B, T, H, D), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn((B, H, D, D), dtype=torch.float32, device=device)

    tri, tri_ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum().to(do.dtype)).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    ref, ref_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
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
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
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

    q = torch.rand((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.rand((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.rand((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.rand((1, T, H, D), dtype=dtype, device=device)).requires_grad_()
    h0 = torch.rand((N, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.rand((N, H, D, D), dtype=torch.float32, device=device)

    ref, ref_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    ((ref * do).sum() + (ref_ht * dht).sum().to(do.dtype)).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)
