
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.simple_gla.chunk import chunk_simple_gla
from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.naive import naive_parallel_simple_gla, naive_recurrent_simple_gla
from fla.ops.simple_gla.parallel import parallel_simple_gla
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 60, 1, 1, torch.float),
            (2, 1024, 8, 128, 1, 0.1, torch.float),
            (2, 1024, 8, 128, 0.1, 1, torch.float),
            (2, 1024, 8, 128, 1, 10, torch.float),
            (4, 2048, 8, 64, 0.1, 1, torch.float),
            (2, 1024, 8, 128, 1, 0.1, torch.float16),
            (2, 1024, 8, 128, 1, 10, torch.float16),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((B, T, H), dtype=dtype, device=device)
    g = (F.logsigmoid(g) / gate_logit_normalizer).requires_grad_()
    h0 = torch.randn(B, H, D, D, device=device).requires_grad_()
    dht = torch.randn_like(h0)
    do = torch.randn_like(v)
    ref, ref_ht = naive_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
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
    assert_close('dg', ref_dg, tri_dg, 0.005, err_atol=2e-4)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('H', 'D', 'scale', 'gate_logit_normalizer', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-scale{}-gate_logit_normalizer{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 64, 1, 1, [0, 15], torch.float),
            (4, 64, 1, 1, [0, 256, 500, 1000], torch.float),
            (4, 100, 0.1, 1, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 100, 1, 1, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 100, 1, 10, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 64, 1, 1, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (4, 128, 1, 1, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
def test_fused_recurrent_varlen(
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((1, T, H), dtype=dtype, device=device)
    g = (F.logsigmoid(g) / gate_logit_normalizer).requires_grad_()
    h0 = torch.randn(N, H, D, D, device=device).requires_grad_()
    dht = torch.randn_like(h0)
    do = torch.randn_like(v)

    refs, ref_hts = [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)):
        ref, ref_ht = naive_recurrent_simple_gla(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=g[:, bos:eos],
            scale=scale,
            initial_state=h0[i],
            output_final_state=True,
        )
        refs.append(ref)
        ref_hts.append(ref_ht)
    ref = torch.cat(refs, 1)
    ref_ht = torch.cat(ref_hts, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
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
    assert_close('dg', ref_dg, tri_dg, 0.005, err_atol=2e-4)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float16),
            (2, 500, 3, 60, 1, 1, torch.float16),
            (1, 1000, 4, 128, 1, 0.1, torch.float16),
            (2, 1000, 4, 128, 0.1, 1, torch.float16),
            (3, 1000, 4, 128, 0.1, 10, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    g = torch.randn((B, T, H), dtype=torch.float32, device=device)
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device=device).requires_grad_(True)
    dht = torch.randn_like(h0)
    g = (F.logsigmoid(g) / gate_logit_normalizer).requires_grad_(True)
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (dht * ref_ht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri * do).sum() + (dht * tri_ht).sum()).backward()
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

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H), dtype=dtype, device=device)).requires_grad_()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    dht = torch.randn_like(h0)
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum() + (dht * ref_ht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (dht * tri_ht).sum()).backward()
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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float16),
            (2, 500, 3, 60, 1, 1, torch.float16),
            (1, 1000, 4, 128, 1, 0.1, torch.float16),
            (2, 1000, 4, 128, 0.1, 1, torch.float16),
            (3, 1000, 4, 128, 0.1, 10, torch.float16),
            (4, 2048, 8, 64, 0.1, 1, torch.float16),
        ]
    ],
)
def test_fused_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    g = torch.randn((B, T, H), dtype=torch.float32, device=device)
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device=device).requires_grad_(True)
    dht = torch.randn_like(h0)
    g = (F.logsigmoid(g) / gate_logit_normalizer).requires_grad_(True)
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (dht * ref_ht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri * do).sum() + (dht * tri_ht).sum()).backward()
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
def test_fused_chunk_varlen(
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

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H), dtype=dtype, device=device)).requires_grad_()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32, device=device).requires_grad_()
    dht = torch.randn_like(h0)
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum() + (dht * ref_ht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (dht * tri_ht).sum()).backward()
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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 1, 1, torch.float16),
            (2, 500, 3, 60, 1, 1, torch.float16),
            (2, 1024, 4, 128, 0.1, 1, torch.float16),
            (3, 1024, 4, 128, 0.1, 10, torch.float16),
            (3, 1024, 4, 256, 0.1, 0.1, torch.float16),
            (4, 2048, 4, 64, 0.1, 0.1, torch.float16),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    USE_G = gate_logit_normalizer > 0
    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    g = F.logsigmoid(torch.randn((B, T, H), dtype=dtype, device=device)) if USE_G else None
    g = (g / gate_logit_normalizer).requires_grad_(True) if USE_G else None
    do = torch.randn_like(v)

    ref, _ = fused_recurrent_simple_gla(q=q, k=k, v=v, g=g, scale=scale, output_final_state=True)
    _, ref_A = naive_parallel_simple_gla(q=q, k=k, v=v, g=g, scale=scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if USE_G:
        ref_dg, g.grad = g.grad.clone(), None

    tri, tri_A = parallel_simple_gla(q=q, k=k, v=v, g=g, scale=scale, output_attentions=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    if USE_G:
        tri_dg, g.grad = g.grad.clone(), None
    assert_close('o', ref, tri, 0.005)
    assert_close('A', ref_A, tri_A, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    if USE_G:
        assert_close('dg', ref_dg, tri_dg, 0.015)


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
def test_parallel_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(torch.randn((1, T, H), dtype=dtype, device=device)).requires_grad_()
    do = torch.randn_like(v)

    ref, _ = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = parallel_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close('o', ref, tri, 0.004)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)


@pytest.mark.parametrize(
    ('vary_A', 'dtype'),
    [
        pytest.param(True, torch.float, id=f'vary_A{True}-dtype{torch.float}'),
        pytest.param(False, torch.float, id=f'vary_A{False}-dtype{torch.float}'),
        pytest.param(True, torch.float16, id=f'vary_A{True}-dtype{torch.float16}'),
        pytest.param(False, torch.float16, id=f'vary_A{False}-dtype{torch.float16}'),
    ],
)
def test_simple_gla_to_mamba2(vary_A, dtype):
    try:
        from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        pytest.skip('mamba_ssm is not installed.')
    torch.manual_seed(42)

    # Dimensions, Denoted (B, T, Q, D, P) in Mamba2 paper
    batch, seq_len, chunk_size, dim, headdim = 2, 512, 8, 64, 16
    n_heads = dim // headdim  # (H) in the paper
    ngroups = n_heads  # (G) in the paper; NOTE: do not use group-query here
    dstate = 64  # (N) in the paper
    atol = 5e-4 if dtype == torch.float else 1e-2

    x = 0.1 * torch.randn(batch, seq_len, n_heads, headdim, dtype=dtype, device=device)
    dt = torch.ones(batch, seq_len, n_heads, dtype=dtype, device=device)  # dt=1 can be ignored

    if vary_A:
        A = -0.1 * torch.rand(1, seq_len, n_heads, dtype=dtype, device=device)
    else:  # constant A for all position
        A = -0.1 * torch.rand(n_heads, dtype=dtype, device=device)

    B = 0.1 * torch.randn(batch, seq_len, ngroups, dstate, dtype=dtype, device=device)
    C = 0.1 * torch.randn(batch, seq_len, ngroups, dstate, dtype=dtype, device=device)

    y_ssd, final_ssd = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C, chunk_size)

    if not vary_A:
        # NOTE: fused kernel does not support varying A with time
        y_fuse, final_fuse = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, return_final_states=True)
        assert y_ssd.allclose(y_fuse, 0, atol), f'y diff: {torch.abs(y_ssd - y_fuse).max()}'
        # fused kernel upcasts state to float32
        # https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/ops/triton/ssd_combined.py#L650
        final_fuse = final_fuse.to(dtype)
        assert final_ssd.allclose(final_fuse, 0, atol), f'final diff: {torch.abs(final_ssd - final_fuse).max()}'

    # mapping inputs Mamba2 -> FLA
    # FLA Now use head_first = False, therefore there is no need to transpose inputs
    q = C
    k = B
    v = x
    g = (A * dt)

    # mapping outputs Mamba2 -> FLA
    y_rearrange = y_ssd
    final_rearrange = final_ssd.transpose(2, 3)

    # comparing output results between FLA kernel and Mamba2 kernel
    # final_gla_fuse :[N, H, K, V]
    outputs_gla_fuse, final_gla_fuse = chunk_simple_gla(q, k, v, g, scale=1.0, output_final_state=True)
    assert y_rearrange.allclose(outputs_gla_fuse, 0, atol), f'y diff: {torch.abs(y_rearrange - outputs_gla_fuse).max()}'
    final_gla_fuse = final_gla_fuse.to(dtype)  # states hard-coded to float32 in FLA kernel
    assert final_rearrange.allclose(final_gla_fuse, 0, atol), f'final diff: {torch.abs(final_ssd - final_gla_fuse).max()}'
