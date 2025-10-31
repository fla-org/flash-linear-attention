
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn
from fla.ops.hgrn.naive import naive_recurrent_hgrn
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 500, torch.float),
            (2, 1024, 500, torch.float),
            (2, 1024, 512, torch.float),
            (2, 1024, 1000, torch.float),
            (4, 2048, 2048, torch.float),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn((B, T, D), dtype=dtype, device=device)
    g = torch.randn((B, T, D), dtype=dtype, device=device)
    h0 = torch.randn_like(x[:, 0])
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g, h0 = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g, h0))

    do = torch.randn_like(x)
    dht = torch.randn_like(h0)
    ref, ref_ht = naive_recurrent_hgrn(x, g, h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (500, [0, 15], torch.float),
            (512, [0, 256, 500, 1000], torch.float),
            (1000, [0, 15, 100, 300, 1200, 2000], torch.float),
            (2048, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
def test_fused_recurrent_varlen(
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    x = torch.randn((1, T, D), dtype=dtype, device=device)
    g = torch.randn((1, T, D), dtype=dtype, device=device)
    h0 = torch.randn(N, D, dtype=dtype, device=device)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g, h0 = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g, h0))

    do = torch.randn_like(x)
    dht = torch.randn_like(h0)
    refs, ref_hts = [], []
    for i in range(N):
        ref, ref_ht = naive_recurrent_hgrn(
            x[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            h0[i:i+1],
            output_final_state=True,
        )
        refs.append(ref)
        ref_hts.append(ref_ht)
    ref = torch.cat(refs, 1)
    ref_ht = torch.cat(ref_hts, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = fused_recurrent_hgrn(x, g, h0, output_final_state=True, cu_seqlens=cu_seqlens)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 500, torch.float16),
            (2, 500, 1000, torch.float16),
            (2, 1000, 1024, torch.float16),
            (4, 2048, 2048, torch.float16),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x = torch.randn((B, T, D), dtype=dtype, device=device)
    g = torch.randn((B, T, D), dtype=dtype, device=device)
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))

    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, 0])
    ref, _ = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, _ = chunk_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close('o', ref, tri, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
