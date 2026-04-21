# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.attn.naive import naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.utils import assert_close, check_shared_mem, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-scale{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1.0),
            (3, 111, 2, 2, 100, 1.0),
            (3, 1024, 2, 8, 60, 0.1),
            (3, 1024, 2, 8, 128, 0.1),
            (4, 2048, 2, 8, 64, 0.1),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    scale: float,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device)

    ref, _ = naive_parallel_attn(q=q.float(), k=k.float(), v=v.float(), scale=scale)
    ref = ref.to(q.dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_attn(q=q, k=k, v=v, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-scale{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1.0),
            (3, 111, 2, 2, 100, 1.0),
            (3, 1024, 2, 8, 60, 0.1),
        ]
    ],
)
def test_parallel_with_g(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    scale: float,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    g = torch.randn((B, T, HQ), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device)

    ref, _ = naive_parallel_attn(q=q.float(), k=k.float(), v=v.float(), g=g.float(), scale=scale)
    ref = ref.to(q.dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri = parallel_attn(q=q, k=k, v=v, g=g, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
    assert_close("dg", ref_dg, tri_dg, 0.005)


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-cu_seqlens{}".format(*test))
        for test in [
            (2, 2, 64, [0, 15]),
            (2, 8, 64, [0, 256, 500, 1000]),
            (2, 2, 100, [0, 15, 100, 300, 1200, 2000]),
        ]
    ],
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    cu_seqlens: list[int],
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens_th = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = q.new_empty(1, T, HQ, D)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        ref[:, bos:eos], _ = naive_parallel_attn(
            q=q[:, bos:eos].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
        )
    ref = ref.to(dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_attn(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens_th,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq.squeeze(), tri_dq.squeeze(), 0.005)
    assert_close("dk", ref_dk.squeeze(), tri_dk.squeeze(), 0.005)
    assert_close("dv", ref_dv.squeeze(), tri_dv.squeeze(), 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 16),
            (3, 111, 2, 2, 100, 32),
            (3, 1024, 2, 8, 128, 64),
        ]
    ],
)
def test_parallel_swa(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    W: int,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device)

    ref, _ = naive_parallel_attn(q=q.float(), k=k.float(), v=v.float(), window_size=W)
    ref = ref.to(q.dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_attn(q=q, k=k, v=v, window_size=W)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64),
            (3, 111, 2, 2, 100),
            (3, 1024, 2, 8, 128),
        ]
    ],
)
def test_parallel_sink(B: int, T: int, H: int, HQ: int, D: int):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=torch.float16, device=device).requires_grad_(True)
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=torch.float16, device=device)

    ref, _ = naive_parallel_attn(
        q=q.float(), k=k.float(), v=v.float(), sink_bias=sink_bias)
    ref = ref.to(q.dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dsink, sink_bias.grad = sink_bias.grad.clone(), None

    tri = parallel_attn(q=q, k=k, v=v, sink_bias=sink_bias)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dsink, sink_bias.grad = sink_bias.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
    assert_close("dsink", ref_dsink, tri_dsink, 0.005)


@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'W', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-W{}-cu_seqlens{}".format(*test))
        for test in [
            (2, 2, 64, 16, [0, 111]),
            (2, 8, 100, 32, [0, 256, 500, 1000]),
        ]
    ],
)
def test_parallel_swa_varlen(
    H: int,
    HQ: int,
    D: int,
    W: int,
    cu_seqlens: list[int],
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    T = cu_seqlens[-1]
    cu_seqlens_th = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    # per-sequence naive reference
    refs_o, refs_dq, refs_dk, refs_dv = [], [], [], []
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        qi = q[:, s:e].detach().float().requires_grad_(True)
        ki = k[:, s:e].detach().float().requires_grad_(True)
        vi = v[:, s:e].detach().float().requires_grad_(True)
        oi, _ = naive_parallel_attn(q=qi, k=ki, v=vi, window_size=W)
        oi = oi.to(dtype)
        oi.backward(do[:, s:e])
        refs_o.append(oi)
        refs_dq.append(qi.grad.to(dtype))
        refs_dk.append(ki.grad.to(dtype))
        refs_dv.append(vi.grad.to(dtype))
    ref = torch.cat(refs_o, dim=1)
    ref_dq = torch.cat(refs_dq, dim=1)
    ref_dk = torch.cat(refs_dk, dim=1)
    ref_dv = torch.cat(refs_dv, dim=1)

    tri = parallel_attn(q=q, k=k, v=v, window_size=W, cu_seqlens=cu_seqlens_th)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parametrize('D', [64, 100, 128])
@pytest.mark.parametrize('cu_seqlens', [
    [0, 15, 30],        # two short seqs
    [0, 200, 400],      # two medium seqs
])
def test_varlen_d_debug(cu_seqlens, D):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    H, HQ = 2, 2
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()

    ref = q.new_empty(1, T, HQ, D)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        ref[:, bos:eos], _ = naive_parallel_attn(
            q=q[:, bos:eos].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
        )
    ref = ref.to(dtype)
    tri = parallel_attn(q=q, k=k, v=v, cu_seqlens=cu)
    assert_close(" o", ref, tri, 0.005)
