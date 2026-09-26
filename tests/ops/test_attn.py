# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.naive import naive_attn_decoding, naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.utils import assert_close, check_shared_mem, device


@pytest.mark.parametrize(
    "op",
    [naive_parallel_attn, parallel_attn, naive_attn_decoding, attn_decoding_one_step],
    ids=["naive", "parallel", "naive_decode", "decode"],
)
@pytest.mark.parametrize(("HQ", "H"), [(3, 2), (1, 2), (2, 0)], ids=["remainder", "fewer-query-heads", "zero-kv-heads"])
def test_rejects_invalid_gqa_head_counts(op, HQ, H):
    q = torch.empty(1, 1, HQ, 16, dtype=torch.float16)
    k = torch.empty(1, 1, H, 16, dtype=torch.float16)
    v = torch.empty_like(k)

    kwargs = {}
    if op in (naive_attn_decoding, attn_decoding_one_step):
        kwargs['cu_seqlens'] = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="must be divisible"):
        op(q=q, k=k, v=v, **kwargs)


@pytest.mark.parametrize(
    ('H', 'HQ', 'K', 'V', 'W', 'use_g', 'do_gate_scale', 'use_sink', 'dtype'),
    [
        pytest.param(*test, id="H{}-HQ{}-K{}-V{}-W{}-g{}-scale{}-sink{}-{}".format(*test))
        for test in [
            (2, 2, 64, 64, None, False, False, False, torch.float16),
            (2, 8, 64, 100, None, True, True, True, torch.float16),
            (2, 2, 64, 64, 0, False, False, False, torch.float16),
            (2, 8, 64, 100, 0, True, True, True, torch.float16),
            (2, 8, 64, 100, 1, True, False, True, torch.float16),
            (2, 2, 100, 64, 17, False, False, False, torch.float16),
            (2, 8, 64, 320, 63, False, False, True, torch.float16),
            (2, 8, 64, 100, 64, True, False, False, torch.float16),
            (2, 8, 64, 100, 65, True, True, True, torch.float16),
            (2, 8, 64, 100, 1024, True, True, True, torch.float16),
            (2, 8, 64, 100, 17, True, True, True, torch.bfloat16),
            (2, 2, 64, 64, -1, False, False, False, torch.float16),
        ]
    ],
)
def test_decoding(
    H: int,
    HQ: int,
    K: int,
    V: int,
    W: int | None,
    use_g: bool,
    do_gate_scale: bool,
    use_sink: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    lengths = [0, 15, 64, 127]
    B, T = len(lengths), sum(lengths)
    q = torch.randn(1, B, HQ, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)
    g = torch.empty(1, T, HQ, dtype=dtype, device=device).uniform_(-0.1, -0.01) if use_g else None
    sink_bias = torch.randn(HQ, dtype=torch.float32, device=device) if use_sink else None
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    kwargs = dict(scale=0.1, cu_seqlens=cu_seqlens, do_gate_scale=do_gate_scale, window_size=W, sink_bias=sink_bias)

    if W is not None and W < 0:
        for implementation in (naive_attn_decoding, attn_decoding_one_step):
            with pytest.raises(ValueError, match="window_size must be nonnegative"):
                implementation(q=q, k=k, v=v, g=g, **kwargs)
        return

    ref = naive_attn_decoding(q=q.float(), k=k.float(), v=v.float(), g=g.float() if use_g else None, **kwargs).to(dtype)
    tri = attn_decoding_one_step(q=q, k=k, v=v, g=g, **kwargs)
    assert torch.isfinite(tri).all()
    assert_close("o", ref, tri, 0.01)

    if W is None:
        del kwargs['window_size']
        default = attn_decoding_one_step(q=q, k=k, v=v, g=g, **kwargs)
        torch.testing.assert_close(default, tri, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'K', 'V', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-K{}-V{}-scale{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 64, 1.0),
            (3, 111, 2, 2, 100, 100, 1.0),
            (3, 1024, 2, 8, 60, 60, 0.1),
            (3, 1024, 2, 8, 128, 128, 0.1),
            (4, 2048, 2, 8, 64, 64, 0.1),
            (2, 127, 2, 8, 64, 100, 0.1),
            (1, 63, 2, 2, 100, 64, 0.1),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
    scale: float,
):
    if not check_shared_mem('hopper') and max(K, V) > 128:
        pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    q = torch.randn((B, T, HQ, K), dtype=torch.float16, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, K), dtype=torch.float16, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, V), dtype=torch.float16, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, V), dtype=torch.float16, device=device)

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


def test_parallel_bwd_full_value_reduction(monkeypatch):
    """Regression test: the backward must not split the value dim (NV == 1).

    On low-shared-memory GPUs (e.g. consumer RTX cards) the backward used to cap BV <= 64,
    so any head dim > 64 split V across programs and produced silently wrong dq/dk. We force
    that low-smem branch here so the bug is caught on any GPU, not just consumer cards. The
    same setup forces a split forward and validates its LSE through backward.
    """
    # Force the low-shared-memory branch regardless of the actual device.
    monkeypatch.setattr("fla.ops.attn.parallel.check_shared_mem", lambda *args, **kwargs: False)

    torch.manual_seed(42)
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    # D=128 (> 64) is what triggered the bug: BV was capped at 64, splitting V into NV=2 blocks.
    B, T, H, HQ, D, scale = 2, 256, 2, 2, 128, 0.1
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

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, q.grad, 0.005)
    assert_close("dk", ref_dk, k.grad, 0.005)
    assert_close("dv", ref_dv, v.grad, 0.005)


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
    ('H', 'HQ', 'K', 'V', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-K{}-V{}-cu_seqlens{}".format(*test))
        for test in [
            (2, 2, 64, 64, [0, 15]),
            (2, 8, 64, 64, [0, 256, 500, 1000]),
            (2, 2, 100, 100, [0, 15, 100, 300, 1200, 2000]),
            (2, 8, 64, 100, [0, 15, 142, 270]),
            (2, 2, 100, 64, [0, 15, 142, 270]),
        ]
    ],
)
@pytest.mark.smoke
def test_parallel_varlen(
    H: int,
    HQ: int,
    K: int,
    V: int,
    cu_seqlens: list[int],
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens_th = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    dtype = torch.float16

    q = torch.randn((1, T, HQ, K), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, K), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, V), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, V), dtype=dtype, device=device)

    ref = q.new_empty(1, T, HQ, V)
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
    ('B', 'T', 'H', 'HQ', 'K', 'V', 'scale', 'window_size', 'cu_seqlens', 'use_g', 'tol'),
    [
        pytest.param(1, 63, 1, 1, 64, 64, None, None, None, False, (0.005, 0.005), id="mha"),
        pytest.param(3, 111, 2, 2, 100, 100, None, None, None, False, (0.005, 0.005), id="mha-K100"),
        pytest.param(3, 1024, 2, 8, 128, 128, None, None, None, False, (0.005, 0.005), id="gqa-K128"),
        pytest.param(2, 127, 2, 8, 64, 100, None, None, None, False, (0.005, 0.005), id="gqa-K64-V100"),
        pytest.param(1, 63, 2, 2, 100, 64, None, None, None, False, (0.005, 0.005), id="mha-K100-V64"),
        pytest.param(2, 192, 2, 8, 64, 64, 0.1, None, None, False, (0.01, 0.02), id="full", marks=pytest.mark.smoke),
        pytest.param(2, 192, 2, 8, 64, 64, 0.1, 64, None, False, (0.01, 0.02), id="swa", marks=pytest.mark.smoke),
        pytest.param(
            1, 300, 2, 8, 64, 64, 0.1, 64, [0, 97, 173, 300], False, (0.01, 0.02),
            id="varlen-swa",
            marks=pytest.mark.smoke,
        ),
        pytest.param(2, 96, 2, 8, 64, 64, 0.1, 0, None, False, (0.01, 0.02), id="empty-row"),
        pytest.param(2, 192, 2, 8, 64, 64, 0.1, None, None, True, (0.01, 0.02), id="gate-full"),
        pytest.param(2, 192, 2, 8, 64, 64, 0.1, 64, None, True, (0.01, 0.02), id="gate-swa"),
        pytest.param(1, 300, 2, 8, 64, 64, 0.1, 64, [0, 97, 173, 300], True, (0.01, 0.02), id="gate-varlen-swa"),
    ],
)
def test_parallel_sink(B, T, H, HQ, K, V, scale, window_size, cu_seqlens, use_g, tol, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    dtype = torch.float16
    q = torch.randn((B, T, HQ, K), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, K), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, V), dtype=dtype, device=device).requires_grad_(True)
    g = torch.empty((B, T, HQ), dtype=dtype, device=device).uniform_(-0.1, -0.01).requires_grad_(True) if use_g else None
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, V), dtype=dtype, device=device)
    inputs = (q, k, v, sink_bias) if g is None else (q, k, v, sink_bias, g)
    names = ('dq', 'dk', 'dv', 'dsink') if g is None else ('dq', 'dk', 'dv', 'dsink', 'dg')

    boundaries = [0, T] if cu_seqlens is None else cu_seqlens
    outputs = []
    for bos, eos in zip(boundaries[:-1], boundaries[1:], strict=True):
        o, _ = naive_parallel_attn(
            q=q[:, bos:eos].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
            g=g[:, bos:eos].float() if g is not None else None,
            scale=scale,
            window_size=window_size,
            sink_bias=sink_bias,
        )
        outputs.append(o)
    ref = torch.cat(outputs, dim=1).to(dtype)
    ref_grads = torch.autograd.grad(ref, inputs, do)

    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device) if cu_seqlens is not None else None
    tri = parallel_attn(q=q, k=k, v=v, g=g, scale=scale, window_size=window_size, cu_seqlens=cu_seqlens, sink_bias=sink_bias)
    tri_grads = torch.autograd.grad(tri, inputs, do)

    assert_close('o', ref, tri, tol[0])
    for name, ref_grad, tri_grad in zip(names, ref_grads, tri_grads, strict=True):
        assert_close(name, ref_grad, tri_grad, tol[1])


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


@pytest.mark.parametrize(
    ('T', 'window_size'),
    [pytest.param(96, None, id='full'), pytest.param(96, 64, id='swa'), pytest.param(48, 0, id='empty-row')],
)
def test_naive_parallel_sink(T, window_size):
    """Check the reference against GPT-OSS's explicit sink-logit softmax, including gradients."""
    torch.manual_seed(42)
    B, H, HQ, D, scale = 2, 2, 8, 64, 0.1
    q = torch.randn((B, T, HQ, D), dtype=torch.float64, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=torch.float64, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=torch.float64, device=device).requires_grad_(True)
    sink_bias = torch.randn((HQ,), dtype=torch.float64, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=torch.float64, device=device)
    inputs = (q, k, v, sink_bias)

    query = q.transpose(1, 2)
    key = k.repeat_interleave(HQ // H, dim=2).transpose(1, 2)
    value = v.repeat_interleave(HQ // H, dim=2).transpose(1, 2)
    logits = query @ key.transpose(-2, -1) * scale
    row = torch.arange(T, device=device)[:, None]
    col = torch.arange(T, device=device)[None, :]
    mask = col > row
    if window_size is not None:
        mask = mask | (row - col >= window_size)
    logits = logits.masked_fill(mask, float('-inf'))
    logits = torch.cat((logits, sink_bias.view(1, HQ, 1, 1).expand(B, HQ, T, 1)), dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    ref = (logits.softmax(dim=-1)[..., :-1] @ value).transpose(1, 2).contiguous()
    ref_grads = torch.autograd.grad(ref, inputs, do)

    naive, _ = naive_parallel_attn(q=q, k=k, v=v, scale=scale, window_size=window_size, sink_bias=sink_bias)
    naive_grads = torch.autograd.grad(naive, inputs, do)

    assert_close('o', ref, naive, 1e-10, err_atol=1e-10)
    for name, ref_grad, naive_grad in zip(('dq', 'dk', 'dv', 'dsink'), ref_grads, naive_grads, strict=True):
        assert_close(name, ref_grad, naive_grad, 1e-10, err_atol=1e-10)


@pytest.mark.parametrize(
    ('HQ', 'V', 'lengths', 'use_sink', 'use_g', 'do_gate_scale', 'scale'),
    [
        pytest.param(8, 64, [128, 128, 128], True, False, False, 0.1, id='sink'),
        pytest.param(4, 320, [64, 64], False, False, False, None, id='value-split'),
        pytest.param(8, 64, [0, 128, 73], True, False, False, 0.1, id='empty-kv'),
        pytest.param(8, 64, [128, 128, 128], True, True, False, 0.1, id='gate'),
        pytest.param(8, 64, [128, 128, 128], True, True, True, 0.1, id='gate-scale'),
    ],
)
def test_attn_decoding_one_step(HQ, V, lengths, use_sink, use_g, do_gate_scale, scale, monkeypatch):
    torch.manual_seed(42)
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    B, T, H, K = len(lengths), sum(lengths), 2, 64
    dtype = torch.float16
    q = torch.randn((1, B, HQ, K), dtype=dtype, device=device)
    k = torch.randn((1, T, H, K), dtype=dtype, device=device)
    v = torch.randn((1, T, H, V), dtype=dtype, device=device)
    g = torch.empty((1, T, HQ), dtype=dtype, device=device).uniform_(-0.1, -0.01) if use_g else None
    sink_bias = torch.randn((HQ,), dtype=torch.float32, device=device) if use_sink else None
    if use_g:
        sink_bias = sink_bias * 0.7
    cu_seqlens = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)

    ref = naive_attn_decoding(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float() if g is not None else None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
        sink_bias=sink_bias,
    ).to(dtype)
    tri = attn_decoding_one_step(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
        sink_bias=sink_bias,
    )
    assert torch.isfinite(tri).all()
    assert_close('o', ref, tri, 0.01)
