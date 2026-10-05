# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.attn.backends.gluon import AttnGluonBackend
from fla.ops.attn.naive import naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close, device, get_device_capability

requires_gluon = pytest.mark.skipif(
    os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') == '1'
    or not AttnGluonBackend.is_available() or get_device_capability()[0] not in (9, 10),
    reason='Gluon attention requires backend dispatch, Triton >= 3.5.1, and compute capability 9.x or 10.x',
)


@pytest.fixture
def gluon_route(monkeypatch):
    monkeypatch.setenv('FLA_ATTN_GLUON', '1')
    monkeypatch.setenv('FLA_TILELANG', '0')
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    calls = {'fwd': 0, 'bwd': 0}
    for name, key in [('parallel_attn_fwd', 'fwd'), ('parallel_attn_bwd', 'bwd')]:
        original = getattr(AttnGluonBackend, name)

        def wrapped(self, *args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(AttnGluonBackend, name, wrapped)
    yield calls
    torch.backends.cuda.matmul.allow_tf32 = previous


def _compare(B, T, H, HQ, K, V, dtype, use_g, use_sink, window, varlen, supplied_indices, strided=False):
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.empty(B, T, HQ, device=device, dtype=torch.float32).uniform_(-0.1, -0.01) if use_g else None
    sink = torch.randn(HQ, device=device, dtype=torch.float32) if use_sink else None
    tensors = [q, k, v, g, sink]
    if strided:
        tensors = [torch.stack((x, x), dim=-1)[..., 0] if x is not None else None for x in tensors]
    tensors = [x.detach().requires_grad_() if x is not None else None for x in tensors]
    refs = [x.detach().float().requires_grad_() if x is not None else None for x in tensors]
    q, k, v, g, sink = tensors
    rq, rk, rv, rg, rs = refs
    scale = K ** -0.5
    cu = None
    indices = None
    if varlen:
        offsets = [0, 15, 79, 79, T]
        cu = torch.tensor(offsets, device=device, dtype=torch.int32)
        if supplied_indices:
            indices = prepare_chunk_indices(cu, 128)
        outputs = []
        for left, right in zip(offsets[:-1], offsets[1:]):
            if left == right:
                continue
            out, _ = naive_parallel_attn(
                q=rq[:, left:right],
                k=rk[:, left:right],
                v=rv[:, left:right],
                g=rg[:, left:right] if rg is not None else None,
                sink_bias=rs,
                scale=scale,
                window_size=window,
            )
            outputs.append(out)
        ref = torch.cat(outputs, dim=1)
    else:
        ref, _ = naive_parallel_attn(q=rq, k=rk, v=rv, g=rg, sink_bias=rs, scale=scale, window_size=window)
    actual = parallel_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        sink_bias=sink,
        scale=scale,
        window_size=window,
        cu_seqlens=cu,
        chunk_indices=indices,
    )
    do = torch.randn_like(actual)
    actual.backward(do)
    ref.backward(do.float())
    assert torch.isfinite(actual).all()
    assert_close('o', ref, actual, 0.005)
    for name, x, r in zip(('dq', 'dk', 'dv', 'dg', 'dsink'), tensors, refs):
        if x is not None:
            assert torch.isfinite(x.grad).all(), name
            assert_close(name, r.grad, x.grad, 0.005)


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('use_g', [False, True], ids=['no-gate', 'gate'])
@pytest.mark.parametrize('use_sink', [False, True], ids=['no-sink', 'sink'])
@pytest.mark.parametrize('window', [None, 17], ids=['full', 'window'])
@pytest.mark.parametrize('HQ', [2, 8], ids=['mha', 'gqa'])
def test_parallel(gluon_route, dtype, varlen, use_g, use_sink, window, HQ):
    _compare(
        B=1 if varlen else 2,
        T=257,
        H=2,
        HQ=HQ,
        K=64,
        V=64,
        dtype=dtype,
        use_g=use_g,
        use_sink=use_sink,
        window=window,
        varlen=varlen,
        supplied_indices=False,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('use_tma', [False, True])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('dim', [128, 512])
def test_parallel_copy_path(gluon_route, monkeypatch, use_tma, varlen, dim):
    from fla.ops.attn.backends.gluon import parallel as implementation

    monkeypatch.setattr(implementation, 'IS_TMA_SUPPORTED', use_tma)
    _compare(
        B=1,
        T=257,
        H=2,
        HQ=4,
        K=dim,
        V=dim,
        dtype=torch.bfloat16,
        use_g=True,
        use_sink=True,
        window=127,
        varlen=varlen,
        supplied_indices=varlen,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('chunk_size', [32, 64])
def test_backward_supplied_indices(gluon_route, chunk_size):
    from fla.ops.attn.parallel import parallel_attn_bwd, parallel_attn_fwd

    torch.manual_seed(42)
    tensors = [torch.randn(1, 257, 2, 128, device=device, dtype=torch.float16) for _ in range(3)]
    refs = [x.float().requires_grad_() for x in tensors]
    cu = torch.tensor([0, 65, 257], device=device, dtype=torch.int32)
    # the original backend interprets supplied indices in 128-row chunks regardless of the legacy chunk_size argument.
    indices = prepare_chunk_indices(cu, 128)
    scale = 128 ** -0.5
    o, lse = parallel_attn_fwd(
        q=tensors[0],
        k=tensors[1],
        v=tensors[2],
        g_cumsum=None,
        sink_bias=None,
        scale=scale,
        cu_seqlens=cu,
        chunk_indices=indices,
    )
    do = torch.randn_like(o)
    grads = parallel_attn_bwd(
        q=tensors[0],
        k=tensors[1],
        v=tensors[2],
        o=o,
        g_cumsum=None,
        lse=lse,
        do=do,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu,
        chunk_indices=indices,
    )
    outputs = []
    for left, right in [(0, 65), (65, 257)]:
        ref, _ = naive_parallel_attn(
            q=refs[0][:, left:right],
            k=refs[1][:, left:right],
            v=refs[2][:, left:right],
            scale=scale,
        )
        outputs.append(ref)
    ref = torch.cat(outputs, dim=1)
    expected = torch.autograd.grad(ref, refs, do.float())
    for name, actual, reference in zip(('dq', 'dk', 'dv'), grads[:3], expected):
        assert_close(name, reference, actual, 0.005)
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('K', 'V', 'T', 'varlen', 'supplied_indices'),
    [
        (1, 3, 111, False, False),
        (65, 127, 257, True, True),
        (16, 16, 111, False, False),
        (60, 100, 127, False, False),
        (100, 60, 257, True, False),
        (128, 128, 511, False, False),
        (192, 128, 257, True, True),
        (256, 256, 257, False, False),
        (256, 256, 257, True, True),
        (128, 320, 257, True, False),
        (256, 512, 257, False, False),
        (320, 128, 257, True, True),
        (512, 256, 257, False, False),
        (512, 512, 257, False, False),
        (512, 512, 257, True, True),
    ],
)
def test_parallel_dimensions(gluon_route, dtype, K, V, T, varlen, supplied_indices):
    _compare(
        B=1,
        T=T,
        H=1,
        HQ=4,
        K=K,
        V=V,
        dtype=dtype,
        use_g=True,
        use_sink=True,
        window=65,
        varlen=varlen,
        supplied_indices=supplied_indices,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('dim', [256, 512])
def test_parallel_long_large_dimensions(gluon_route, dtype, varlen, dim):
    _compare(
        B=1,
        T=2048,
        H=1,
        HQ=2,
        K=dim,
        V=dim,
        dtype=dtype,
        use_g=False,
        use_sink=False,
        window=None,
        varlen=varlen,
        supplied_indices=False,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('window', [0, 1, 63, 64, 65, 1024])
@pytest.mark.parametrize('varlen', [False, True])
def test_parallel_window_boundary(gluon_route, window, varlen):
    _compare(
        B=1,
        T=257,
        H=1,
        HQ=4,
        K=128,
        V=128,
        dtype=torch.float16,
        use_g=True,
        use_sink=True,
        window=window,
        varlen=varlen,
        supplied_indices=False,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
def test_parallel_strided(gluon_route):
    _compare(
        B=1,
        T=257,
        H=1,
        HQ=4,
        K=100,
        V=64,
        dtype=torch.float16,
        use_g=True,
        use_sink=True,
        window=None,
        varlen=True,
        supplied_indices=False,
        strided=True,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('varlen', [False, True])
def test_parallel_route_parity(gluon_route, monkeypatch, varlen):
    torch.manual_seed(42)
    q = torch.randn(1, 257, 4, 128, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 257, 1, 128, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    g = torch.empty(1, 257, 4, device=device).uniform_(-0.1, -0.01).requires_grad_()
    sink = torch.randn(4, device=device, requires_grad=True)
    cu = torch.tensor([0, 63, 257], device=device, dtype=torch.int32) if varlen else None
    do = torch.randn_like(q)
    kwargs = dict(q=q, k=k, v=v, g=g, sink_bias=sink, window_size=65, cu_seqlens=cu)
    results = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('FLA_ATTN_GLUON', enabled)
        out = parallel_attn(**kwargs)
        grads = torch.autograd.grad(out, (q, k, v, g, sink), do)
        results.append((out, *grads))
    for name, original, candidate in zip(('o', 'dq', 'dk', 'dv', 'dg', 'dsink'), *results):
        assert_close(name, original, candidate, 0.005)
    explicit = parallel_attn(**kwargs, scale=128 ** -0.5)
    torch.testing.assert_close(explicit, results[1][0], rtol=0, atol=0)
    assert gluon_route == {'fwd': 2, 'bwd': 1}


def test_backend_opt_in(monkeypatch):
    monkeypatch.delenv('FLA_ATTN_GLUON', raising=False)
    assert not AttnGluonBackend.is_enabled()
    monkeypatch.setenv('FLA_ATTN_GLUON', '1')
    assert AttnGluonBackend.is_enabled()


def test_backend_rejects_cpu():
    q = torch.empty(1, 1, 1, 64, dtype=torch.float16)
    accepted, reason = AttnGluonBackend().parallel_attn_fwd_verifier(q=q, k=q, v=q, g_cumsum=None, sink_bias=None, scale=0.125)
    assert not accepted
    assert 'compute capability' in reason


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_backend_rejects_dtype(dtype):
    q = torch.empty(1, 1, 1, 64, device=device, dtype=dtype)
    accepted, reason = AttnGluonBackend().parallel_attn_fwd_verifier(q=q, k=q, v=q, g_cumsum=None, sink_bias=None, scale=0.125)
    assert not accepted
    assert 'matching fp16 or bf16' in reason


@requires_gluon
@pytest.mark.parametrize(('K', 'V'), [(0, 64), (64, 0), (513, 64), (64, 513)])
def test_backend_rejects_dimension(K, V):
    q = torch.empty(1, 1, 1, K, device=device, dtype=torch.float16)
    v = torch.empty(1, 1, 1, V, device=device, dtype=torch.float16)
    accepted, reason = AttnGluonBackend().parallel_attn_fwd_verifier(q=q, k=q, v=v, g_cumsum=None, sink_bias=None, scale=0.125)
    assert not accepted
    assert 'dimensions' in reason


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(('K', 'V'), [(1, 3), (65, 127), (64, 64), (100, 60), (256, 512), (512, 256), (512, 512)])
@pytest.mark.parametrize('window', [None, 0, 65])
def test_decoding_split(monkeypatch, dtype, K, V, window):
    from fla.ops.attn.decoding import attn_decoding_one_step
    from fla.ops.attn.naive import naive_attn_decoding

    monkeypatch.setenv('FLA_ATTN_GLUON', '1')
    monkeypatch.setenv('FLA_TILELANG', '0')
    calls = []
    original = AttnGluonBackend.attn_decoding_fwd

    def wrapped(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AttnGluonBackend, 'attn_decoding_fwd', wrapped)
    torch.manual_seed(42)
    q = torch.randn(1, 4, 4, K, device=device, dtype=dtype)
    k = torch.randn(1, 2311, 1, K, device=device, dtype=dtype)
    v = torch.randn(1, 2311, 1, V, device=device, dtype=dtype)
    g = torch.empty(1, 2311, 4, device=device, dtype=dtype).uniform_(-0.1, -0.01)
    sink = torch.randn(4, device=device)
    cu = torch.tensor([0, 0, 257, 258, 2311], device=device, dtype=torch.int32)
    kwargs = dict(cu_seqlens=cu, do_gate_scale=True, window_size=window, sink_bias=sink)
    actual = attn_decoding_one_step(q=q, k=k, v=v, g=g, **kwargs)
    ref = naive_attn_decoding(q=q.float(), k=k.float(), v=v.float(), g=g.float(), **kwargs)
    assert torch.isfinite(actual).all()
    assert_close('o', ref, actual, 0.01)
    explicit = attn_decoding_one_step(q=q, k=k, v=v, g=g, scale=K ** -0.5, **kwargs)
    torch.testing.assert_close(actual, explicit, rtol=0, atol=0)
    assert len(calls) == 2


@requires_gluon
def test_backend_hardware_fallback(gluon_route, monkeypatch):
    monkeypatch.setattr('fla.ops.attn.backends.gluon.get_device_capability', lambda *args: (8, 0))
    _compare(
        B=1,
        T=127,
        H=1,
        HQ=1,
        K=64,
        V=64,
        dtype=torch.float16,
        use_g=False,
        use_sink=False,
        window=None,
        varlen=False,
        supplied_indices=False,
    )
    assert gluon_route == {'fwd': 0, 'bwd': 0}


@requires_gluon
def test_parallel_unaligned_storage(gluon_route):
    torch.manual_seed(42)
    tensors = [torch.randn(1 + 127 * 64, device=device, dtype=torch.float16)[1:] for _ in range(3)]
    tensors = [x.view(1, 127, 1, 64).requires_grad_() for x in tensors]
    refs = [x.detach().float().requires_grad_() for x in tensors]
    actual = parallel_attn(q=tensors[0], k=tensors[1], v=tensors[2])
    ref, _ = naive_parallel_attn(q=refs[0], k=refs[1], v=refs[2])
    do = torch.randn_like(actual)
    actual.backward(do)
    ref.backward(do.float())
    assert_close('o', ref, actual, 0.005)
    for name, x, r in zip(('dq', 'dk', 'dv'), tensors, refs):
        assert_close(name, r.grad, x.grad, 0.005)
    assert gluon_route == {'fwd': 1, 'bwd': 1}
