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
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.naive import naive_attn_decoding, naive_parallel_attn
from fla.ops.attn.parallel import parallel_attn
from fla.ops.utils import prepare_chunk_indices
from fla.utils import assert_close, check_shared_mem, device, get_device_capability

requires_gluon = pytest.mark.skipif(
    os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') == '1'
    or not AttnGluonBackend.is_available() or get_device_capability()[0] not in (9, 10),
    reason='Gluon attention requires backend dispatch, Triton >= 3.5.1, and compute capability 9.x or 10.x',
)


@pytest.fixture(autouse=True)
def attention_precision(monkeypatch):
    monkeypatch.setenv('TRITON_F32_DEFAULT', 'ieee')
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture
def gluon_route(monkeypatch):
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_ATTN_GLUON', '1')
    monkeypatch.setenv('FLA_TILELANG', '0')
    calls = {'fwd': 0, 'bwd': 0}
    for name, key in [('parallel_attn_fwd', 'fwd'), ('parallel_attn_bwd', 'bwd')]:
        original = getattr(AttnGluonBackend, name)

        def wrapped(self, *args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(AttnGluonBackend, name, wrapped)
    yield calls


@pytest.fixture
def gluon_decoding_route(monkeypatch):
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_ATTN_GLUON', '1')
    calls = []
    original = AttnGluonBackend.attn_decoding_one_step

    def wrapped(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AttnGluonBackend, 'attn_decoding_one_step', wrapped)
    return calls


def _assert_parallel_close(
    q,
    k,
    v,
    g=None,
    sink_bias=None,
    scale=None,
    window_size=None,
    cu_seqlens=None,
    chunk_indices=None,
    tol=(0.005, 0.005),
):
    inputs = [x.detach().requires_grad_() if x is not None else None for x in (q, k, v, g, sink_bias)]
    refs = [x.detach().float().requires_grad_() if x is not None else None for x in inputs]
    rq, rk, rv, rg, rs = refs
    offsets = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    outputs = []
    for bos, eos in zip(offsets[:-1], offsets[1:]):
        if bos == eos:
            continue
        o, _ = naive_parallel_attn(
            q=rq[:, bos:eos],
            k=rk[:, bos:eos],
            v=rv[:, bos:eos],
            g=rg[:, bos:eos] if rg is not None else None,
            scale=scale,
            window_size=window_size,
            sink_bias=rs,
        )
        outputs.append(o)
    ref = torch.cat(outputs, dim=1)
    q, k, v, g, sink_bias = inputs
    actual = parallel_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        sink_bias=sink_bias,
    )
    do = torch.randn_like(actual)
    actual.backward(do)
    ref.backward(do.float())
    assert torch.isfinite(actual).all()
    assert_close('o', ref, actual, tol[0])
    for name, x, r in zip(('dq', 'dk', 'dv', 'dg', 'dsink'), inputs, refs):
        if x is not None:
            assert torch.isfinite(x.grad).all(), name
            assert_close(name, r.grad, x.grad, tol[1])


@pytest.mark.parametrize(
    "op",
    [naive_parallel_attn, parallel_attn, naive_attn_decoding, attn_decoding_one_step],
    ids=["naive", "parallel", "naive_decode", "decode"],
)
@pytest.mark.parametrize(("HQ", "H"), [(3, 2), (1, 2), (2, 0)], ids=["remainder", "fewer-query-heads", "zero-kv-heads"])
@pytest.mark.parametrize('use_device', [False, True], ids=['cpu', 'device'])
def test_rejects_invalid_gqa_head_counts(op, HQ, H, use_device):
    q = torch.empty(1, 1, HQ, 16, dtype=torch.float16, device=device if use_device else 'cpu')
    k = torch.empty(1, 1, H, 16, dtype=torch.float16, device=device if use_device else 'cpu')
    v = torch.empty_like(k)

    kwargs = {}
    if op in (naive_attn_decoding, attn_decoding_one_step):
        kwargs['cu_seqlens'] = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match="must be divisible"):
        op(q=q, k=k, v=v, **kwargs)


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
def test_parallel(B, T, H, HQ, K, V, scale):
    if not check_shared_mem('hopper') and max(K, V) > 128:
        pytest.skip(reason='Insufficient shared memory')
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float16)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16)
    _assert_parallel_close(q=q, k=k, v=v, scale=scale)


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
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, D, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    g = torch.randn(B, T, HQ, device=device, dtype=torch.float16)
    _assert_parallel_close(q=q, k=k, v=v, g=g, scale=scale)


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('use_g', [False, True], ids=['no-gate', 'gate'])
@pytest.mark.parametrize('use_sink', [False, True], ids=['no-sink', 'sink'])
@pytest.mark.parametrize('window_size', [None, 17], ids=['full', 'window'])
@pytest.mark.parametrize('HQ', [2, 8], ids=['mha', 'gqa'])
def test_parallel_features(gluon_route, dtype, varlen, use_g, use_sink, window_size, HQ):
    torch.manual_seed(42)
    B, T, H, D = 1 if varlen else 2, 257, 2, 64
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    g = torch.empty(B, T, HQ, device=device).uniform_(-0.1, -0.01) if use_g else None
    sink_bias = torch.randn(HQ, device=device) if use_sink else None
    cu_seqlens = torch.tensor([0, 15, 79, 79, T], device=device, dtype=torch.int32) if varlen else None
    _assert_parallel_close(q=q, k=k, v=v, g=g, sink_bias=sink_bias, window_size=window_size, cu_seqlens=cu_seqlens)
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
def test_parallel_head_dim(gluon_route, dtype, K, V, T, varlen, supplied_indices):
    torch.manual_seed(42)
    q = torch.randn(1, T, 4, K, device=device, dtype=dtype)
    k = torch.randn(1, T, 1, K, device=device, dtype=dtype)
    v = torch.randn(1, T, 1, V, device=device, dtype=dtype)
    g = torch.empty(1, T, 4, device=device).uniform_(-0.1, -0.01)
    sink_bias = torch.randn(4, device=device)
    cu_seqlens = torch.tensor([0, 15, 79, 79, T], device=device, dtype=torch.int32) if varlen else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, 128) if supplied_indices else None
    _assert_parallel_close(
        q=q,
        k=k,
        v=v,
        g=g,
        sink_bias=sink_bias,
        window_size=65,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('D', [256, 512])
def test_parallel_head_dim_long_sequence(gluon_route, dtype, varlen, D):
    torch.manual_seed(42)
    q = torch.randn(1, 2048, 2, D, device=device, dtype=dtype)
    k = torch.randn(1, 2048, 1, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu_seqlens = torch.tensor([0, 15, 79, 79, 2048], device=device, dtype=torch.int32) if varlen else None
    _assert_parallel_close(q=q, k=k, v=v, cu_seqlens=cu_seqlens)
    assert gluon_route == {'fwd': 1, 'bwd': 1}


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
            (2, 2, 64, 64, [0, 15, 30]),
            (2, 2, 100, 100, [0, 15, 30]),
            (2, 2, 128, 128, [0, 15, 30]),
            (2, 2, 64, 64, [0, 200, 400]),
            (2, 2, 100, 100, [0, 200, 400]),
            (2, 2, 128, 128, [0, 200, 400]),
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
    q = torch.randn(1, T, HQ, K, device=device, dtype=torch.float16)
    k = torch.randn(1, T, H, K, device=device, dtype=torch.float16)
    v = torch.randn(1, T, H, V, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    _assert_parallel_close(q=q, k=k, v=v, cu_seqlens=cu_seqlens)


@requires_gluon
def test_parallel_varlen_strided(gluon_route):
    torch.manual_seed(42)
    q = torch.randn(1, 257, 4, 100, device=device, dtype=torch.float16)
    k = torch.randn(1, 257, 1, 100, device=device, dtype=torch.float16)
    v = torch.randn(1, 257, 1, 64, device=device, dtype=torch.float16)
    g = torch.empty(1, 257, 4, device=device).uniform_(-0.1, -0.01)
    sink_bias = torch.randn(4, device=device)
    q, k, v, g, sink_bias = [torch.stack((x, x), dim=-1)[..., 0] for x in (q, k, v, g, sink_bias)]
    cu_seqlens = torch.tensor([0, 15, 79, 79, 257], device=device, dtype=torch.int32)
    _assert_parallel_close(q=q, k=k, v=v, g=g, sink_bias=sink_bias, cu_seqlens=cu_seqlens)
    assert gluon_route == {'fwd': 1, 'bwd': 1}


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
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, D, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    _assert_parallel_close(q=q, k=k, v=v, window_size=W)


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
    T = cu_seqlens[-1]
    q = torch.randn(1, T, HQ, D, device=device, dtype=torch.float16)
    k = torch.randn(1, T, H, D, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    _assert_parallel_close(q=q, k=k, v=v, window_size=W, cu_seqlens=cu_seqlens)


@requires_gluon
@pytest.mark.parametrize('window_size', [0, 1, 63, 64, 65, 1024])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
def test_parallel_swa_boundary(gluon_route, window_size, varlen):
    torch.manual_seed(42)
    q = torch.randn(1, 257, 4, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 257, 1, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    g = torch.empty(1, 257, 4, device=device).uniform_(-0.1, -0.01)
    sink_bias = torch.randn(4, device=device)
    cu_seqlens = torch.tensor([0, 15, 79, 79, 257], device=device, dtype=torch.int32) if varlen else None
    _assert_parallel_close(q=q, k=k, v=v, g=g, sink_bias=sink_bias, window_size=window_size, cu_seqlens=cu_seqlens)
    assert gluon_route == {'fwd': 1, 'bwd': 1}


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
def test_parallel_sink(B, T, H, HQ, K, V, scale, window_size, cu_seqlens, use_g, tol):
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=torch.float16)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float16)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16)
    g = torch.empty(B, T, HQ, device=device, dtype=torch.float16).uniform_(-0.1, -0.01) if use_g else None
    sink_bias = torch.randn(HQ, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32) if cu_seqlens is not None else None
    _assert_parallel_close(
        q=q,
        k=k,
        v=v,
        g=g,
        sink_bias=sink_bias,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
        tol=tol,
    )


def test_parallel_bwd_full_value_reduction(monkeypatch):
    """Regression test: the backward must not split the value dim (NV == 1).

    On low-shared-memory GPUs (e.g. consumer RTX cards) the backward used to cap BV <= 64,
    so any head dim > 64 split V across programs and produced silently wrong dq/dk. We force
    that low-smem branch here so the bug is caught on any GPU, not just consumer cards. The
    same setup forces a split forward and validates its LSE through backward.
    """
    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_ATTN_GLUON', '0')
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


@requires_gluon
@pytest.mark.parametrize('chunk_size', [32, 64])
def test_parallel_bwd_chunk_indices(gluon_route, chunk_size):
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
@pytest.mark.parametrize('use_tma', [False, True])
@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize('dim', [128, 512])
def test_parallel_copy_path(gluon_route, monkeypatch, use_tma, varlen, dim):
    from fla.ops.attn.backends.gluon import parallel

    calls = {'generic': 0, 'pipeline': 0}
    for name, kernel in (
        ('generic', parallel.parallel_attn_fwd_kernel_gluon),
        ('pipeline', parallel.parallel_attn_fwd_kernel_pipeline),
    ):
        original = kernel.run

        def wrapped(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(kernel, 'run', wrapped)
    monkeypatch.setattr(parallel, 'IS_TMA_SUPPORTED', use_tma)
    torch.manual_seed(42)
    q = torch.randn(1, 257, 4, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 257, 2, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    g = torch.empty(1, 257, 4, device=device).uniform_(-0.1, -0.01)
    sink_bias = torch.randn(4, device=device)
    cu_seqlens = torch.tensor([0, 15, 79, 79, 257], device=device, dtype=torch.int32) if varlen else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, 128) if varlen else None
    _assert_parallel_close(
        q=q,
        k=k,
        v=v,
        g=g,
        sink_bias=sink_bias,
        window_size=127,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert gluon_route == {'fwd': 1, 'bwd': 1}
    pipeline = use_tma and dim <= 128 and get_device_capability()[0] == 10
    assert calls == {'generic': int(not pipeline), 'pipeline': int(pipeline)}


@requires_gluon
@pytest.mark.parametrize('use_tma', [False, True])
def test_parallel_varlen_compilation(gluon_route, monkeypatch, use_tma):
    from fla.ops.attn.backends.gluon import parallel

    monkeypatch.setattr(parallel, 'IS_TMA_SUPPORTED', use_tma)
    torch.manual_seed(42)
    tensors = [torch.randn(1, 512, 3, 64, device=device, dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    refs = [x.detach().float().requires_grad_() for x in tensors]
    do = torch.randn_like(tensors[0])
    kernels = (
        parallel.parallel_attn_fwd_kernel_gluon,
        parallel.parallel_attn_fwd_kernel_pipeline,
        parallel.parallel_attn_bwd_kernel_gluon,
    )
    counts = None
    for offsets in ([0, 512], [0, 63, 512], [0, 63, 127, 512], [0, 1, 1, 63, 127, 256, 512]):
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
        expected = []
        for left, right in zip(offsets[:-1], offsets[1:]):
            if left < right:
                out, _ = naive_parallel_attn(q=refs[0][:, left:right], k=refs[1][:, left:right], v=refs[2][:, left:right])
                expected.append(out)
        ref = torch.cat(expected, dim=1)
        actual = parallel_attn(q=tensors[0], k=tensors[1], v=tensors[2], cu_seqlens=cu_seqlens)
        grads = torch.autograd.grad(actual, tensors, do)
        ref_grads = torch.autograd.grad(ref, refs, do.float())
        for name, value, expected in zip(('o', 'dq', 'dk', 'dv'), (actual, *grads), (ref, *ref_grads)):
            assert torch.isfinite(value).all(), name
            assert_close(name, expected, value, 0.005)
        current = [sum(len(state[0]) for state in kernel.device_caches.values()) for kernel in kernels]
        if counts is None:
            counts = current
        else:
            assert current == counts, 'Changing packed boundaries must reuse compiled attention kernels'
    assert gluon_route == {'fwd': 4, 'bwd': 4}


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


@requires_gluon
@pytest.mark.parametrize('varlen', [False, True])
@pytest.mark.parametrize('use_tma', [False, True])
def test_parallel_large_grid(gluon_route, monkeypatch, varlen, use_tma):
    from fla.ops.attn.backends.gluon import parallel

    monkeypatch.setattr(parallel, 'IS_TMA_SUPPORTED', use_tma)
    torch.manual_seed(42)
    n, hq = (65536, 1) if varlen else (8192, 8)
    q = torch.randn(n, 1, hq, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(n, 1, 1, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    refs = [x.float().requires_grad_() for x in (q, k, v)]
    tensors = [x.reshape(1, n, *x.shape[2:]) if varlen else x for x in (q, k, v)]
    tensors = [x.requires_grad_() for x in tensors]
    cu = torch.arange(n + 1, device=device, dtype=torch.int32) if varlen else None
    actual = parallel_attn(q=tensors[0], k=tensors[1], v=tensors[2], cu_seqlens=cu)
    ref, _ = naive_parallel_attn(q=refs[0], k=refs[1], v=refs[2])
    do = torch.randn_like(actual)
    actual.backward(do)
    ref.backward(do.reshape_as(ref).float())
    assert_close('o', ref.reshape_as(actual), actual, 0.005)
    for name, tensor, reference in zip(('dq', 'dk', 'dv'), tensors, refs):
        assert_close(name, reference.grad.reshape_as(tensor), tensor.grad, 0.005)
    assert gluon_route == {'fwd': 1, 'bwd': 1}


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


@pytest.mark.parametrize('op', [naive_attn_decoding, attn_decoding_one_step], ids=['naive', 'decode'])
def test_decoding_invalid_window(op):
    q = torch.empty(1, 1, 1, 64, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match='window_size must be nonnegative'):
        op(q=q, k=q, v=q, cu_seqlens=cu_seqlens, window_size=-1)


@pytest.mark.parametrize(
    ('H', 'HQ', 'K', 'V', 'W', 'use_g', 'do_gate_scale', 'use_sink', 'dtype', 'lengths', 'scale', 'sink_scale'),
    [
        pytest.param(
            *test,
            id="H{}-HQ{}-K{}-V{}-W{}-g{}-gate-scale{}-sink{}-{}-lengths{}-scale{}-sink-scale{}".format(*test),
        )
        for test in [
            (H, HQ, K, V, W, use_g, do_gate_scale, use_sink, dtype, [0, 15, 64, 127], 0.1, 1.0)
            for H, HQ, K, V, W, use_g, do_gate_scale, use_sink, dtype in [
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
            ]
        ] + [
            (2, HQ, 64, V, None, use_g, do_gate_scale, use_sink, torch.float16, lengths, scale, 0.7 if use_g else 1.0)
            for HQ, V, lengths, use_sink, use_g, do_gate_scale, scale in [
                (8, 64, [128, 128, 128], True, False, False, 0.1),
                (4, 320, [64, 64], False, False, False, None),
                (8, 64, [0, 128, 73], True, False, False, 0.1),
                (8, 64, [128, 128, 128], True, True, False, 0.1),
                (8, 64, [128, 128, 128], True, True, True, 0.1),
            ]
        ]
    ],
)
@pytest.mark.parametrize('strided', [False, True], ids=['contiguous', 'strided'])
def test_decoding(H, HQ, K, V, W, use_g, do_gate_scale, use_sink, dtype, lengths, scale, sink_scale, strided):
    torch.manual_seed(42)
    B, T = len(lengths), sum(lengths)
    q = torch.randn(1, B, HQ, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)
    g = torch.empty(1, T, HQ, dtype=dtype, device=device).uniform_(-0.1, -0.01) if use_g else None
    sink_bias = torch.randn(HQ, dtype=torch.float32, device=device) if use_sink else None
    if sink_bias is not None:
        sink_bias = sink_bias * sink_scale
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    if strided:
        q, k, v, g, sink_bias, cu_seqlens = [
            torch.stack((x, x), dim=-1)[..., 0] if x is not None else None
            for x in (q, k, v, g, sink_bias, cu_seqlens)
        ]
    kwargs = dict(scale=scale, cu_seqlens=cu_seqlens, do_gate_scale=do_gate_scale, window_size=W, sink_bias=sink_bias)

    ref = naive_attn_decoding(q=q.float(), k=k.float(), v=v.float(), g=g.float() if use_g else None, **kwargs).to(dtype)
    tri = attn_decoding_one_step(q=q, k=k, v=v, g=g, **kwargs)
    assert torch.isfinite(tri).all()
    assert_close("o", ref, tri, 0.01)

    if W is None:
        # attention layers pass the gate positionally.
        default = attn_decoding_one_step(q, k, v, g, scale, cu_seqlens, do_gate_scale, sink_bias=sink_bias)
        torch.testing.assert_close(default, tri, rtol=0, atol=0)


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('K', 'V', 'offset'),
    [(1, 3, 0), (65, 127, 0), (64, 64, 0), (100, 60, 0), (256, 512, 0), (512, 256, 0), (512, 512, 0),
     (64, 64, 1), (64, 64, 2), (64, 64, 4), (64, 64, 8), (100, 60, 2)],
)
@pytest.mark.parametrize('window', [None, 0, 65])
def test_decoding_split(gluon_decoding_route, dtype, K, V, offset, window):
    torch.manual_seed(42)
    q = torch.randn(1, 4, 4, K, device=device, dtype=dtype)
    k = torch.randn(2311 * K + offset, device=device, dtype=dtype)[offset:].view(1, 2311, 1, K)
    v = torch.randn(2311 * V + offset, device=device, dtype=dtype)[offset:].view(1, 2311, 1, V)
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
    assert len(gluon_decoding_route) == 2


@requires_gluon
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(('T', 'K'), [(256, 128), (8192, 512)])
def test_decoding_score_precision(gluon_decoding_route, dtype, T, K):
    torch.manual_seed(42)
    q = torch.randn(1, 1, 8, K, device=device, dtype=dtype)
    k = torch.randn(1, T, 2, K, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    actual = attn_decoding_one_step(q=q, k=k, v=v, cu_seqlens=cu)
    ref = naive_attn_decoding(q=q.float(), k=k.float(), v=v.float(), cu_seqlens=cu)
    assert_close('o', ref, actual, 0.01)
    assert len(gluon_decoding_route) == 1


@requires_gluon
def test_decoding_large_grid(gluon_decoding_route):
    torch.manual_seed(42)
    q = torch.randn(1, 8192, 8, 16, device=device, dtype=torch.float16)
    k = torch.randn(1, 8192, 1, 16, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    cu = torch.arange(8193, device=device, dtype=torch.int32)
    actual = attn_decoding_one_step(q=q, k=k, v=v, cu_seqlens=cu)
    assert_close('o', v.expand_as(q), actual, 0.01)
    assert len(gluon_decoding_route) == 1
