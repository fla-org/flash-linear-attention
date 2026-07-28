# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import fla.ops.generalized_delta_rule.dplr.backends.tilelang as dplr_tilelang_backend
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from fla.ops.generalized_delta_rule.dplr.backends.tilelang import DPLRTileLangBackend
from fla.utils import IS_NVIDIA_HOPPER, assert_close, device

_TILELANG_USABLE = DPLRTileLangBackend.is_available()
_DISPATCH_DISABLED = os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"

requires_tilelang_route = pytest.mark.skipif(
    _DISPATCH_DISABLED or not _TILELANG_USABLE,
    reason='TileLang backend not available or dispatch disabled',
)


def _verifier_inputs(K: int = 64, V: int | None = None, dtype: torch.dtype = torch.bfloat16, gk_dtype: torch.dtype | None = None):
    V = K if V is None else V

    def make(d, dt=dtype):
        return torch.empty(1, 16, 2, d, dtype=dt)

    return make(K), make(K), make(V), make(K), make(K), make(K, gk_dtype or dtype)


@pytest.mark.parametrize(('K', 'dtype', 'chunk_size'), [(64, torch.bfloat16, 32), (128, torch.bfloat16, 16), (64, torch.float16, 16)])
def test_chunk_verifier_accepts(K: int, dtype: torch.dtype, chunk_size: int):
    ok, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(
        *_verifier_inputs(K=K, dtype=dtype), chunk_size=chunk_size,
    )
    assert ok and reason is None


def test_chunk_verifier_accepts_chunk64_on_hopper(monkeypatch):
    monkeypatch.setattr(dplr_tilelang_backend, 'IS_NVIDIA_HOPPER', True)
    ok, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(*_verifier_inputs(), chunk_size=64)
    assert ok and reason is None


@pytest.mark.parametrize(
    ('case', 'reason'),
    [
        ('cp', 'context parallelism'),
        ('fp32', 'does not support dtype'),
        ('dtype_mismatch', 'dtypes to match'),
        ('kv_mismatch', 'K == V'),
        ('head_dim', 'head dim'),
        ('chunk64', 'Hopper'),
        ('chunk48', 'chunk_size'),
    ],
)
def test_chunk_verifier_rejects(monkeypatch, case: str, reason: str):
    kwargs = {}
    if case == 'cp':
        args = _verifier_inputs()
        kwargs['cp_context'] = SimpleNamespace()
    elif case == 'fp32':
        args = _verifier_inputs(dtype=torch.float32)
    elif case == 'dtype_mismatch':
        args = _verifier_inputs(gk_dtype=torch.float32)
    elif case == 'kv_mismatch':
        args = _verifier_inputs(K=64, V=128)
    elif case == 'head_dim':
        args = _verifier_inputs(K=100, V=100)
    elif case == 'chunk64':
        monkeypatch.setattr(dplr_tilelang_backend, 'IS_NVIDIA_HOPPER', False)
        args = _verifier_inputs()
        kwargs['chunk_size'] = 64
    else:
        args = _verifier_inputs()
        kwargs['chunk_size'] = 48
    if 'chunk_size' not in kwargs:
        kwargs['chunk_size'] = 32
    ok, why = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(*args, **kwargs)
    assert not ok and reason in why


def _spy_on_tilelang_route(monkeypatch):
    calls = []
    orig = DPLRTileLangBackend.chunk_dplr_delta_rule

    def spy(self, *args, **kwargs):
        calls.append(None)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(DPLRTileLangBackend, 'chunk_dplr_delta_rule', spy)
    return calls


def _assert_route_parity(monkeypatch, run, names):
    monkeypatch.setenv('FLA_TILELANG', '0')
    ref = run()
    calls = _spy_on_tilelang_route(monkeypatch)
    monkeypatch.setenv('FLA_TILELANG', '1')
    tri = run()
    assert calls, 'TileLang backend route was not taken'
    # cross-backend fp32 accumulation order differs slightly; worst observed 0.0088
    for name, r, t in zip(names, ref, tri):
        assert_close(name, r, t, 0.01)


@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'safe_gate', 'dtype', 'chunk_size', 'disable_recompute'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-safe_gate{}-{}-chunk_size{}-disable_recompute{}".format(*test),
            marks=pytest.mark.skipif(
                test[6] == 64 and not IS_NVIDIA_HOPPER,
                reason='chunk_size 64 route requires sm90',
            ),
        )
        for test in [
            (2, 512, 3, 64, True, torch.bfloat16, 32, False),
            (2, 512, 3, 64, True, torch.bfloat16, 32, True),
            (2, 512, 3, 128, False, torch.bfloat16, 32, False),
            (2, 512, 3, 64, True, torch.bfloat16, 16, False),
            (2, 512, 3, 128, False, torch.bfloat16, 16, False),
            (2, 512, 3, 64, True, torch.float16, 32, False),
            (1, 63, 1, 64, True, torch.float16, 16, False),
            (2, 512, 3, 64, True, torch.bfloat16, 64, False),
        ]
    ],
)
def test_chunk_tilelang_route_parity(
    monkeypatch,
    B: int,
    T: int,
    H: int,
    D: int,
    safe_gate: bool,
    dtype: torch.dtype,
    chunk_size: int,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)).to(dtype)
    if safe_gate:
        gk = gk.clamp(-5, 0)
    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = (x.to(device) for x in (q, k, v, a, b, gk, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run():
        q_, k_, v_, a_, b_, gk_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, a, b, gk, h0))
        o, st = chunk_dplr_delta_rule(
            q=q_, k=k_, v=v_, a=a_, b=b_, gk=gk_,
            scale=1.0,
            initial_state=h0_,
            output_final_state=True,
            safe_gate=safe_gate,
            chunk_size=chunk_size,
            disable_recompute=disable_recompute,
        )
        ((o * do).sum() + (st * dht).sum()).backward()
        return o, st, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad, h0_.grad

    _assert_route_parity(monkeypatch, run, ('o', 'ht', 'dq', 'dk', 'dv', 'da', 'db', 'dgk', 'dh0'))


@requires_tilelang_route
@pytest.mark.parametrize(
    ('H', 'D', 'safe_gate', 'cu_seqlens', 'dtype', 'chunk_size'),
    [
        pytest.param(*test, id="H{}-D{}-safe_gate{}-cu_seqlens{}-{}-chunk_size{}".format(*test))
        for test in [
            (3, 64, True, [0, 256, 500, 1000], torch.bfloat16, 32),
            (3, 128, False, [0, 256, 500, 1000], torch.bfloat16, 32),
            (3, 64, True, [0, 256, 500, 1000], torch.bfloat16, 16),
            (3, 64, True, [0, 256, 500, 1000], torch.float16, 32),
        ]
    ],
)
def test_chunk_varlen_tilelang_route_parity(
    monkeypatch,
    H: int,
    D: int,
    safe_gate: bool,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    chunk_size: int,
):
    torch.manual_seed(42)
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn(1, T, H, D, dtype=dtype)
    k = torch.randn(1, T, H, D, dtype=dtype)
    v = torch.randn(1, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(1, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float)).to(dtype)
    if safe_gate:
        gk = gk.clamp(-5, 0)
    h0 = torch.randn(N, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = (x.to(device) for x in (q, k, v, a, b, gk, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run():
        q_, k_, v_, a_, b_, gk_, h0_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, a, b, gk, h0))
        o, st = chunk_dplr_delta_rule(
            q=q_, k=k_, v=v_, a=a_, b=b_, gk=gk_,
            initial_state=h0_,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            safe_gate=safe_gate,
            chunk_size=chunk_size,
        )
        ((o * do).sum() + (st * dht).sum()).backward()
        return o, st, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad, h0_.grad

    _assert_route_parity(monkeypatch, run, ('o', 'ht', 'dq', 'dk', 'dv', 'da', 'db', 'dgk', 'dh0'))
