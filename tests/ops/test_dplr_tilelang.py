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
from fla.ops.generalized_delta_rule.dplr.backends.tilelang.schedules import chunk64_schedule_or_none
from fla.ops.generalized_delta_rule.dplr.naive import dplr_recurrence
from fla.utils import assert_close, device, get_device_capability, get_device_smem_optin

_TILELANG_USABLE = DPLRTileLangBackend.is_available()
_DISPATCH_DISABLED = os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
_CUDA_AVAILABLE = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason='verifier device queries need CUDA',
)
requires_tilelang_route = pytest.mark.skipif(
    _DISPATCH_DISABLED or not _TILELANG_USABLE,
    reason='TileLang backend not available or dispatch disabled',
)


def _cs64_launchable(K: int) -> bool:
    if not _CUDA_AVAILABLE:
        return False
    cc_major, cc_minor = get_device_capability(0)
    return chunk64_schedule_or_none(
        K=K, V=K, in_dtype='bfloat16',
        smem_cap=get_device_smem_optin(0), cc=cc_major * 10 + cc_minor,
    ) is not None


def _verifier_inputs(
    K: int = 64,
    V: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    gk_dtype: torch.dtype | None = None,
    B: int = 4,
    H: int = 32,
):
    V = K if V is None else V

    def make(d, dt=dtype):
        return torch.empty(B, 16, H, d, dtype=dt, device=device)

    return make(K), make(K), make(V), make(K), make(K), make(K, gk_dtype or dtype)


@requires_cuda
@pytest.mark.parametrize(('K', 'dtype', 'chunk_size'), [(64, torch.bfloat16, 32), (128, torch.bfloat16, 32), (64, torch.float16, 16)])
def test_chunk_verifier_accepts(K: int, dtype: torch.dtype, chunk_size: int):
    ok, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(
        *_verifier_inputs(K=K, dtype=dtype), safe_gate=True, chunk_size=chunk_size,
    )
    assert ok and reason is None


@requires_cuda
def test_chunk_verifier_accepts_cp():
    # CP requires a real context: cu_seqlens set, no initial_state, no final state
    cp_context = SimpleNamespace(
        cu_seqlens=torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32, device=device),
    )
    ok, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(
        *_verifier_inputs(), safe_gate=True, chunk_size=32, cp_context=cp_context,
    )
    assert ok and reason is None


@requires_cuda
def test_chunk_verifier_accepts_fp32_gk():
    # gk may stay fp32 while activations are fp16/bf16 (FLA's own test convention)
    ok, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(
        *_verifier_inputs(gk_dtype=torch.float32), safe_gate=True, chunk_size=32,
    )
    assert ok and reason is None


@requires_cuda
@pytest.mark.parametrize('K', [64, 128])
def test_chunk_verifier_accepts_chunk64_on_large_smem_device(monkeypatch, K: int):
    monkeypatch.setattr(dplr_tilelang_backend, 'get_device_smem_optin', lambda idx: 232448)
    monkeypatch.setattr(dplr_tilelang_backend, 'get_device_capability', lambda idx: (9, 0))
    ok, reason = DPLRTileLangBackend().chunk_dplr_delta_rule_verifier(
        *_verifier_inputs(K=K), safe_gate=True, chunk_size=64,
    )
    assert ok and reason is None


@requires_cuda
@pytest.mark.parametrize(
    ('case', 'reason'),
    [
        ('fp32', 'does not support dtype'),
        ('dtype_mismatch', 'dtypes to match'),
        ('kv_mismatch', 'K == V'),
        ('head_dim', 'head dim'),
        ('safe_gate', 'requires safe_gate=True'),
        ('chunk64_a100_k128', 'no launchable backward schedule'),
        ('chunk64_small_smem_k128', 'no launchable backward schedule'),
        ('chunk64_small_smem_k64', 'no launchable backward schedule'),
        ('chunk16_k128', 'slower than Triton'),
        ('chunk48', 'chunk_size'),
        ('small_grid', 'small grids'),
        ('low_v2_small_grid', 'low-smem stream backward'),
        ('cp_initial_state', 'initial_state with CP'),
        ('cp_final_state', 'output_final_state with CP'),
        ('cp_no_cu_seqlens', 'cu_seqlens for CP'),
    ],
)
def test_chunk_verifier_rejects(monkeypatch, case: str, reason: str):
    kwargs = {'safe_gate': True}
    if case == 'fp32':
        args = _verifier_inputs(dtype=torch.float32)
    elif case == 'dtype_mismatch':
        args = list(_verifier_inputs())
        args[1] = torch.empty_like(args[1], dtype=torch.float32)
        args = tuple(args)
    elif case == 'kv_mismatch':
        args = _verifier_inputs(K=64, V=128)
    elif case == 'head_dim':
        args = _verifier_inputs(K=100, V=100)
    elif case == 'safe_gate':
        args = _verifier_inputs()
        kwargs['safe_gate'] = False
    elif case == 'chunk64_a100_k128':
        # high=297472B, mid=215552B and low=167936B all exceed A100's 166912B optin
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_smem_optin', lambda idx: 166912)
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_capability', lambda idx: (8, 0))
        args = _verifier_inputs(K=128)
        kwargs['chunk_size'] = 64
    elif case == 'chunk64_small_smem_k128':
        # the fused A-backward stage needs 131200B off cc90, and every stream
        # schedule (mid=215552B, low=167936B) also exceeds the 99KB cap
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_smem_optin', lambda idx: 101376)
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_capability', lambda idx: (12, 0))
        args = _verifier_inputs(K=128)
        kwargs['chunk_size'] = 64
    elif case == 'chunk64_small_smem_k64':
        # the K=64 stream backward fits cc120's 99KB cap via low_v2 (81920B),
        # but the fused A-backward stage needs 131200B there
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_smem_optin', lambda idx: 101376)
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_capability', lambda idx: (12, 0))
        args = _verifier_inputs(K=64)
        kwargs['chunk_size'] = 64
    elif case == 'chunk16_k128':
        args = _verifier_inputs(K=128)
        kwargs['chunk_size'] = 16
    elif case == 'small_grid':
        args = _verifier_inputs(B=1, H=1)
    elif case == 'low_v2_small_grid':
        # cc120-class 99KB cap forces the low_v2 stream schedule at K=128;
        # N*H=64 below half the (pinned) SM count must fall back
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_smem_optin', lambda idx: 101376)
        monkeypatch.setattr(dplr_tilelang_backend, 'get_device_capability', lambda idx: (12, 0))
        monkeypatch.setattr(dplr_tilelang_backend, 'get_multiprocessor_count', lambda idx: 188)
        args = _verifier_inputs(B=2, H=32, K=128)
    elif case == 'cp_initial_state':
        args = _verifier_inputs()
        kwargs['initial_state'] = torch.empty(4, 32, 64, 64, device=device)
        kwargs['cp_context'] = SimpleNamespace(
            cu_seqlens=torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32, device=device),
        )
    elif case == 'cp_final_state':
        args = _verifier_inputs()
        kwargs['output_final_state'] = True
        kwargs['cp_context'] = SimpleNamespace(
            cu_seqlens=torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32, device=device),
        )
    elif case == 'cp_no_cu_seqlens':
        args = _verifier_inputs()
        kwargs['cp_context'] = SimpleNamespace()
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
    ('B', 'T', 'H', 'D', 'dtype', 'chunk_size', 'disable_recompute', 'use_state'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-{}-chunk_size{}-disable_recompute{}-use_state{}".format(*test),
            marks=pytest.mark.skipif(
                test[5] == 64 and not _cs64_launchable(test[3]),
                reason='chunk_size 64 has no launchable backward schedule on this device',
            ),
        )
        for test in [
            (8, 512, 32, 64, torch.bfloat16, 32, False, True),
            (8, 512, 32, 64, torch.bfloat16, 32, True, True),
            (8, 512, 32, 128, torch.bfloat16, 32, False, True),
            (8, 512, 32, 64, torch.bfloat16, 16, False, True),
            (8, 512, 32, 64, torch.float16, 32, False, True),
            (8, 63, 32, 64, torch.float16, 16, False, True),
            (8, 512, 32, 64, torch.bfloat16, 64, False, True),
            # RWKV7 training branch: no initial state, no final state
            (8, 512, 32, 64, torch.bfloat16, 32, False, False),
            (8, 512, 32, 64, torch.bfloat16, 64, False, False),
        ]
    ],
)
def test_chunk_tilelang_route_parity(
    monkeypatch,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    chunk_size: int,
    disable_recompute: bool,
    use_state: bool,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)).to(dtype).clamp(-5, 0)
    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = (x.to(device) for x in (q, k, v, a, b, gk, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run():
        q_, k_, v_, a_, b_, gk_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, a, b, gk))
        h0_ = h0.detach().clone().requires_grad_(True) if use_state else None
        o, st = chunk_dplr_delta_rule(
            q=q_, k=k_, v=v_, a=a_, b=b_, gk=gk_,
            scale=1.0,
            initial_state=h0_,
            output_final_state=use_state,
            safe_gate=True,
            chunk_size=chunk_size,
            disable_recompute=disable_recompute,
        )
        loss = (o * do).sum()
        if use_state:
            loss = loss + (st * dht).sum()
        loss.backward()
        outs = [o, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad]
        if use_state:
            outs += [st, h0_.grad]
        return outs

    names = ['o', 'dq', 'dk', 'dv', 'da', 'db', 'dgk'] + (['ht', 'dh0'] if use_state else [])
    _assert_route_parity(monkeypatch, run, names)


@requires_tilelang_route
def test_chunk_tilelang_route_parity_fp32_gk(monkeypatch):
    torch.manual_seed(42)
    B, T, H, D = 8, 512, 32, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    # FLA's own tests keep gk in fp32 with bf16 activations
    gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)).clamp(-5, 0)
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
            safe_gate=True,
            chunk_size=32,
        )
        ((o * do).sum() + (st * dht).sum()).backward()
        return o, st, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad, h0_.grad

    _assert_route_parity(monkeypatch, run, ('o', 'ht', 'dq', 'dk', 'dv', 'da', 'db', 'dgk', 'dh0'))


@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype', 'chunk_size', 'gate_style'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-{}-chunk_size{}-{}".format(*test),
            marks=pytest.mark.skipif(
                test[5] == 64 and not _cs64_launchable(test[3]),
                reason='chunk_size 64 has no launchable backward schedule on this device',
            ),
        )
        for test in [
            # gates pushed toward the documented -5 bound: gk is natural-log
            # decay, so the mid-chunk-centered exponents reach up to
            # (BT/2)*5*log2(e) = 115 (BT=32) / 58 (BT=16) log2, the worst
            # spread the safe_gate contract licenses at these chunk sizes
            (8, 512, 32, 64, torch.bfloat16, 32, 'saturated_lb'),
            (8, 512, 32, 64, torch.bfloat16, 16, 'saturated_lb'),
            (8, 512, 32, 128, torch.bfloat16, 32, 'saturated_lb'),
            # RWKV7's w is architecturally clamped to (-0.61, 0), which keeps
            # the BT=64 half-range at 28 log2; saturated to push both ends
            (8, 512, 32, 64, torch.bfloat16, 64, 'rwkv7'),
            (8, 512, 32, 128, torch.bfloat16, 64, 'rwkv7'),
        ]
    ],
)
def test_chunk_tilelang_route_parity_gate_stress(
    monkeypatch,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    chunk_size: int,
    gate_style: str,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    if gate_style == 'saturated_lb':
        # KDA's convention (tests/ops/test_kda.py): amplify the logits so the
        # clamp pins a fraction of gates at the documented -5 bound. Sustained
        # pinning of most gates is excluded on purpose: at cs32 the Triton
        # reference itself overflows fp32 there (TileLang stays finite), so
        # parity is undefined
        gk = (F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / 0.6).clamp(-5, 0).to(dtype)
    else:
        # sigmoid saturates to 0/1, pinning gk at the ends of (-0.61, 0)
        gk = (-0.61 * torch.sigmoid(5 * torch.randn(B, T, H, D, dtype=torch.float))).to(dtype)
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
            safe_gate=True,
            chunk_size=chunk_size,
        )
        ((o * do).sum() + (st * dht).sum()).backward()
        return o, st, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad, h0_.grad

    _assert_route_parity(monkeypatch, run, ('o', 'ht', 'dq', 'dk', 'dv', 'da', 'db', 'dgk', 'dh0'))


@requires_tilelang_route
@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype', 'chunk_size', 'use_state'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}-chunk_size{}-use_state{}".format(*test))
        for test in [
            (32, 64, [0, 256, 500, 760, 1000], torch.bfloat16, 32, True),
            (32, 128, [0, 256, 500, 760, 1000], torch.bfloat16, 32, True),
            (32, 64, [0, 256, 500, 760, 1000], torch.bfloat16, 16, True),
            (32, 64, [0, 256, 500, 760, 1000], torch.float16, 32, True),
            (32, 64, [0, 256, 500, 760, 1000], torch.bfloat16, 32, False),
        ]
    ],
)
def test_chunk_varlen_tilelang_route_parity(
    monkeypatch,
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    chunk_size: int,
    use_state: bool,
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
    gk = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float)).to(dtype).clamp(-5, 0)
    h0 = torch.randn(N, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = (x.to(device) for x in (q, k, v, a, b, gk, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run():
        q_, k_, v_, a_, b_, gk_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, a, b, gk))
        h0_ = h0.detach().clone().requires_grad_(True) if use_state else None
        o, st = chunk_dplr_delta_rule(
            q=q_, k=k_, v=v_, a=a_, b=b_, gk=gk_,
            initial_state=h0_,
            output_final_state=use_state,
            cu_seqlens=cu_seqlens,
            safe_gate=True,
            chunk_size=chunk_size,
        )
        loss = (o * do).sum()
        if use_state:
            loss = loss + (st * dht).sum()
        loss.backward()
        outs = [o, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad]
        if use_state:
            outs += [st, h0_.grad]
        return outs

    names = ['o', 'dq', 'dk', 'dv', 'da', 'db', 'dgk'] + (['ht', 'dh0'] if use_state else [])
    _assert_route_parity(monkeypatch, run, names)


def _naive_recurrence(q, k, v, a, b, gk, h0):
    # per-token fp32 PyTorch baseline (no chunk math); dplr_recurrence works
    # on [B, H, T, D] and applies the K**-0.5 scale internally
    o, st = dplr_recurrence(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        a.transpose(1, 2), b.transpose(1, 2), gk.transpose(1, 2),
        initial_state=h0, output_final_state=True,
    )
    return o.transpose(1, 2), st


def _assert_naive_parity(monkeypatch, inputs, do, dht, names, call):
    q, k, v, a, b, gk, h0 = inputs

    def run(fn):
        q_, k_, v_, a_, b_, gk_ = (x.detach().clone().requires_grad_(True) for x in (q, k, v, a, b, gk))
        h0_ = h0.detach().clone().requires_grad_(True)
        o, st = call(fn, q_, k_, v_, a_, b_, gk_, h0_)
        ((o * do).sum() + (st * dht).sum()).backward()
        return o, st, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad, h0_.grad

    calls = _spy_on_tilelang_route(monkeypatch)
    monkeypatch.setenv('FLA_TILELANG', '1')
    tri = run('op')
    assert calls, 'TileLang backend route was not taken'
    ref = run('naive')
    # same ratios test_dplr_delta uses for the Triton path vs the fp32 ref
    for name, r, t in zip(names, ref, tri):
        assert_close(name, r, t, 0.007 if name == 'o' else 0.008)


_NAIVE_PARITY_NAMES = ('o', 'ht', 'dq', 'dk', 'dv', 'da', 'db', 'dgk', 'dh0')


@requires_tilelang_route
@pytest.mark.parametrize(
    ('T', 'D', 'dtype', 'chunk_size', 'gate_style'),
    [
        pytest.param(
            *test,
            id="T{}-D{}-{}-chunk_size{}-{}".format(*test),
            marks=pytest.mark.skipif(
                test[3] == 64 and not _cs64_launchable(test[1]),
                reason='chunk_size 64 has no launchable backward schedule on this device',
            ),
        )
        for test in [
            (256, 64, torch.bfloat16, 32, 'standard'),
            (256, 64, torch.bfloat16, 16, 'standard'),
            (256, 128, torch.bfloat16, 32, 'standard'),
            (256, 64, torch.float16, 32, 'standard'),
            (256, 64, torch.bfloat16, 64, 'standard'),
            (256, 64, torch.bfloat16, 32, 'saturated_lb'),
            (256, 128, torch.bfloat16, 64, 'rwkv7'),
        ]
    ],
)
def test_chunk_tilelang_naive_ref_parity(
    monkeypatch,
    T: int,
    D: int,
    dtype: torch.dtype,
    chunk_size: int,
    gate_style: str,
):
    torch.manual_seed(42)
    B, H = 8, 32
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    if gate_style == 'saturated_lb':
        gk = (F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)) / 0.6).clamp(-5, 0).to(dtype)
    elif gate_style == 'rwkv7':
        gk = (-0.61 * torch.sigmoid(5 * torch.randn(B, T, H, D, dtype=torch.float))).to(dtype)
    else:
        gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float)).to(dtype).clamp(-5, 0)
    h0 = torch.randn(B, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = (x.to(device) for x in (q, k, v, a, b, gk, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def call(fn, q_, k_, v_, a_, b_, gk_, h0_):
        if fn == 'naive':
            return _naive_recurrence(q_, k_, v_, a_, b_, gk_, h0_)
        return chunk_dplr_delta_rule(
            q=q_, k=k_, v=v_, a=a_, b=b_, gk=gk_,
            initial_state=h0_, output_final_state=True,
            safe_gate=True, chunk_size=chunk_size,
        )

    _assert_naive_parity(monkeypatch, (q, k, v, a, b, gk, h0), do, dht, _NAIVE_PARITY_NAMES, call)


@requires_tilelang_route
@pytest.mark.parametrize(
    ('D', 'cu_seqlens', 'dtype', 'chunk_size'),
    [
        pytest.param(*test, id="D{}-cu_seqlens{}-{}-chunk_size{}".format(*test))
        for test in [
            (64, [0, 130, 260, 390, 512], torch.bfloat16, 32),
            (128, [0, 130, 260, 390, 512], torch.bfloat16, 32),
        ]
    ],
)
def test_chunk_varlen_tilelang_naive_ref_parity(
    monkeypatch,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    chunk_size: int,
):
    torch.manual_seed(42)
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    H = 32
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn(1, T, H, D, dtype=dtype)
    k = torch.randn(1, T, H, D, dtype=dtype)
    v = torch.randn(1, T, H, D, dtype=dtype)
    a = F.normalize(torch.rand(1, T, H, D, dtype=dtype), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float)).to(dtype).clamp(-5, 0)
    h0 = torch.randn(N, H, D, D, dtype=torch.float)
    q, k, v, a, b, gk, h0 = (x.to(device) for x in (q, k, v, a, b, gk, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def call(fn, q_, k_, v_, a_, b_, gk_, h0_):
        if fn == 'naive':
            # the baseline has no varlen form; run each sequence and repack
            outs, sts = [], []
            for i in range(N):
                s, e = cu_seqlens[i], cu_seqlens[i + 1]
                o_i, st_i = _naive_recurrence(
                    q_[:, s:e], k_[:, s:e], v_[:, s:e],
                    a_[:, s:e], b_[:, s:e], gk_[:, s:e], h0_[i: i + 1],
                )
                outs.append(o_i)
                sts.append(st_i)
            return torch.cat(outs, 1), torch.cat(sts, 0)
        return chunk_dplr_delta_rule(
            q=q_, k=k_, v=v_, a=a_, b=b_, gk=gk_,
            initial_state=h0_, output_final_state=True,
            cu_seqlens=cu, safe_gate=True, chunk_size=chunk_size,
        )

    _assert_naive_parity(monkeypatch, (q, k, v, a, b, gk, h0), do, dht, _NAIVE_PARITY_NAMES, call)


@requires_tilelang_route
def test_chunk_tilelang_fwd_opcheck():
    torch.manual_seed(42)
    B, T, H, D = 2, 64, 8, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype, device=device), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device)).clamp(-5, 0)
    h0 = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    cu = torch.empty((0,), dtype=torch.int32, device=device)

    torch.library.opcheck(
        torch.ops.fla.chunk_dplr_delta_rule_fwd,
        (q, k, v, a, b, gk, h0, cu, 1.0, True, True, False, 32),
    )


@requires_tilelang_route
def test_chunk_tilelang_torch_compile_fullgraph_smoke():
    from fla.ops.generalized_delta_rule.dplr.backends.tilelang.chunk import chunk_dplr_delta_rule_tilelang

    torch.manual_seed(42)
    B, T, H, D = 2, 256, 8, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype, device=device), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device)).clamp(-5, 0).to(dtype)
    h0 = torch.randn(B, H, D, D, dtype=torch.float, device=device)

    def fn(q, k, v, a, b, gk, h0):
        return chunk_dplr_delta_rule_tilelang(
            q=q, k=k, v=v, a=a, b=b, gk=gk,
            scale=1.0,
            initial_state=h0,
            output_final_state=True,
            safe_gate=True,
            chunk_size=32,
        )

    ref_o, ref_st = fn(q, k, v, a, b, gk, h0)
    compiled = torch.compile(fn, fullgraph=True)
    tri_o, tri_st = compiled(q, k, v, a, b, gk, h0)
    assert_close('o', ref_o, tri_o, 0.005)
    assert_close('ht', ref_st, tri_st, 0.005)


@requires_tilelang_route
def test_chunk_tilelang_checkpoint_elision(monkeypatch):
    from fla.ops.generalized_delta_rule.dplr.backends.tilelang import chunk as tl_chunk
    from fla.ops.generalized_delta_rule.dplr.backends.tilelang.chunk import (
        chunk_dplr_delta_rule_tilelang,
        dplr_checkpoint_context_fn,
    )

    torch.manual_seed(42)
    B, T, H, D = 2, 256, 8, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    a = F.normalize(torch.rand(B, T, H, D, dtype=dtype, device=device), p=2, dim=-1)
    b = -a
    gk = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float, device=device)).clamp(-5, 0).to(dtype)
    h0 = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    elided_calls = []
    orig_elided = tl_chunk._chunk_dplr_delta_rule_fwd_ctx_elided_op

    def elided_spy(*args, **kwargs):
        elided_calls.append(None)
        return orig_elided(*args, **kwargs)

    monkeypatch.setattr(tl_chunk, '_chunk_dplr_delta_rule_fwd_ctx_elided_op', elided_spy)

    def fwd(q, k, v, a, b, gk, h0):
        return chunk_dplr_delta_rule_tilelang(
            q=q, k=k, v=v, a=a, b=b, gk=gk,
            scale=1.0,
            initial_state=h0,
            output_final_state=True,
            safe_gate=True,
            chunk_size=32,
            disable_recompute=True,
        )

    def run(use_checkpoint):
        leaves = (x.detach().clone().requires_grad_(True) for x in (q, k, v, a, b, gk, h0))
        q_, k_, v_, a_, b_, gk_, h0_ = leaves
        if use_checkpoint:
            o, st = torch.utils.checkpoint.checkpoint(
                fwd, q_, k_, v_, a_, b_, gk_, h0_,
                use_reentrant=False,
                context_fn=dplr_checkpoint_context_fn,
            )
        else:
            o, st = fwd(q_, k_, v_, a_, b_, gk_, h0_)
        ((o * do).sum() + (st * dht).sum()).backward()
        return o, st, q_.grad, k_.grad, v_.grad, a_.grad, b_.grad, gk_.grad, h0_.grad

    ref = run(use_checkpoint=False)
    tri = run(use_checkpoint=True)
    assert elided_calls, 'checkpoint forward did not take the ctx-elided route'
    for name, r, t in zip(('o', 'ht', 'dq', 'dk', 'dv', 'da', 'db', 'dgk', 'dh0'), ref, tri):
        assert_close(name, r, t, 0.01)
