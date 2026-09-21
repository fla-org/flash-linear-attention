# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Cache and dispatch tests for intracard CP."""

import os
from unittest.mock import patch

import pytest
import torch

import fla.ops.common.intracard_cp as intracard_cp_mod
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.intracard_cp import _intracard_cache
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
from fla.utils import IS_NVIDIA, assert_close, device


@pytest.fixture(autouse=True)
def clear_intracard_cache():
    _intracard_cache.clear()
    yield
    _intracard_cache.clear()


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize('T', [2048, 16384, 32768])
@pytest.mark.parametrize('use_graph', [False, True])
def test_chunk_kda_intracard_cache_hit_same_cu_seqlens_object(monkeypatch, T, use_graph):
    """KDA uses intracard for eager inference and preserves the device-only static metadata path."""
    # Enable intracard CP backend explicitly as it's disabled by default
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    torch.manual_seed(42)
    dtype = torch.bfloat16

    # Cover early fallback, metadata without splitting, and the split path.
    B, H, D = 1, 1, 32

    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.full((B, T, H, D), -0.05, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()

    call_count = 0
    original_precompute = intracard_cp_mod._precompute_intracard_indices

    def counted_precompute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_precompute(*args, **kwargs)

    monkeypatch.setattr(intracard_cp_mod, "_precompute_intracard_indices", counted_precompute)

    with torch.inference_mode(), patch.object(
        intracard_cp_mod, 'intracard_fwd_h', wraps=intracard_cp_mod.intracard_fwd_h,
    ) as intracard:
        o1, ht1 = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=True,
            use_graph=use_graph,
        )
        o2, ht2 = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=True,
            use_graph=use_graph,
        )

    assert intracard.call_count == (0 if use_graph else 2)
    assert call_count == int(T >= 24576 and not use_graph)
    assert len(_intracard_cache) == int(T >= 24576 and not use_graph)
    if _intracard_cache:
        key = next(iter(_intracard_cache))
        entry = _intracard_cache[key]
        assert key[0] == id(cu_seqlens)
        assert entry.cu_seqlens_ref() is cu_seqlens
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ht1, ht2, atol=0, rtol=0)

    monkeypatch.setenv('FLA_INTRACARD_CP', '0')
    with torch.inference_mode():
        ref, ref_ht = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
            use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
            output_final_state=True, use_graph=use_graph,
        )
    assert_close('o', ref, o1, 0.005)
    assert_close('ht', ref_ht, ht1, 0.005)


def test_intracard_backend_disabled_by_default():
    """Verify that IntraCardCPBackend is disabled by default."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    # When env var is not set, backend should be disabled (default_enable=False)
    assert IntraCardCPBackend.default_enable is False


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not IS_NVIDIA, reason="requires NVIDIA")
@pytest.mark.parametrize('cpu_mirror', [False, True])
@torch.inference_mode()
def test_chunk_kda_intracard_graph_replay(monkeypatch, cpu_mirror):
    """Enabling intracard must preserve graph replay with changing packed sequence boundaries."""
    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    torch.manual_seed(42)
    q = torch.randn(1, 32768, 1, 32, device=device, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(q)
    v = torch.randn_like(q)
    g = torch.full_like(q, -0.05, dtype=torch.float32)
    beta = torch.rand(1, 32768, 1, device=device, dtype=torch.bfloat16)
    h0 = torch.randn(3, 1, 32, 32, device=device)
    cu_cpu = torch.tensor([0, 24576, 32768, 32768], dtype=torch.int32)
    cu = cu_cpu.to(device)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True)

    def step():
        return chunk_kda(**kwargs, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if cpu_mirror else None, use_graph=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, state = step()

    monkeypatch.setenv('FLA_INTRACARD_CP', '0')
    for offsets in ([0, 24576, 32768, 32768], [0, 8192, 32768, 32768], [0, 16384, 32768, 32768]):
        cu_cpu.copy_(torch.tensor(offsets, dtype=torch.int32))
        cu.copy_(cu_cpu)
        graph.replay()
        ref, ref_state = chunk_kda(**kwargs, cu_seqlens=cu.clone())
        assert_close('o', ref, output, 0.005)
        assert_close('ht', ref_state, state, 0.005)


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not IS_NVIDIA, reason="requires NVIDIA")
@pytest.mark.parametrize('num_seqs', [0, 2])
@pytest.mark.parametrize('has_initial_state', [False, True])
@torch.inference_mode()
def test_chunk_kda_intracard_empty(monkeypatch, num_seqs, has_initial_state):
    """Empty packed input keeps its initial state when intracard is enabled."""
    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    torch.manual_seed(42)
    q = torch.empty(1, 0, 1, 32, device=device, dtype=torch.bfloat16)
    beta = torch.empty(1, 0, 1, device=device, dtype=torch.bfloat16)
    cu = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    h0 = torch.randn(num_seqs, 1, 32, 32, device=device) if has_initial_state else None
    output, state = chunk_kda(q, q, q, q, beta, initial_state=h0, output_final_state=True, cu_seqlens=cu)
    torch.testing.assert_close(output, q, atol=0, rtol=0)
    torch.testing.assert_close(state, h0 if h0 is not None else torch.zeros_like(state), atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not IS_NVIDIA, reason="requires NVIDIA")
@pytest.mark.parametrize('inference_tensors', [False, True])
@pytest.mark.parametrize('cpu_mirror', [False, True])
@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize('boundary', [8192, 8191, 0], ids=['aligned', 'unaligned', 'empty'])
def test_intracard_cache_inplace_offsets(monkeypatch, inference_tensors, cpu_mirror, state_v_first, boundary):
    """Reusing an offsets buffer must match a fresh buffer after its sequence boundaries change."""
    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    torch.manual_seed(42)
    with torch.inference_mode(inference_tensors):
        cu_cpu = torch.tensor([0, 24576, 32768], dtype=torch.int32)
        cu = cu_cpu.to(device)
        k = torch.nn.functional.normalize(torch.randn(1, 32768, 1, 32, device=device), dim=-1).to(torch.bfloat16)
        w = k * 0.1
        u = torch.randn(1, 32768, 1, 16, device=device, dtype=torch.bfloat16)
        g = -torch.arange(1, 65, device=device, dtype=torch.float32).repeat(512).reshape(1, 32768, 1) / 64
        h0 = torch.randn(2, 1, *([16, 32] if state_v_first else [32, 16]), device=device, dtype=torch.float32)

    kwargs = dict(k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True, state_v_first=state_v_first)
    with torch.inference_mode(), patch.object(
        intracard_cp_mod, '_precompute_intracard_indices', wraps=intracard_cp_mod._precompute_intracard_indices,
    ) as precompute:
        original = chunk_gated_delta_rule_fwd_h(
            **kwargs, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if cpu_mirror else None,
        )
        repeated = chunk_gated_delta_rule_fwd_h(
            **kwargs, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if cpu_mirror else None,
        )
        assert precompute.call_count == 1
        for actual, expected in zip(repeated, original):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # Move the long sequence without changing the offsets buffer or sub-sequence length.
        cu_cpu.copy_(torch.tensor([0, boundary, 32768], dtype=torch.int32))
        cu.copy_(cu_cpu)
        changed = chunk_gated_delta_rule_fwd_h(
            **kwargs, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if cpu_mirror else None,
        )
        repeated = chunk_gated_delta_rule_fwd_h(
            **kwargs, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if cpu_mirror else None,
        )
        fresh = chunk_gated_delta_rule_fwd_h(
            **kwargs, cu_seqlens=cu.clone(), cu_seqlens_cpu=cu_cpu.clone() if cpu_mirror else None,
        )
        for actual, cached, expected in zip(changed, repeated, fresh):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(cached, expected, atol=0, rtol=0)
        assert precompute.call_count == 3


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not IS_NVIDIA, reason="requires NVIDIA")
@pytest.mark.parametrize('operator', [chunk_kda, chunk_gated_delta_rule], ids=['kda', 'gdn'])
@pytest.mark.parametrize('cpu_mirror', [False, True])
@pytest.mark.parametrize(('T', 'before', 'after'), [(512, 128, 320), (32768, 24576, 8192)])
def test_chunk_intracard_inplace_offsets(monkeypatch, operator, cpu_mirror, T, before, after):
    """Public operators rebuild metadata after a versioned packed offsets buffer changes."""
    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    torch.manual_seed(42)
    q = torch.randn(1, T, 1, 32, device=device, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(q)
    v = torch.randn_like(q)
    g = torch.full(q.shape if operator is chunk_kda else q.shape[:-1], -0.05, device=device)
    beta = torch.rand(1, T, 1, device=device, dtype=torch.bfloat16)
    h0 = torch.randn(2, 1, 32, 32, device=device)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True)
    cu_cpu = torch.tensor([0, before, T], dtype=torch.int32)
    cu = cu_cpu.to(device)

    with torch.inference_mode():
        for boundary in (before, after, 0, before):
            cu_cpu.copy_(torch.tensor([0, boundary, T], dtype=torch.int32))
            cu.copy_(cu_cpu)
            actual = operator(**kwargs, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if cpu_mirror else None)
            expected = operator(**kwargs, cu_seqlens=cu.clone(), cu_seqlens_cpu=cu_cpu.clone() if cpu_mirror else None)
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_intracard_backend_disabled_when_env_var_is_zero(monkeypatch):
    """Verify that IntraCardCPBackend is disabled when FLA_INTRACARD_CP=0."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    monkeypatch.setenv("FLA_INTRACARD_CP", "0")
    assert IntraCardCPBackend.is_enabled() is False


def test_intracard_backend_enabled_when_env_var_is_one(monkeypatch):
    """Verify that IntraCardCPBackend is enabled when FLA_INTRACARD_CP=1."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    assert IntraCardCPBackend.is_enabled() is True


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gdn_intracard_gqa(monkeypatch):
    """E2E: chunk_gated_delta_rule intracard path produces correct results with GQA (Hq < H).

    Uses a long varlen sequence to exercise the intracard split path,
    with Hq=2 key/query heads and H=4 value/output heads.
    """
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # T must be large enough to bypass early_return in intracard_fwd_h.
    B, T, Hq, H, D = 1, 32768, 2, 4, 64

    q = F.normalize(torch.randn(B, T, Hq, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, Hq, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()

    # Run with intracard path (inference_mode triggers it)
    with torch.inference_mode():
        o_intra, ht_intra = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=True,
        )

    # Run without intracard: disable the backend temporarily
    from fla.ops.common.backends import common_registry
    saved_backends = common_registry._backends.copy()
    common_registry._backends.clear()
    try:
        with torch.inference_mode():
            o_ref, ht_ref = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                output_final_state=True,
            )
    finally:
        common_registry._backends = saved_backends

    assert torch.allclose(o_intra, o_ref, atol=1e-2, rtol=1e-2), \
        f"Output mismatch: max diff={(o_intra - o_ref).abs().max().item()}"
    assert torch.allclose(ht_intra, ht_ref, atol=1e-2, rtol=1e-2), \
        f"Final state mismatch: max diff={(ht_intra - ht_ref).abs().max().item()}"
