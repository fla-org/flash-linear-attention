# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check intra-card CP routing, cache reuse, and packed/grouped-head outputs and final states."""

import os
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
import torch.nn.functional as F

from fla import backends
from fla.ops.common import intracard_cp
from fla.ops.common.backends.intracard import IntraCardCPBackend
from fla.ops.common.intracard_cp import _intracard_cache
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
from fla.utils import IS_NPU, IS_TF32_SUPPORTED, device


@pytest.fixture(autouse=True)
def _clear_intracard_cache():
    _intracard_cache.clear()
    yield
    _intracard_cache.clear()


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason="requires NVIDIA Ampere or newer, or Ascend")
def test_chunk_kda_cache(monkeypatch):
    """Reuse the intracard precompute cache for repeated calls with the same cu_seqlens object."""
    # intracard CP is disabled by default.
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    # Ascend fwd_h is priority 0. Raise intra-card above it for this test only.
    monkeypatch.setattr(IntraCardCPBackend, 'priority', -1)
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # with chunk_size=64 and MIN_SUBSEQ_CHUNKS=128, subseq_len is at least 8192.
    # the 32768-token sequence bypasses early_return and reaches the split threshold of 3 * subseq_len.
    B, T, H, D = 1, 32768, 1, 32

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
    original_precompute = intracard_cp._precompute_intracard_indices

    def counted_precompute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_precompute(*args, **kwargs)

    monkeypatch.setattr(intracard_cp, "_precompute_intracard_indices", counted_precompute)

    with torch.inference_mode():
        o1, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        o2, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    assert call_count == 1, "second call should hit cache and skip precompute"
    assert len(_intracard_cache) == 1
    key = next(iter(_intracard_cache))
    entry = _intracard_cache[key]
    assert key[0] == id(cu_seqlens)
    assert entry.cu_seqlens_ref() is cu_seqlens
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)


def test_backend_default():
    """Verify that IntraCardCPBackend is disabled by default."""
    assert IntraCardCPBackend.default_enable is False


@pytest.mark.parametrize('enabled', [False, True], ids=['disabled', 'enabled'])
def test_backend_enabled(monkeypatch, enabled: bool):
    """The environment switch overrides the backend's disabled default."""
    monkeypatch.setenv("FLA_INTRACARD_CP", "1" if enabled else "0")
    assert IntraCardCPBackend().is_enabled() is enabled


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason='requires NVIDIA Ampere or newer, or Ascend')
@pytest.mark.parametrize(('op', 'Hq', 'H'), [
    pytest.param(chunk_gated_delta_rule, 2, 4, id='gdn-grouped'),
    pytest.param(chunk_kda, 1, 1, id='kda'),
])
def test_chunk_packed_boundaries(monkeypatch, op, Hq, H):
    """Preserve packed/grouped-head outputs, final states, and sample independence under splitting."""
    # Ascend fwd_h is priority 0. Raise intra-card above it for this test only.
    monkeypatch.setattr(IntraCardCPBackend, 'priority', -1)
    for name in ('FLA_FLASH_QLA', 'FLA_FLASH_KDA', 'FLA_TILELANG'):
        monkeypatch.setenv(name, '0')
    torch.manual_seed(42)
    # one batch covers split sequences separated by a short sequence and adjacent split sequences
    lengths = (32768, 64, 32768, 32768)
    q, k = [F.normalize(torch.randn(1, sum(lengths), Hq, 64, device=device), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, sum(lengths), H, 64, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(v.shape if op is chunk_kda else v.shape[:-1], device=device))
    beta = torch.randn(v.shape[:-1], device=device).sigmoid()
    cu_seqlens_cpu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, output_final_state=True,
                  cu_seqlens=cu_seqlens_cpu.to(device), cu_seqlens_cpu=cu_seqlens_cpu)

    with torch.inference_mode():
        monkeypatch.setenv('FLA_INTRACARD_CP', '0')
        o_ref, ht_ref = op(**kwargs)
        assert not _intracard_cache, 'the reference must bypass intra-card splitting'

        monkeypatch.setenv('FLA_INTRACARD_CP', '1')
        o_intra, ht_intra = op(**kwargs)
        assert len(_intracard_cache) == 1, 'the test must execute intra-card splitting'
        split_info = next(iter(_intracard_cache.values())).split_info
        assert split_info.split_seq_ids == [0, 2, 3]
        assert all(n > 1 for n in split_info.num_subseqs)

        short_start, short_end = cu_seqlens_cpu[1:3].tolist()
        changed_v = v.clone()
        changed_v[:, short_start:short_end].add_(1)
        o_changed, _ = op(**{**kwargs, 'v': changed_v})

    assert torch.equal(o_intra[:, short_end:], o_changed[:, short_end:]), (
        "changing the short sequence must not change later sequences' outputs"
    )
    torch.testing.assert_close(o_intra, o_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ht_intra, ht_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(backends._DISPATCH_DISABLED, reason='Backend dispatch was disabled before import')
@pytest.mark.parametrize('extra', [{}, {'chunk_offsets': None}], ids=['chunk_offsets_omitted', 'chunk_offsets_none'])
def test_intracard_backend_dispatch(monkeypatch, extra):
    """KDA's eager forward passes chunk_offsets=None; the opt-in intra-card backend must still be selected."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    monkeypatch.setattr(backends.TritonAscendBackend, 'is_available', classmethod(lambda cls: False))
    monkeypatch.setenv('FLA_INTRACARD_CP', '1')
    result = object()
    implementation = create_autospec(IntraCardCPBackend.chunk_gated_delta_rule_fwd_h, return_value=result)
    monkeypatch.setattr(IntraCardCPBackend, 'chunk_gated_delta_rule_fwd_h', implementation)

    x = SimpleNamespace(shape=(1, 64, 2, 64))
    with torch.inference_mode():
        assert chunk_gated_delta_rule_fwd_h(k=x, w=x, u=x, cu_seqlens=torch.tensor([0, 64]), **extra) is result
    implementation.assert_called_once()


def test_intracard_backend_verifier_rejects_graph_mode():
    """Graph-mode callers pass fixed-shape chunk_offsets; intra-card CP cannot be captured."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    x = SimpleNamespace(shape=(1, 64, 2, 64))
    with torch.inference_mode():
        verdict = IntraCardCPBackend().chunk_gated_delta_rule_fwd_h_verifier(
            k=x, w=x, u=x, cu_seqlens=torch.tensor([0, 64]), chunk_offsets=torch.tensor([0, 1]),
        )
    assert verdict == (False, 'chunk_offsets is given (CUDA Graph mode)')
