# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check intra-card CP routing, cache reuse, and grouped-head outputs and final states."""

import os

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


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not (IS_TF32_SUPPORTED or IS_NPU), reason="requires NVIDIA Ampere or newer, or Ascend")
def test_chunk_gdn_grouped_heads(monkeypatch):
    """Compare intra-card splitting with the ordinary chunk path when query heads are shared by value heads."""
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    monkeypatch.setenv("FLA_TILELANG", "0")
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # the sequence must be long enough to bypass early_return in intracard_fwd_h.
    B, T, Hq, H, D = 1, 32768, 2, 4, 64

    q = F.normalize(torch.randn(B, T, Hq, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, Hq, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()

    split_calls = 0
    original_precompute = intracard_cp._precompute_intracard_indices

    def counted_precompute(*args, **kwargs):
        nonlocal split_calls
        split_calls += 1
        return original_precompute(*args, **kwargs)

    # index construction occurs only after the backend actually splits the sequence
    monkeypatch.setattr(intracard_cp, "_precompute_intracard_indices", counted_precompute)
    with torch.inference_mode():
        o_intra, ht_intra = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    assert split_calls == 1, "the grouped-head case must execute intra-card splitting"

    common_registry = backends._load_operation_registry('common')
    with monkeypatch.context() as patch:
        patch.setattr(common_registry, '_backends', {})
        with torch.inference_mode():
            o_ref, ht_ref = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )

    assert split_calls == 1, "the reference must bypass intra-card splitting"
    assert torch.allclose(o_intra, o_ref, atol=1e-2, rtol=1e-2), (
        f"Output mismatch: max diff={(o_intra - o_ref).abs().max().item()}"
    )
    assert torch.allclose(ht_intra, ht_ref, atol=1e-2, rtol=1e-2), (
        f"Final state mismatch: max diff={(ht_intra - ht_ref).abs().max().item()}"
    )
