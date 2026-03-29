"""End-to-end cache tests for intracard CP via chunk_kda."""

import pytest
import torch

import fla.ops.common.intracard_cp as intracard_cp_mod
from fla.ops.common.intracard_cp import _intracard_cache
from fla.ops.kda import chunk_kda
from fla.utils import device


@pytest.fixture(autouse=True)
def clear_intracard_cache():
    _intracard_cache.clear()
    yield
    _intracard_cache.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_kda_intracard_cache_hit_same_cu_seqlens_object(monkeypatch):
    """E2E: chunk_kda should reuse intracard precompute cache on second call.

    This test intentionally uses a very long varlen sequence so that:
    1) intracard path is selected in inference mode, and
    2) early_return is bypassed and split path is exercised.
    """
    # Enable intracard CP backend explicitly as it's disabled by default
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # T must be large enough to bypass early_return in intracard_fwd_h.
    # With chunk_size=64 and MIN_SUBSEQ_CHUNKS=128, subseq_len floor is 8192.
    # We choose T=32768 to satisfy both:
    #   - early_return check: seq_len >= 2 * subseq_len
    #   - split threshold: seq_len >= 3 * subseq_len
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
    original_precompute = intracard_cp_mod._precompute_intracard_indices

    def counted_precompute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_precompute(*args, **kwargs)

    monkeypatch.setattr(intracard_cp_mod, "_precompute_intracard_indices", counted_precompute)

    with torch.inference_mode():
        o1, _ = chunk_kda(
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
        )
        o2, _ = chunk_kda(
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
        )

    assert call_count == 1, "second call should hit cache and skip precompute"
    assert len(_intracard_cache) == 1
    key = next(iter(_intracard_cache))
    entry = _intracard_cache[key]
    assert key[0] == id(cu_seqlens)
    assert entry.cu_seqlens_ref() is cu_seqlens
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)


def test_intracard_backend_disabled_by_default():
    """Verify that IntraCardCPBackend is disabled by default."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    # When env var is not set, backend should be disabled (default_enable=False)
    assert IntraCardCPBackend.default_enable is False


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gdn_intracard_gqa(monkeypatch):
    """E2E: chunk_gated_delta_rule intracard path produces correct results with GQA (Hq < H).

    Uses a long varlen sequence to exercise the intracard split path,
    with Hq=2 key/query heads and H=4 value/output heads.
    """
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

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
