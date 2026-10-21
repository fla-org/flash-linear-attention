# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""End-to-end cache tests for intracard CP via chunk_kda."""

import os

import pytest
import torch

from fla.ops.common import intracard_cp
from fla.ops.common.intracard_cp import _intracard_cache
from fla.ops.cp import context
from fla.ops.kda import chunk_kda
from fla.utils import IS_TF32_SUPPORTED, assert_close, device


@pytest.fixture(autouse=True)
def clear_intracard_cache():
    _intracard_cache.clear()
    yield
    _intracard_cache.clear()


@pytest.mark.parametrize('entry', ['fwd', 'pre_scan', 'merge'])
@pytest.mark.parametrize('legacy_false', [False, True], ids=['unsupported-device', 'legacy-false'])
def test_intracard_precision_requirement(entry, legacy_false, monkeypatch):
    monkeypatch.setattr(context, 'IS_NVIDIA', False)
    x = torch.empty(1, 64, 1, 32)
    kwargs = {'use_tf32x3_affine_chain': False} if legacy_false else {}
    error = ValueError if legacy_false else RuntimeError
    message = 'use_tf32x3_affine_chain=False' if legacy_false else 'compute capability >= 8.0'
    with pytest.raises(error, match=message):
        if entry == 'fwd':
            intracard_cp.intracard_fwd_h(k=x, w=x, u=x, **kwargs)
        elif entry == 'pre_scan':
            intracard_cp.intracard_pre_scan(
                k=x,
                w=x,
                u=x,
                g=None,
                gk=None,
                cu_seqlens=None,
                num_subseqs=0,
                **kwargs,
            )
        else:
            intracard_cp.intracard_merge(
                hm=x,
                split_info=None,
                num_non_first=0,
                merge_seq_offsets=[],
                merge_init_offsets=[],
                device=x.device,
                **kwargs,
            )


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not IS_TF32_SUPPORTED, reason="requires NVIDIA Ampere or newer")
def test_chunk_kda_intracard_cache_hit_same_cu_seqlens_object(monkeypatch):
    """Reuse the intracard precompute cache for repeated calls with the same cu_seqlens object."""
    # intracard CP is disabled by default.
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # with chunk_size=64 and MIN_SUBSEQ_CHUNKS=128, subseq_len is at least 8192.
    # this T=32768 case bypasses early_return (2 * subseq_len) and exercises splitting (3 * subseq_len).
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

    assert IntraCardCPBackend.default_enable is False


def test_intracard_backend_disabled_when_env_var_is_zero(monkeypatch):
    """Verify that IntraCardCPBackend is disabled when FLA_INTRACARD_CP=0."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    monkeypatch.setenv("FLA_INTRACARD_CP", "0")
    assert IntraCardCPBackend().is_enabled() is False


def test_intracard_backend_enabled_when_env_var_is_one(monkeypatch):
    """Verify that IntraCardCPBackend is enabled when FLA_INTRACARD_CP=1."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    assert IntraCardCPBackend().is_enabled() is True


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(not IS_TF32_SUPPORTED, reason="requires NVIDIA Ampere or newer")
def test_chunk_gdn_intracard_gqa(monkeypatch):
    """E2E: chunk_gated_delta_rule intracard path produces correct results with GQA (Hq < H).

    Uses a long varlen sequence to exercise the intracard split path,
    with Hq=2 key/query heads and H=4 value/output heads.
    """
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
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

    # inference mode enables the intracard path.
    with torch.inference_mode():
        o_intra, ht_intra = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=True,
        )

    from fla import backends

    common_registry = backends._load_operation_registry('common')
    saved_backends = common_registry._backends.copy()
    common_registry._backends.clear()
    try:
        with torch.inference_mode():
            o_ref, ht_ref = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
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


@pytest.mark.skipif(not IS_TF32_SUPPORTED, reason='requires NVIDIA Ampere or newer')
@pytest.mark.parametrize('vector_gate', [False, True], ids=['scalar-gate', 'vector-gate'])
@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize(('K', 'V'), [(192, 64), (256, 80)])
def test_intracard_pre_scan_merge(vector_gate: bool, state_v_first: bool, K: int, V: int):
    torch.manual_seed(42)
    T, H, HV = 319, 2, 4
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16) * 0.05
    w = torch.randn(1, T, HV, K, device=device, dtype=torch.bfloat16) * 0.05
    u = torch.randn(1, T, HV, V, device=device, dtype=torch.bfloat16)
    decay = -0.01 * (torch.arange(T, device=device) % 64 + 1)
    if vector_gate:
        g, gk = None, decay.view(1, T, 1, 1).expand(1, T, HV, K).contiguous()
    else:
        g, gk = decay.view(1, T, 1).expand(1, T, HV).contiguous(), None
    state_shape = (1, HV, V, K) if state_v_first else (1, HV, K, V)
    h0 = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.1
    cu_seqlens_split = torch.tensor([0, 128, 256, T], device=device, dtype=torch.int32)
    if vector_gate:
        # scalar and vector gates must not share an autotuned pipeline configuration
        intracard_cp.intracard_pre_scan(
            k=k,
            w=w,
            u=u,
            g=decay.view(1, T, 1).expand(1, T, HV).contiguous(),
            gk=None,
            cu_seqlens=cu_seqlens_split,
            num_subseqs=3,
        )
    hm = intracard_cp.intracard_pre_scan(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens_split,
        num_subseqs=3,
    )
    states, count = intracard_cp.intracard_merge(
        hm=hm,
        split_info=intracard_cp.SplitSeqInfo(split_seq_ids=[0], start_subseq_idx=[0], num_subseqs=[3]),
        num_non_first=2,
        merge_seq_offsets=[0, 3],
        merge_init_offsets=[0, 2],
        device=k.device,
        initial_state=h0,
        state_v_first=state_v_first,
    )
    ref_h, _, _ = intracard_cp._raw_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=h0,
        state_v_first=state_v_first,
        cu_seqlens=torch.tensor([0, T], device=device, dtype=torch.int32),
    )
    assert count == 2
    assert torch.isfinite(hm).all()
    assert_close('h', ref_h[0, [2, 4]], states, 0.006)
