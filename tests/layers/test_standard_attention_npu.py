# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers.attn import Attention
from fla.models.utils import Cache
from fla.ops.attn.standard import standard_attention
from fla.utils import IS_NPU, device


def _npu_available():
    if not hasattr(torch, "npu"):
        return False
    try:
        return torch.npu.is_available()
    except (RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(not IS_NPU or not _npu_available(), reason="requires an available Ascend NPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_standard_attention_inference_matches_reference(monkeypatch, dtype):
    torch.manual_seed(0)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(1, 3, 4, 16, dtype=dtype, device=device)
    k = torch.randn(1, 3, 2, 16, dtype=dtype, device=device)
    v = torch.randn(1, 3, 2, 16, dtype=dtype, device=device)

    actual = standard_attention(q, k, v, training=False).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu())
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_npu_standard_attention_auto_falls_back_for_unsupported_head_dim(monkeypatch):
    torch.manual_seed(10)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "auto")
    # Ascend's native BNSD kernel requires a larger aligned head dimension;
    # auto mode must preserve correctness through the generic fallback.
    q = torch.randn(1, 3, 4, 8, dtype=torch.float16, device=device)
    k = torch.randn(1, 3, 2, 8, dtype=torch.float16, device=device)
    v = torch.randn(1, 3, 2, 8, dtype=torch.float16, device=device)

    actual = standard_attention(q, k, v, training=False).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu())
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_npu_standard_attention_decode_matches_reference(monkeypatch):
    torch.manual_seed(1)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(1, 1, 4, 16, dtype=torch.float16, device=device)
    k = torch.randn(1, 5, 2, 16, dtype=torch.float16, device=device)
    v = torch.randn(1, 5, 2, 16, dtype=torch.float16, device=device)

    actual = standard_attention(q, k, v, training=False).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu())
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_standard_attention_training_backward_matches_reference(monkeypatch, dtype):
    torch.manual_seed(2)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(1, 3, 4, 16, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(1, 3, 2, 16, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(1, 3, 2, 16, dtype=dtype, device=device, requires_grad=True)
    do = torch.randn_like(q)

    output = standard_attention(q, k, v, training=True)
    (output * do).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    q_ref = q.detach().cpu().requires_grad_(True)
    k_ref = k.detach().cpu().requires_grad_(True)
    v_ref = v.detach().cpu().requires_grad_(True)
    do_ref = do.cpu()
    output_ref = standard_attention(q_ref, k_ref, v_ref)
    (output_ref * do_ref).sum().backward()

    torch.testing.assert_close(output.cpu(), output_ref, rtol=2e-1, atol=2e-1)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad)):
        torch.testing.assert_close(actual_grad.cpu(), expected_grad, rtol=2e-1, atol=2e-1)


def test_npu_standard_attention_window_matches_reference(monkeypatch):
    torch.manual_seed(5)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(1, 4, 4, 16, dtype=torch.float16, device=device)
    k = torch.randn(1, 6, 2, 16, dtype=torch.float16, device=device)
    v = torch.randn(1, 6, 2, 16, dtype=torch.float16, device=device)

    actual = standard_attention(q, k, v, training=False, window_size=3).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu(), window_size=3)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_npu_standard_attention_multitoken_decode_matches_reference(monkeypatch):
    torch.manual_seed(6)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(1, 2, 4, 16, dtype=torch.float16, device=device)
    k = torch.randn(1, 5, 2, 16, dtype=torch.float16, device=device)
    v = torch.randn(1, 5, 2, 16, dtype=torch.float16, device=device)

    actual = standard_attention(q, k, v, training=False).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu())
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_npu_standard_attention_long_sequence_matches_reference(monkeypatch):
    torch.manual_seed(7)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(2, 32, 4, 16, dtype=torch.float16, device=device)
    k = torch.randn(2, 32, 2, 16, dtype=torch.float16, device=device)
    v = torch.randn(2, 32, 2, 16, dtype=torch.float16, device=device)

    actual = standard_attention(q, k, v, training=False).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu())
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_npu_standard_attention_packed_fallback_matches_reference(monkeypatch):
    torch.manual_seed(3)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "auto")
    q = torch.randn(5, 4, 16, dtype=torch.float16, device=device)
    k = torch.randn(5, 2, 16, dtype=torch.float16, device=device)
    v = torch.randn(5, 2, 16, dtype=torch.float16, device=device)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 3, 5], dtype=torch.int32)

    actual = standard_attention(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k).cpu()

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    expected = standard_attention(q.cpu(), k.cpu(), v.cpu(), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_npu_attention_layer_prefill_decode_matches_full_sequence(monkeypatch):
    torch.manual_seed(4)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    full_layer = Attention(hidden_size=64, num_heads=4, num_kv_heads=2, layer_idx=0).to(
        device=device, dtype=torch.float16
    ).eval()
    cached_layer = Attention(hidden_size=64, num_heads=4, num_kv_heads=2, layer_idx=0).to(
        device=device, dtype=torch.float16
    ).eval()
    cached_layer.load_state_dict(full_layer.state_dict())
    hidden_states = torch.randn(1, 5, 64, dtype=torch.float16, device=device)

    full_output, _, _ = full_layer(hidden_states)
    cache = Cache()
    cached_layer(hidden_states[:, :4], past_key_values=cache, use_cache=True)
    decode_output, _, _ = cached_layer(hidden_states[:, 4:], past_key_values=cache, use_cache=True)

    torch.testing.assert_close(full_output[:, 4:].cpu(), decode_output.cpu(), rtol=5e-2, atol=5e-2)
