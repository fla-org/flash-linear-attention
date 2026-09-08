# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib

import pytest
import torch

from fla.layers.attn import Attention
from fla.models.utils import Cache
from fla.ops.attn.standard import select_standard_attention_backend, standard_attention

standard_backend = importlib.import_module("fla.ops.attn.standard")


class _IdentityRotary(torch.nn.Module):
    """Keep these backend tests independent of device-specific rotary kernels."""

    def forward(self, q, k, **kwargs):
        return q, k


def _layer(**kwargs):
    layer = Attention(**kwargs).to(device="cpu").eval()
    layer.rotary = _IdentityRotary()
    return layer


@pytest.fixture(autouse=True)
def _force_cpu_fallback(monkeypatch):
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")


def test_standard_attention_gqa_and_bottom_right_causal_alignment():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 5, 2, 8)
    v = torch.randn(1, 5, 2, 8)

    actual = standard_attention(q, k, v)
    k_expanded = k.repeat_interleave(2, dim=2)
    v_expanded = v.repeat_interleave(2, dim=2)
    scores = torch.matmul(q.transpose(1, 2), k_expanded.transpose(1, 2).transpose(-2, -1)) / 8**0.5
    # The one-token query is the last token of the five-token KV sequence.
    expected = torch.softmax(scores, dim=-1) @ v_expanded.transpose(1, 2)
    torch.testing.assert_close(actual, expected.transpose(1, 2))


@pytest.mark.parametrize(
    ('q_len', 'kv_len', 'window_size'),
    [(4, 4, None), (3, 5, None), (2, 5, 3), (5, 3, None), (4, 4, 2)],
)
def test_reference_and_sdpa_match_forward_and_backward(q_len, kv_len, window_size, monkeypatch):
    torch.manual_seed(q_len * 10 + kv_len)
    q = torch.randn(2, q_len, 4, 8, requires_grad=True)
    k = torch.randn(2, kv_len, 2, 8, requires_grad=True)
    v = torch.randn(2, kv_len, 2, 6, requires_grad=True)
    do = torch.randn(2, q_len, 4, 6)

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    ref = standard_attention(q, k, v, window_size=window_size)
    (ref * do).sum().backward()
    ref_grads = tuple(x.grad.detach().clone() for x in (q, k, v))

    q.grad = k.grad = v.grad = None
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "sdpa")
    actual = standard_attention(q, k, v, window_size=window_size)
    (actual * do).sum().backward()

    torch.testing.assert_close(actual, ref, rtol=1e-5, atol=1e-5)
    for actual_grad, ref_grad in zip((q.grad, k.grad, v.grad), ref_grads):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=1e-5, atol=1e-5)


def test_packed_reference_and_sdpa_match_with_unequal_sequence_lengths(monkeypatch):
    torch.manual_seed(3)
    q = torch.randn(5, 4, 8, requires_grad=True)
    k = torch.randn(5, 2, 8, requires_grad=True)
    v = torch.randn(5, 2, 6, requires_grad=True)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    do = torch.randn(5, 4, 6)

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "reference")
    ref = standard_attention(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
    (ref * do).sum().backward()
    ref_grads = tuple(x.grad.detach().clone() for x in (q, k, v))

    q.grad = k.grad = v.grad = None
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "sdpa")
    actual = standard_attention(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
    (actual * do).sum().backward()

    torch.testing.assert_close(actual, ref, rtol=1e-5, atol=1e-5)
    for actual_grad, ref_grad in zip((q.grad, k.grad, v.grad), ref_grads):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ('cu_seqlens_q', 'cu_seqlens_k', 'message'),
    [
        (torch.tensor([1, 2]), torch.tensor([0, 1]), "start at 0"),
        (torch.tensor([0, 2, 1]), torch.tensor([0, 1, 2]), "non-decreasing"),
        (torch.tensor([0, 1]), torch.tensor([0, 1]), "packed tensor length"),
        (torch.tensor([0, 2, 3]), torch.tensor([0, 1, 2]), "max_seqlen_q"),
    ],
)
def test_packed_input_contract_rejects_invalid_lengths(cu_seqlens_q, cu_seqlens_k, message):
    q = torch.randn(3, 2, 4)
    k = torch.randn(2, 1, 4)
    v = torch.randn(2, 1, 4)
    kwargs = dict(cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
    if message == "max_seqlen_q":
        kwargs["max_seqlen_q"] = 1
        kwargs["max_seqlen_k"] = 1
    with pytest.raises(ValueError, match=message):
        standard_attention(q, k, v, **kwargs)


def test_backend_selection_contract(monkeypatch):
    q = torch.randn(1, 2, 2, 4)
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "auto")
    assert select_standard_attention_backend(q, q, q) == "sdpa"

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "invalid")
    with pytest.raises(ValueError, match="invalid"):
        select_standard_attention_backend(q, q, q)

    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "flash_attn")
    with pytest.raises(RuntimeError, match="CUDA/HIP"):
        select_standard_attention_backend(q, q, q)


def test_attention_dense_packed_padding_and_cache_paths():
    torch.manual_seed(1)
    layer = _layer(hidden_size=32, num_heads=4, num_kv_heads=2, layer_idx=0)
    hidden_states = torch.randn(2, 4, 32)

    dense, _, _ = layer(hidden_states)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed, _, _ = layer(hidden_states, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(dense, packed)

    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    masked, _, _ = layer(hidden_states, attention_mask=attention_mask)
    assert torch.equal(masked[0, 3], torch.zeros_like(masked[0, 3]))
    assert torch.equal(masked[1, 2:], torch.zeros_like(masked[1, 2:]))


def test_attention_fallback_supports_backward(monkeypatch):
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "sdpa")
    layer = _layer(hidden_size=32, num_heads=4, num_kv_heads=2, layer_idx=0).train()
    hidden_states = torch.randn(2, 3, 32, requires_grad=True)
    output, _, _ = layer(hidden_states)
    output.square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_attention_prefill_decode_cache_matches_full_sequence():
    torch.manual_seed(2)
    full_layer = _layer(hidden_size=32, num_heads=4, num_kv_heads=2, layer_idx=0)
    cached_layer = _layer(hidden_size=32, num_heads=4, num_kv_heads=2, layer_idx=0)
    cached_layer.load_state_dict(full_layer.state_dict())
    hidden_states = torch.randn(1, 5, 32)

    full_output, _, _ = full_layer(hidden_states)
    cache = Cache()
    cached_layer(hidden_states[:, :4], past_key_values=cache, use_cache=True)
    decode_output, _, _ = cached_layer(hidden_states[:, 4:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(full_output[:, 4:], decode_output)


def test_attention_output_attentions_contract_is_explicit():
    layer = _layer(hidden_size=32, num_heads=4, num_kv_heads=2, layer_idx=0)
    hidden_states = torch.randn(1, 2, 32)
    _, attentions, _ = layer(hidden_states, output_attentions=True)
    assert attentions is None


def test_forced_npu_backend_rejects_non_npu_inputs(monkeypatch):
    monkeypatch.setenv("FLA_STANDARD_ATTN_BACKEND", "npu")
    q = torch.randn(1, 2, 2, 4)
    with pytest.raises(RuntimeError, match="requires an NPU tensor"):
        standard_attention(q, q, q)


def test_npu_backend_builds_bnsd_gqa_arguments(monkeypatch):
    class _FakeNPU:
        def __init__(self):
            self.calls = []

        def npu_fused_infer_attention_score(self, q, k, v, **kwargs):
            self.calls.append(("infer", q, k, v, kwargs))
            return q, None

        def npu_fusion_attention(self, q, k, v, head_num, input_layout, **kwargs):
            self.calls.append(("train", q, k, v, head_num, input_layout, kwargs))
            return q, None, None, None, 0, 0, 0

    fake_npu = _FakeNPU()
    monkeypatch.setattr(standard_backend, "_load_torch_npu", lambda: fake_npu)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 5, 2, 8)
    v = torch.randn(2, 5, 2, 8)

    output = standard_backend._npu_attention(
        q, k, v, scale=8**-0.5, causal=True, window_size=None, query_start=None, training=False
    )
    assert output.shape == q.shape
    kind, q_bnsd, k_bnsd, v_bnsd, kwargs = fake_npu.calls[-1]
    assert kind == "infer"
    assert (q_bnsd.shape, k_bnsd.shape, v_bnsd.shape) == ((2, 4, 3, 8), (2, 2, 5, 8), (2, 2, 5, 8))
    assert kwargs["input_layout"] == "BNSD"
    assert kwargs["num_heads"] == 4
    assert kwargs["num_key_value_heads"] == 2
    assert kwargs["sparse_mode"] == 1

    standard_backend._npu_attention(
        q, k, v, scale=8**-0.5, causal=True, window_size=3, query_start=None, training=True
    )
    assert fake_npu.calls[-1][0] == "train"
    assert fake_npu.calls[-1][-1]["sparse_mode"] == 1


@pytest.mark.parametrize(
    ('q_len', 'kv_len', 'window_size', 'message'),
    [
        (1, 5, 2, "masked single-query"),
        (5, 3, None, "fully masked rows"),
    ],
)
def test_npu_backend_rejects_unsupported_mask_boundaries(
    monkeypatch, q_len, kv_len, window_size, message
):
    class _FakeNPU:
        def npu_fused_infer_attention_score(self, *args, **kwargs):
            raise AssertionError("the native operator must not be called")

    monkeypatch.setattr(standard_backend, "_load_torch_npu", lambda: _FakeNPU())
    q = torch.randn(1, q_len, 2, 4)
    k = torch.randn(1, kv_len, 2, 4)
    v = torch.randn(1, kv_len, 2, 4)
    with pytest.raises(RuntimeError, match=message):
        standard_backend._npu_attention(
            q,
            k,
            v,
            scale=4**-0.5,
            causal=True,
            window_size=window_size,
            query_start=None,
            training=False,
        )
