# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.layers import HigherOrderLinearAttention
from fla.ops.hla import recurrent_hla


def _signed_clamp_min(x: torch.Tensor, eps: float) -> torch.Tensor:
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    return sign * x.abs().clamp_min(eps)


def _parallel_masked_hla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale
    weights = torch.einsum("bthd,bshd->bhts", q, k)
    mask = torch.tril(torch.ones(q.shape[1], q.shape[1], dtype=torch.bool, device=q.device))
    weights = weights.masked_fill(~mask.view(1, 1, q.shape[1], q.shape[1]), 0.0)
    second_order = torch.matmul(weights, weights.transpose(-1, -2))
    second_order = second_order.masked_fill(~mask.view(1, 1, q.shape[1], q.shape[1]), 0.0)
    output = torch.einsum("bhts,bshv->bthv", second_order, v)
    if normalize:
        den = second_order.sum(dim=-1).transpose(1, 2).unsqueeze(-1)
        output = output / _signed_clamp_min(den, eps)
    return output


def test_recurrent_hla_matches_parallel_masked_form():
    torch.manual_seed(0)
    q = torch.randn(2, 7, 3, 5, dtype=torch.float64)
    k = torch.randn(2, 7, 3, 5, dtype=torch.float64)
    v = torch.randn(2, 7, 3, 4, dtype=torch.float64)

    actual, _ = recurrent_hla(q, k, v)
    expected = _parallel_masked_hla(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_recurrent_hla_chunked_cache_matches_full_sequence():
    torch.manual_seed(1)
    q = torch.randn(2, 9, 4, 6)
    k = torch.randn(2, 9, 4, 6)
    v = torch.randn(2, 9, 4, 3)

    full, _ = recurrent_hla(q, k, v)
    first, state = recurrent_hla(q[:, :4], k[:, :4], v[:, :4], output_final_state=True)
    second, _ = recurrent_hla(q[:, 4:], k[:, 4:], v[:, 4:], initial_state=state)
    torch.testing.assert_close(torch.cat([first, second], dim=1), full, rtol=1e-5, atol=1e-5)


def test_recurrent_hla_backward_matches_parallel_masked_form():
    torch.manual_seed(2)
    q = torch.randn(1, 5, 2, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 5, 2, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 5, 2, 3, dtype=torch.float64, requires_grad=True)

    actual, _ = recurrent_hla(q, k, v)
    actual.sum().backward()
    actual_grads = [x.grad.detach().clone() for x in (q, k, v)]

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    expected = _parallel_masked_hla(q_ref, k_ref, v_ref)
    expected.sum().backward()
    expected_grads = [x.grad.detach().clone() for x in (q_ref, k_ref, v_ref)]

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)


def test_recurrent_hla_normalized_preserves_denominator_sign():
    q = torch.tensor([[[[1.0]], [[-0.1]]]], dtype=torch.float64)
    k = torch.ones_like(q)
    v = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float64)
    weights = torch.einsum("bthd,bshd->bhts", q, k)
    mask = torch.tril(torch.ones(q.shape[1], q.shape[1], dtype=torch.bool, device=q.device))
    second_order = torch.matmul(
        weights.masked_fill(~mask.view(1, 1, q.shape[1], q.shape[1]), 0.0),
        weights.masked_fill(~mask.view(1, 1, q.shape[1], q.shape[1]), 0.0).transpose(-1, -2),
    ).masked_fill(~mask.view(1, 1, q.shape[1], q.shape[1]), 0.0)
    assert second_order.sum(dim=-1)[0, 0, 1] < 0

    expected = _parallel_masked_hla(q, k, v, normalize=True)
    actual, _ = recurrent_hla(q, k, v, normalize=True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_hla_layer_forward_shape():
    torch.manual_seed(3)
    layer = HigherOrderLinearAttention(hidden_size=32, num_heads=4, head_dim=8, output_norm="identity")
    x = torch.randn(2, 6, 32)
    y, attentions, cache = layer(x)
    assert y.shape == x.shape
    assert attentions is None
    assert cache is None


def test_hla_layer_attention_mask_excludes_padded_qkv_state():
    torch.manual_seed(4)
    layer = HigherOrderLinearAttention(hidden_size=16, num_heads=2, head_dim=8, output_norm="identity")
    x = torch.randn(1, 5, 16)
    attention_mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=x.dtype)
    valid_positions = torch.tensor([0, 1, 3, 4])

    masked, _, _ = layer(x, attention_mask=attention_mask)
    trimmed, _, _ = layer(x[:, valid_positions])

    torch.testing.assert_close(masked[:, valid_positions], trimmed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(masked[:, 2], torch.zeros_like(masked[:, 2]), rtol=0, atol=1e-6)
