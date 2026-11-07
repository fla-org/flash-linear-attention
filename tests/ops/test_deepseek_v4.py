# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.deepseek_v4 import naive_compress, naive_v4_attention, naive_v4_indexer
from fla.utils import assert_close, device

# Transcriptions of the DeepSeek-V4 modules of transformers (no cache, before `kv_norm` and rotary embeddings), in their
# [B, H, T, D] layout.


def reference_compress(kv, gate, position_bias, ratio, overlap):
    batch = kv.shape[0]
    usable = (kv.shape[1] // ratio) * ratio
    chunk_kv, chunk_gate = kv[:, :usable], gate[:, :usable]
    n_windows = chunk_kv.shape[1] // ratio
    if n_windows == 0:
        return kv.new_zeros((batch, 0, kv.shape[-1] // (2 if overlap else 1)))
    chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
    chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + position_bias
    if overlap:
        head_dim = chunk_kv.shape[-1] // 2
        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, head_dim))
        new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, head_dim), float("-inf"))
        new_kv[:, :, ratio:] = chunk_kv[..., head_dim:]
        new_gate[:, :, ratio:] = chunk_gate[..., head_dim:]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :head_dim]
        chunk_kv, chunk_gate = new_kv, new_gate
    return (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)


def reference_indexer(q, compressed_kv, weights, index_topk, compress_rate, softmax_scale):
    T = q.shape[1]
    position_ids = torch.arange(T, device=q.device)[None].expand(q.shape[0], -1)
    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores) * softmax_scale
    index_scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    compressed_len = compressed_kv.shape[1]
    top_k = min(index_topk, compressed_len)
    causal_threshold = (position_ids + 1) // compress_rate
    entry_indices = torch.arange(compressed_len, device=index_scores.device)
    future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
    index_scores = index_scores.masked_fill(future_mask, float("-inf"))
    top_k_indices = index_scores.topk(top_k, dim=-1).indices
    invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)


def reference_attention(q, kv, compressed_kv, sinks, window_size, compress_rate, top_k_indices, scaling):
    B, T, H, _ = q.shape
    q, kv = q.transpose(1, 2), kv[:, None]
    i = torch.arange(T, device=q.device)
    allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window_size)
    attention_mask = torch.zeros(B, 1, T, T, device=q.device).masked_fill(~allowed, float("-inf"))
    if compressed_kv is not None:
        compressed_kv = compressed_kv[:, None]
        compressed_len = compressed_kv.shape[2]
        if top_k_indices is None:
            position_ids = i[None].expand(B, -1)
            causal_threshold = (position_ids + 1) // compress_rate
            block_bias = compressed_kv.new_zeros((B, 1, T, compressed_len))
            block_bias = block_bias.masked_fill(
                torch.arange(compressed_len, device=q.device).view(1, 1, 1, -1)
                >= causal_threshold.unsqueeze(1).unsqueeze(-1),
                float("-inf"),
            )
        else:
            valid = top_k_indices >= 0
            safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
            block_bias = compressed_kv.new_full((B, 1, T, compressed_len + 1), float("-inf"))
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :compressed_len]
        kv = torch.cat([kv, compressed_kv], dim=2)
        attention_mask = torch.cat([attention_mask, block_bias], dim=-1)
    attn_weights = torch.matmul(q, kv.transpose(2, 3)) * scaling + attention_mask
    sinks = sinks.reshape(1, -1, 1, 1).expand(B, -1, T, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    return torch.matmul(probs[..., :-1], kv).transpose(1, 2)


@pytest.mark.parametrize(('B', 'T', 'D', 'ratio', 'overlap'), [
    (2, 37, 16, 4, True),
    (1, 64, 32, 4, True),
    (2, 300, 16, 128, False),
    (1, 3, 16, 4, True),
])
def test_compress(B, T, D, ratio, overlap):
    torch.manual_seed(42)
    C = 2 * D if overlap else D
    kv = torch.randn(B, T, C, device=device)
    gate = torch.randn(B, T, C, device=device)
    position_bias = torch.randn(ratio, C, device=device)
    ref = reference_compress(kv, gate, position_bias, ratio, overlap)
    tri = naive_compress(kv, gate, position_bias, ratio, overlap)
    assert tri.shape == ref.shape == (B, T // ratio, D)
    if T >= ratio:
        assert_close('compressed', ref, tri, 1e-5)


@pytest.mark.parametrize(('B', 'T', 'HI', 'DI', 'topk', 'ratio'), [
    (2, 64, 4, 16, 4, 4),
    (1, 37, 2, 32, 32, 4),
    (2, 100, 3, 16, 8, 4),
])
def test_indexer(B, T, HI, DI, topk, ratio):
    torch.manual_seed(42)
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, T // ratio, DI, device=device)
    w_idx = torch.randn(B, T, HI, device=device) * HI ** -0.5
    ref = reference_indexer(q_idx, k_idx, w_idx, topk, ratio, DI ** -0.5)
    tri = naive_v4_indexer(q_idx, k_idx, w_idx, topk, ratio)
    assert torch.equal(ref, tri)
    # every selected entry is visible to its query, and every query selects as many entries as it can
    visible = (torch.arange(T, device=device) + 1) // ratio
    assert (tri < visible[None, :, None]).all()
    assert torch.equal((tri >= 0).sum(-1), visible.clamp(max=tri.shape[-1])[None].expand(B, -1))


@pytest.mark.parametrize(('B', 'T', 'H', 'D', 'window_size', 'ratio', 'mode'), [
    (2, 64, 4, 32, 16, 4, 'csa'),
    (1, 100, 2, 16, 8, 4, 'csa'),
    (2, 300, 4, 16, 128, 128, 'hca'),
    (1, 64, 2, 16, 8, 16, 'hca'),
    (2, 50, 3, 16, 7, None, 'swa'),
])
def test_attention(B, T, H, D, window_size, ratio, mode):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, device=device)
    kv = torch.randn(B, T, D, device=device)
    sinks = torch.randn(H, device=device)
    compressed_kv, indices = None, None
    if mode != 'swa':
        compressed_kv = torch.randn(B, T // ratio, D, device=device)
    if mode == 'csa':
        q_idx = torch.randn(B, T, 2, 16, device=device)
        w_idx = torch.randn(B, T, 2, device=device)
        indices = reference_indexer(q_idx, torch.randn(B, T // ratio, 16, device=device), w_idx, 5, ratio, 0.25)
    ref = reference_attention(q, kv, compressed_kv, sinks, window_size, ratio, indices, D ** -0.5)
    tri = naive_v4_attention(q, kv, compressed_kv, sinks, window_size, ratio, indices)
    assert_close('o', ref, tri, 1e-5)


def test_attention_ignores_future_indices():
    torch.manual_seed(42)
    B, T, H, D, ratio = 1, 32, 2, 16, 4
    q = torch.randn(B, T, H, D, device=device)
    kv = torch.randn(B, T, D, device=device)
    compressed_kv = torch.randn(B, T // ratio, D, device=device)
    sinks = torch.randn(H, device=device)
    # selecting every entry, including those a query cannot see yet, is the same as attending to all visible ones
    indices = torch.arange(T // ratio, device=device).expand(B, T, -1)
    ref = naive_v4_attention(q, kv, compressed_kv, sinks, 8, ratio)
    tri = naive_v4_attention(q, kv, compressed_kv, sinks, 8, ratio, indices)
    assert_close('o', ref, tri, 1e-6)
