# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.dsa import dsa_indexer, naive_dsa_indexer
from fla.ops.dsa.indexer import dsa_indexer_score
from fla.utils import assert_close, device


def reference_index_scores(q_idx, k_idx, w_idx, scale, relu):
    """Causal index scores [B, T, T], computed as in `naive_dsa_indexer`."""
    score = torch.einsum('bmhd,bnd->bhmn', q_idx.float(), k_idx.float()) * scale
    if relu:
        score = score.relu()
    logits = score.sum(1) if w_idx is None else torch.einsum('bhmn,bmh->bmn', score, w_idx.float())
    i = torch.arange(q_idx.shape[1], device=q_idx.device)
    return logits.masked_fill(i[:, None] < i[None, :], float('-inf'))


def assert_same_selection(ref_scores, ref, tri):
    """Same number of valid slots per query, and the selected keys have the same scores up to ties."""
    assert tri.shape == ref.shape
    assert torch.equal((tri >= 0).sum(-1), (ref >= 0).sum(-1))
    assert ((tri >= 0) | (tri == -1)).all()
    valid = tri >= 0
    assert torch.equal(valid, ref >= 0)
    # no duplicates among the valid slots
    sorted_tri = tri.masked_fill(~valid, -1).sort(-1).values
    assert not ((sorted_tri[..., 1:] == sorted_tri[..., :-1]) & (sorted_tri[..., 1:] >= 0)).any()
    gather = lambda idx: ref_scores.gather(-1, idx.clamp(min=0)).masked_fill(~valid, 0).sort(-1).values  # noqa: E731
    assert_close('selected scores', gather(ref), gather(tri), 1e-4)


@pytest.mark.parametrize(
    ('B', 'T', 'HI', 'DI', 'topk', 'use_weights', 'relu', 'chunk_size', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-HI{}-DI{}-topk{}-w{}-relu{}-chunk{}-{}".format(*test))
        for test in [
            (2, 100, 4, 32, 16, True, True, None, torch.float),
            (1, 200, 8, 64, 32, True, True, 37, torch.float),
            (2, 64, 2, 48, 100, False, False, None, torch.float),
            (1, 130, 3, 16, 8, True, False, 64, torch.float),
            (2, 512, 16, 128, 64, True, True, 128, torch.bfloat16),
        ]
    ],
)
def test_indexer(B, T, HI, DI, topk, use_weights, relu, chunk_size, dtype):
    torch.manual_seed(42)
    q_idx = torch.randn(B, T, HI, DI, dtype=dtype, device=device)
    k_idx = torch.randn(B, T, DI, dtype=dtype, device=device)
    w_idx = torch.randn(B, T, HI, dtype=dtype, device=device) if use_weights else None
    scale = DI ** -0.5
    activation = 'relu' if relu else None

    ref_scores = reference_index_scores(q_idx, k_idx, w_idx, scale, relu)
    tri_scores = dsa_indexer_score(q_idx, k_idx, w_idx, scale, relu)
    assert torch.equal(torch.isinf(ref_scores), torch.isinf(tri_scores))
    finite = ~torch.isinf(ref_scores)
    assert_close('scores', ref_scores[finite], tri_scores[finite], 1e-4 if dtype == torch.float else 5e-3)

    ref = naive_dsa_indexer(q_idx, k_idx, w_idx, topk, activation=activation)
    tri = dsa_indexer(q_idx, k_idx, w_idx, topk, activation=activation, chunk_size=chunk_size)
    assert_same_selection(ref_scores, ref, tri)


def test_indexer_varlen():
    torch.manual_seed(42)
    cu_seqlens = torch.tensor([0, 15, 100, 101, 230], dtype=torch.long, device=device)
    T, HI, DI, topk = 230, 4, 32, 24
    q_idx = torch.randn(1, T, HI, DI, device=device)
    k_idx = torch.randn(1, T, DI, device=device)
    w_idx = torch.randn(1, T, HI, device=device)
    ref = naive_dsa_indexer(q_idx, k_idx, w_idx, topk, activation='relu', cu_seqlens=cu_seqlens)
    tri = dsa_indexer(q_idx, k_idx, w_idx, topk, activation='relu', cu_seqlens=cu_seqlens, chunk_size=50)
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        ref_scores = reference_index_scores(q_idx[:, bos:eos], k_idx[:, bos:eos], w_idx[:, bos:eos], DI ** -0.5, True)
        assert_same_selection(ref_scores, ref[:, bos:eos], tri[:, bos:eos])
