# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.msa import msa_indexer, naive_msa, naive_msa_indexer, parallel_msa
from fla.utils import assert_close, device


def reference_indexer(q_idx, k_idx, block_size, topk, local_blocks):
    """The indexer of the MiniMax-M3 reference, in its [B, H, T, ...] layout."""
    B, T, H, _ = q_idx.shape
    idx_q, idx_k = q_idx.transpose(1, 2).float(), k_idx[:, None].float()
    num_key_blocks = -(-T // block_size)
    pad = num_key_blocks * block_size - T
    scores = torch.matmul(idx_q, idx_k.transpose(-1, -2))
    position_ids = torch.arange(T, device=q_idx.device)[None].expand(B, -1)
    token_future = torch.arange(T, device=q_idx.device)[None, None, None, :] > position_ids[:, None, :, None]
    scores = scores.masked_fill(token_future, float("-inf"))
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    block_scores = scores.view(B, H, T, num_key_blocks, block_size).amax(dim=-1)
    q_block = position_ids // block_size
    if local_blocks > 0:
        local = torch.arange(local_blocks, device=q_idx.device)
        local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
        block_scores.scatter_(-1, local_idx.unsqueeze(1).expand(-1, H, -1, -1), float("inf"))
    topk_scores, topk_indices = block_scores.topk(min(topk, num_key_blocks), dim=-1)
    return topk_indices.masked_fill(topk_scores == float("-inf"), -1).transpose(1, 2)


def make_inputs(B, T, HQ, H, D, dtype):
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, D, dtype=dtype, device=device).requires_grad_()
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    q_idx = torch.randn(B, T, H, 32, device=device)
    k_idx = torch.randn(B, T, 32, device=device)
    return q, k, v, q_idx, k_idx


@pytest.mark.parametrize(('T', 'block_size', 'topk', 'local_blocks'), [(100, 16, 4, 1), (256, 32, 3, 2), (70, 64, 16, 1)])
def test_indexer(T, block_size, topk, local_blocks):
    _, _, _, q_idx, k_idx = make_inputs(2, T, 32, 2, 32, torch.float)
    ref = reference_indexer(q_idx, k_idx, block_size, topk, local_blocks)
    naive = naive_msa_indexer(q_idx, k_idx, block_size, topk, local_blocks)
    tri = msa_indexer(q_idx, k_idx, block_size, topk, local_blocks, chunk_size=48)
    S = ref.shape[-1]
    assert torch.equal(naive[..., :S], ref) and (naive[..., S:] == -1).all()
    assert torch.equal(tri, naive)


@pytest.mark.parametrize(
    ('B', 'T', 'HQ', 'H', 'D', 'block_size', 'topk', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-HQ{}-H{}-D{}-bs{}-topk{}-{}".format(*test))
        for test in [
            (2, 128, 16, 1, 32, 16, 4, torch.float),
            (1, 200, 32, 2, 64, 32, 3, torch.float),
            (2, 100, 16, 1, 32, 64, 4, torch.float),
            (2, 512, 64, 4, 64, 64, 4, torch.bfloat16),
        ]
    ],
)
def test_parallel(B, T, HQ, H, D, block_size, topk, dtype):
    q, k, v, q_idx, k_idx = make_inputs(B, T, HQ, H, D, dtype)
    indices = msa_indexer(q_idx, k_idx, block_size, topk)
    do = torch.randn_like(q)

    ref = naive_msa(q, k, v, indices, block_size)
    (ref * do).sum().backward()
    ref_grads = [q.grad, k.grad, v.grad]
    q.grad = k.grad = v.grad = None
    tri = parallel_msa(q, k, v, indices, block_size)
    (tri * do).sum().backward()

    tol = 1e-3 if dtype == torch.float else 0.008
    assert_close('o', ref, tri, tol)
    for name, r, t in zip(['dq', 'dk', 'dv'], ref_grads, [q.grad, k.grad, v.grad]):
        assert_close(name, r, t, tol)


def test_parallel_varlen():
    cu_seqlens = torch.tensor([0, 30, 130, 200], dtype=torch.long, device=device)
    q, k, v, q_idx, k_idx = make_inputs(1, 200, 16, 1, 32, torch.float)
    indices = msa_indexer(q_idx, k_idx, 16, 4, cu_seqlens=cu_seqlens)
    assert torch.equal(indices, naive_msa_indexer(q_idx, k_idx, 16, 4, cu_seqlens=cu_seqlens))
    do = torch.randn_like(q)
    ref = naive_msa(q, k, v, indices, 16, cu_seqlens=cu_seqlens)
    (ref * do).sum().backward()
    ref_grads = [q.grad, k.grad, v.grad]
    q.grad = k.grad = v.grad = None
    tri = parallel_msa(q, k, v, indices, 16, cu_seqlens=cu_seqlens)
    (tri * do).sum().backward()
    assert_close('o', ref, tri, 1e-3)
    for name, r, t in zip(['dq', 'dk', 'dv'], ref_grads, [q.grad, k.grad, v.grad]):
        assert_close(name, r, t, 1e-3)
