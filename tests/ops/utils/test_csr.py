# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.csr import prepare_block_csr
from fla.utils import device


def ref_block_csr(block_indices, block_counts, cu_seqlens, num_blocks, block_size):
    # reference: same validity, but CSR built with unique() (sorted within each block)
    B, T, H, S = block_indices.shape
    BS = block_size
    dev = block_indices.device
    bi = block_indices
    b = torch.arange(B, device=dev).view(B, 1, 1, 1)
    t = torch.arange(T, device=dev).view(1, T, 1, 1)
    h = torch.arange(H, device=dev).view(1, 1, H, 1)
    slot = torch.arange(S, device=dev).view(1, 1, 1, S)
    quota = block_counts.unsqueeze(-1) if torch.is_tensor(block_counts) else block_counts
    valid = (bi >= 0) & (bi < num_blocks) & (bi * BS <= t) & (slot < quota)
    b, t, h, bi = (x.expand_as(block_indices)[valid] for x in (b, t, h, bi))
    if cu_seqlens is None:
        block_id = (b * H + h) * num_blocks + bi
        q_pos = b * T + t
        n_blocks = B * H * num_blocks
    else:
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        chunk_offsets = torch.cat([cu_seqlens.new_zeros(1), ((lens + BS - 1) // BS).cumsum(0)])
        seq = torch.searchsorted(cu_seqlens[1:].contiguous(), t, right=True)
        block_id = (chunk_offsets[seq] + bi) * H + h
        q_pos = t
        n_blocks = int(chunk_offsets[-1]) * H
    key = torch.unique(block_id * (B * T) + q_pos)
    csr_indices = (key % (B * T)).to(torch.int32)
    csr_offsets = torch.zeros(n_blocks + 1, dtype=torch.int32, device=dev)
    csr_offsets[1:] = torch.bincount(key // (B * T), minlength=n_blocks).cumsum(0)
    return csr_indices, csr_offsets


def gen_block_indices(B, T, H, S, num_blocks, block_size):
    # distinct, causal block ids per query (mirrors top-k selection); -1 pads short tails
    g = torch.Generator().manual_seed(42)
    bi = torch.full((B, T, H, S), -1, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            hi = min(num_blocks, t // block_size + 1)
            for h in range(H):
                k = min(S, hi)
                if k > 0:
                    bi[b, t, h, :k] = torch.randperm(hi, generator=g)[:k]
    return bi.to(device)


def assert_csr_match(out, ref):
    (qi, qo), (rqi, rqo) = out, ref
    # offsets (per-block counts) must match exactly
    torch.testing.assert_close(qo.long(), rqo.long())
    qi, qo, rqi, rqo = qi.cpu(), qo.cpu(), rqi.cpu(), rqo.cpu()
    # within a block the scatter order is arbitrary, so compare the query sets
    for i in range(qo.numel() - 1):
        got = set(qi[qo[i]:qo[i + 1]].tolist())
        exp = set(rqi[rqo[i]:rqo[i + 1]].tolist())
        assert got == exp, f"block {i}: {got} != {exp}"


def make_block_counts(B, T, H, S, kind):
    if kind == "int":
        return S
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, S + 1, (B, T, H), generator=g).to(device)


@pytest.mark.parametrize("counts_kind", ["int", "tensor"])
@pytest.mark.parametrize(
    ("B", "T", "H", "S", "block_size"),
    [(1, 64, 1, 8, 16), (2, 100, 2, 4, 16), (1, 60, 4, 8, 32)],
)
def test_block_csr(B, T, H, S, block_size, counts_kind):
    num_blocks = (T + block_size - 1) // block_size
    block_indices = gen_block_indices(B, T, H, S, num_blocks, block_size)
    block_counts = make_block_counts(B, T, H, S, counts_kind)
    out = prepare_block_csr(block_indices, block_counts, None, None, num_blocks, block_size)
    ref = ref_block_csr(block_indices, block_counts, None, num_blocks, block_size)
    assert_csr_match(out, ref)


@pytest.mark.parametrize("counts_kind", ["int", "tensor"])
@pytest.mark.parametrize("seqlens", [[15, 49], [30, 1, 33, 64], [100]])
@pytest.mark.parametrize(("H", "S", "block_size"), [(1, 8, 16), (4, 8, 16)])
def test_block_csr_varlen(seqlens, H, S, block_size, counts_kind):
    cu_seqlens = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    T = int(cu_seqlens[-1])
    num_blocks = (max(seqlens) + block_size - 1) // block_size
    block_indices = gen_block_indices(1, T, H, S, num_blocks, block_size)
    block_counts = make_block_counts(1, T, H, S, counts_kind)
    chunk_indices = prepare_chunk_indices(cu_seqlens, block_size)
    out = prepare_block_csr(block_indices, block_counts, cu_seqlens, chunk_indices, num_blocks, block_size)
    ref = ref_block_csr(block_indices, block_counts, cu_seqlens, num_blocks, block_size)
    assert_csr_match(out, ref)
