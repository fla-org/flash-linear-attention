import pytest
import torch

from fla.ops.utils.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_split_cu_seqlens,
    prepare_token_indices,
)
from fla.utils import device


def ref_prepare_sequence_ids(cu_seqlens):
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    ids = []
    for seq_idx, length in enumerate(seqlens):
        ids.extend([seq_idx] * length)
    return torch.tensor(ids, device=cu_seqlens.device, dtype=cu_seqlens.dtype)


def ref_prepare_position_ids(cu_seqlens):
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in seqlens
    ])


def ref_prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=None, dtype=torch.int32):
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()

    return torch.tensor(
        [
            i
            for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:])
            for i in range(bos, eos, split_size)
        ] + [cu_seqlens[-1]],
        dtype=dtype,
        device=device,
    )


def ref_prepare_chunk_indices(cu_seqlens, chunk_size):
    lens = (cu_seqlens[1:] - cu_seqlens[:-1])
    n_chunks_per_seq = ((lens + chunk_size - 1) // chunk_size).tolist()

    indices = torch.cat([
        torch.arange(n, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        for n in n_chunks_per_seq
    ])
    seq_ids = indices.eq(0).cumsum(0) - 1
    return torch.stack([seq_ids, indices], 1)


def ref_prepare_chunk_offsets(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = (lens + chunk_size - 1) // chunk_size
    return torch.cat([
        cu_seqlens.new_tensor([0]),
        chunk_counts
    ]).cumsum(-1)

# ==========================================
# Test Cases
# ==========================================


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
@pytest.mark.parametrize("max_seq_len", [10, 128, 1024])
def test_prepare_ids_correctness(batch_size, max_seq_len):
    torch.manual_seed(42)

    seqlens = torch.randint(1, max_seq_len, (batch_size,), device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        seqlens.cumsum(0)
    ])

    # 1. Test Sequence IDs
    ref_seq = ref_prepare_sequence_ids(cu_seqlens)
    opt_seq = prepare_sequence_ids(cu_seqlens)
    torch.testing.assert_close(ref_seq, opt_seq, msg="Sequence IDs mismatch")

    # 2. Test Position IDs
    ref_pos = ref_prepare_position_ids(cu_seqlens)
    opt_pos = prepare_position_ids(cu_seqlens)
    torch.testing.assert_close(ref_pos, opt_pos, msg="Position IDs mismatch")

    # 3. Test Token Indices (Stack of the above)
    ref_stack = torch.stack([ref_seq, ref_pos], 1)
    opt_stack = prepare_token_indices(cu_seqlens)
    torch.testing.assert_close(ref_stack, opt_stack, msg="Token Indices mismatch")


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [100, 500])
@pytest.mark.parametrize("chunk_size", [16, 64, 128])
def test_chunk_utils_correctness(batch_size, max_seq_len, chunk_size):
    torch.manual_seed(42)

    seqlens = torch.randint(1, max_seq_len, (batch_size,), device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        seqlens.cumsum(0)
    ])

    ref_offsets = ref_prepare_chunk_offsets(cu_seqlens, chunk_size)
    opt_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)

    torch.testing.assert_close(
        ref_offsets.long(),
        opt_offsets.long(),
        msg="Chunk Offsets mismatch"
    )

    # 2. Test Chunk Indices
    ref_indices = ref_prepare_chunk_indices(cu_seqlens, chunk_size)
    opt_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    torch.testing.assert_close(
        ref_indices.long(),
        opt_indices.long(),
        msg="Chunk Indices mismatch"
    )


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("seq_len", [128, 1024])
@pytest.mark.parametrize("split_size", [32, 128, 129])
def test_split_cu_seqlens_correctness(batch_size, seq_len, split_size):
    torch.manual_seed(42)

    # Case A: cu_seqlens is None (Fixed length)
    ref_split_none = ref_prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=None)
    opt_split_none = prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=None, device=device)

    torch.testing.assert_close(
        ref_split_none,
        opt_split_none,
        msg="Split cu_seqlens (Fixed Len) mismatch"
    )

    # Case B: Variable length cu_seqlens
    real_lens = torch.randint(1, seq_len, (batch_size,), device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), real_lens.cumsum(0)])

    ref_split_var = ref_prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=cu_seqlens)
    opt_split_var = prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=cu_seqlens, device=device)

    torch.testing.assert_close(
        ref_split_var,
        opt_split_var,
        msg="Split cu_seqlens (Var Len) mismatch"
    )


def test_edge_cases():
    chunk_size = 32

    seqlens = torch.tensor([32, 64], device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), seqlens.cumsum(0)])

    ref = ref_prepare_chunk_indices(cu_seqlens, chunk_size)
    opt = prepare_chunk_indices(cu_seqlens, chunk_size)
    torch.testing.assert_close(ref.long(), opt.long())

    seqlens = torch.tensor([5, 10], device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), seqlens.cumsum(0)])

    ref = ref_prepare_chunk_indices(cu_seqlens, chunk_size)
    opt = prepare_chunk_indices(cu_seqlens, chunk_size)
    torch.testing.assert_close(ref.long(), opt.long())
