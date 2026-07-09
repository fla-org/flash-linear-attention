# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

import fla.utils as fu
from fla.ops.utils.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_split_cu_seqlens,
    prepare_token_indices,
)
from fla.utils import device

# Shared chunk size so all helpers expose a single-arg `(cu_seqlens) -> tensor`.
CHUNK_SIZE = 16


def ref_prepare_position_ids(cu_seqlens):
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in seqlens
    ])


def ref_prepare_sequence_ids(cu_seqlens):
    return ref_prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


def ref_prepare_token_indices(cu_seqlens):
    return torch.stack([ref_prepare_sequence_ids(cu_seqlens), ref_prepare_position_ids(cu_seqlens)], 1)


def ref_prepare_chunk_indices(cu_seqlens, chunk_size=CHUNK_SIZE):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks_per_seq = ((lens + chunk_size - 1) // chunk_size).tolist()
    indices = torch.cat([
        torch.arange(n, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        for n in n_chunks_per_seq
    ])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)


def ref_prepare_chunk_offsets(cu_seqlens, chunk_size=CHUNK_SIZE):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = (lens + chunk_size - 1) // chunk_size
    return torch.cat([cu_seqlens.new_tensor([0]), chunk_counts]).cumsum(-1)


def ref_prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=None, dtype=torch.int32):
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()
    return torch.tensor(
        [i for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]) for i in range(bos, eos, split_size)] + [cu_seqlens[-1]],
        dtype=dtype,
        device=device,
    )


def _prepare_chunk_indices(cu_seqlens, cu_seqlens_cpu=None):
    return prepare_chunk_indices(cu_seqlens, CHUNK_SIZE, cu_seqlens_cpu=cu_seqlens_cpu)


# (name, fla impl `(cu, cu_cpu=None) -> tensor`, reference impl `(cu) -> tensor`).
HELPERS = [
    ("chunk_indices", _prepare_chunk_indices, ref_prepare_chunk_indices),
    ("position_ids", prepare_position_ids, ref_prepare_position_ids),
    ("sequence_ids", prepare_sequence_ids, ref_prepare_sequence_ids),
    ("token_indices", prepare_token_indices, ref_prepare_token_indices),
]
HELPER_IDS = [h[0] for h in HELPERS]

skip_npu_compile = pytest.mark.skipif(
    fu.IS_NPU,
    reason='torch.compile graph-count contract is not supported on NPU yet',
)


def make_cu_seqlens(seqlens):
    seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)
    return torch.cat([torch.zeros(1, device=device, dtype=torch.int32), seqlens.cumsum(0)])


def count_unique_graphs(impl, seqlens_cases):
    """Compile `impl` and return how many distinct graphs Dynamo builds for the given shapes.

    `repeat_interleave` has a data-dependent output shape, so tracing it under
    `fullgraph=True` requires opting into dynamic output-shape ops. `tensor_cache`
    memoises by tensor identity, so it is disabled here to force every call back
    into the compiled body. Both globals are restored on exit.
    """
    cache_disabled = fu.FLA_DISABLE_TENSOR_CACHE
    capture = torch._dynamo.config.capture_dynamic_output_shape_ops
    fu.FLA_DISABLE_TENSOR_CACHE = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    try:
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(impl, fullgraph=True, dynamic=True)
        for seqlens in seqlens_cases:
            cu_seqlens = make_cu_seqlens(seqlens)
            torch.testing.assert_close(compiled(cu_seqlens).long(), impl(cu_seqlens).long())
        return counters["stats"].get("unique_graphs", 0)
    finally:
        fu.FLA_DISABLE_TENSOR_CACHE = cache_disabled
        torch._dynamo.config.capture_dynamic_output_shape_ops = capture


@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=HELPER_IDS)
@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
@pytest.mark.parametrize("max_seq_len", [10, 128, 1024])
def test_matches_reference(name, impl, ref, batch_size, max_seq_len):
    torch.manual_seed(42)
    seqlens = torch.randint(1, max_seq_len, (batch_size,), device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), seqlens.cumsum(0)])
    torch.testing.assert_close(impl(cu_seqlens).long(), ref(cu_seqlens).long(), msg=f"{name} mismatch")


@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=HELPER_IDS)
def test_cu_seqlens_cpu_matches_device(name, impl, ref):
    # `cu_seqlens_cpu` is an avoid-sync optimisation; the output must not change.
    cu_seqlens = make_cu_seqlens([5, 7, 8, 13, 14])
    cu_seqlens_cpu = cu_seqlens.cpu()
    torch.testing.assert_close(impl(cu_seqlens).long(), impl(cu_seqlens, cu_seqlens_cpu).long(), msg=f"{name} mismatch")


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [100, 500])
@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
def test_chunk_offsets_correctness(batch_size, max_seq_len, chunk_size):
    torch.manual_seed(42)
    cu_seqlens = make_cu_seqlens(torch.randint(1, max_seq_len, (batch_size,)).tolist())
    ref = ref_prepare_chunk_offsets(cu_seqlens, chunk_size)
    opt = prepare_chunk_offsets(cu_seqlens, chunk_size)
    torch.testing.assert_close(ref.long(), opt.long(), msg="Chunk offsets mismatch")


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("seq_len", [128, 1024])
@pytest.mark.parametrize("split_size", [32, 128, 129])
def test_split_cu_seqlens_correctness(batch_size, seq_len, split_size):
    torch.manual_seed(42)
    # Case A: cu_seqlens is None -> fixed-length sequences.
    ref = ref_prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=None)
    opt = prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=None, device=device)
    torch.testing.assert_close(ref, opt, msg="Split cu_seqlens (fixed len) mismatch")

    # Case B: variable-length cu_seqlens.
    cu_seqlens = make_cu_seqlens(torch.randint(1, seq_len, (batch_size,)).tolist())
    ref = ref_prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=cu_seqlens)
    opt = prepare_split_cu_seqlens(batch_size, seq_len, split_size, cu_seqlens=cu_seqlens, device=device)
    torch.testing.assert_close(ref, opt, msg="Split cu_seqlens (var len) mismatch")


def test_edge_cases():
    chunk_size = 32
    for seqlens in ([32, 64], [5, 10]):
        cu_seqlens = make_cu_seqlens(seqlens)
        ref = ref_prepare_chunk_indices(cu_seqlens, chunk_size)
        opt = prepare_chunk_indices(cu_seqlens, chunk_size)
        torch.testing.assert_close(ref.long(), opt.long())


@skip_npu_compile
@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=HELPER_IDS)
def test_no_recompile_across_multi_segment_shapes(name, impl, ref):
    shapes = [[5, 7, 8], [10, 20, 15, 35, 20], [7, 7], [3, 5, 8, 8, 16], [1, 1, 1, 1, 1, 1, 1]]
    n = count_unique_graphs(impl, shapes)
    assert n == 1, f"{name}: expected 1 graph across {len(shapes)} multi-segment shapes, got {n} -- Dynamo recompiled."


@skip_npu_compile
@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=HELPER_IDS)
def test_no_recompile_across_single_segment_shapes(name, impl, ref):
    shapes = [[16], [64], [100], [7], [999]]
    n = count_unique_graphs(impl, shapes)
    assert n == 1, f"{name}: expected 1 graph across {len(shapes)} single-segment shapes, got {n}."


@skip_npu_compile
@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=HELPER_IDS)
def test_single_vs_multi_segment_specializes_once(name, impl, ref):
    # A size-1 `counts` hits PyTorch's 0/1 dynamic-shape specialization, so single-
    # and multi-segment inputs each get one graph -- 2 total, not a per-batch recompile.
    shapes = [[64], [5, 7, 8], [32], [7, 7, 7], [100], [3, 5]]
    n = count_unique_graphs(impl, shapes)
    assert n == 2, f"{name}: expected 2 graphs (single- vs multi-segment specialization), got {n}."
