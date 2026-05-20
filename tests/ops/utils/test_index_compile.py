# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""
Targeted tests for the torch.compile-friendly varlen index helpers in
`fla.ops.utils.index`:

  - prepare_chunk_indices
  - prepare_position_ids
  - prepare_sequence_ids
  - prepare_token_indices

All four historically built their per-segment `arange`s through a
`.tolist()` + Python list-comprehension over `cu_seqlens`. That is both a
host sync and a Dynamo graph break -- the latter caused a recompile-per-batch
loop whenever the segment count or per-segment lengths varied. The rewrite
routes them through the vectorised `_segmented_arange` helper
(`prepare_sequence_ids` / `prepare_token_indices` are unchanged but become
compile-clean transitively, since they consume the rewritten functions).

This file pins, for every helper:

  1. Equivalence to a reference (pre-rewrite) implementation on hand-built
     cu_seqlens patterns.
  2. `torch.compile(fullgraph=True, dynamic=True)` traces it without a graph
     break (`fullgraph=True` makes a break a hard error), and varying-shape
     inputs reuse a single compiled graph. We assert this directly by
     counting the distinct graphs Dynamo built (`counters['stats']
     ['unique_graphs']`): 1 == compiled once and reused, > 1 == a recompile.
  3. The one bounded exception: a single-segment `cu_seqlens` (length 2) and
     a multi-segment one compile to *separate* graphs -- see
     `test_single_vs_multi_segment_specializes_once` for why. That is a
     one-off specialization, not a per-batch recompile.

The general correctness sweep (random seqlens x chunk_sizes) already lives in
`tests/ops/test_index.py`; this file focuses on the compile contract and
explicit edge cases.
"""

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from fla.ops.utils.index import (
    prepare_chunk_indices,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)

# chunk_size used wherever `prepare_chunk_indices` needs one, so all four
# helpers can share a single-arg `(cu_seqlens) -> tensor` signature below.
_CHUNK_SIZE = 4


# --- reference (pre-rewrite) implementations, kept verbatim ----------------

def _ref_position_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in lens
    ])


def _ref_sequence_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return _ref_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


def _ref_token_indices(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [_ref_sequence_ids(cu_seqlens), _ref_position_ids(cu_seqlens)], 1
    ).to(cu_seqlens)


def _ref_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int = _CHUNK_SIZE) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_per_seq = ((lens + chunk_size - 1) // chunk_size).tolist()
    indices = torch.cat([
        torch.arange(n, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        for n in n_per_seq
    ])
    seq_ids = indices.eq(0).cumsum(0) - 1
    return torch.stack([seq_ids, indices], 1).to(cu_seqlens)


# (name, fla impl, reference impl) -- each impl/reference is `(cu) -> tensor`.
HELPERS = [
    ("prepare_chunk_indices", lambda cu: prepare_chunk_indices(cu, _CHUNK_SIZE), _ref_chunk_indices),
    ("prepare_position_ids", prepare_position_ids, _ref_position_ids),
    ("prepare_sequence_ids", prepare_sequence_ids, _ref_sequence_ids),
    ("prepare_token_indices", prepare_token_indices, _ref_token_indices),
]
_HELPER_IDS = [h[0] for h in HELPERS]


# Hand-built cu_seqlens patterns. Zero-length segments are deliberately
# excluded: they are not a valid cu_seqlens shape, and the old
# `.eq(0).cumsum`-based seg-id derivation silently mishandles them anyway
# (the rewrite happens to be more correct there -- out of contract).
CU_SEQLENS_CASES = [
    [0, 32],                              # single segment, exactly divisible
    [0, 100],                             # single segment, not divisible
    [0, 5],                               # single segment, shorter than chunk_size
    [0, 16, 32, 48, 64],                  # multiple equal-length segments
    [0, 5, 12, 20],                       # 3 segments of lengths 5, 7, 8
    [0, 7, 14, 21, 28],                   # equal odd-length segments
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # many tiny (length-1) segments
    [0, 1000],                            # one large segment
    [0, 3],                               # tiny single segment
    [0, 47, 53, 99, 100, 256, 512, 600],  # long ragged batch
]
_CASE_IDS = ["-".join(map(str, c)) for c in CU_SEQLENS_CASES]


# cu_seqlens with >= 2 segments (length >= 3). Both the segment count and the
# per-segment lengths vary across the list -- under the rewrite every one of
# these traces to the *same* compiled graph.
MULTI_SEGMENT_SHAPES = [
    [0, 5, 12, 20],
    [0, 10, 30, 45, 80, 100],
    [0, 7, 14],
    [0, 3, 8, 16, 24, 40],
    [0, 1, 2, 3, 4, 5, 6, 7],
]
# Single-segment (length-2) cu_seqlens, i.e. one packed sequence.
SINGLE_SEGMENT_SHAPES = [[0, 16], [0, 64], [0, 100], [0, 7], [0, 999]]


@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=_HELPER_IDS)
@pytest.mark.parametrize("cu_list", CU_SEQLENS_CASES, ids=_CASE_IDS)
def test_helper_matches_reference(name, impl, ref, cu_list):
    cu = torch.tensor(cu_list, dtype=torch.int32)
    # The helpers are `@tensor_cache`-decorated; the cache memoises by tensor
    # identity, so each fresh tensor here is a cache miss -- we exercise the
    # rewritten code path on every parametrised case.
    got = impl(cu)
    want = ref(cu)
    assert torch.equal(got, want), (
        f"{name} mismatch for cu_seqlens={cu_list}\n"
        f"got  = {got.tolist()}\nwant = {want.tolist()}"
    )


def test_cu_seqlens_cpu_equivalent_to_device_branch():
    """`cu_seqlens_cpu` is an avoid-sync optimisation; output must not change."""
    cu = torch.tensor([0, 5, 12, 20, 33, 47], dtype=torch.int32)
    cu_cpu = cu.clone()
    assert torch.equal(
        prepare_chunk_indices(cu, _CHUNK_SIZE),
        prepare_chunk_indices(cu, _CHUNK_SIZE, cu_seqlens_cpu=cu_cpu),
    )
    assert torch.equal(
        prepare_position_ids(cu),
        prepare_position_ids(cu, cu_seqlens_cpu=cu_cpu),
    )
    assert torch.equal(
        prepare_sequence_ids(cu),
        prepare_sequence_ids(cu, cu_seqlens_cpu=cu_cpu),
    )
    assert torch.equal(
        prepare_token_indices(cu),
        prepare_token_indices(cu, cu_seqlens_cpu=cu_cpu),
    )


def _compile_and_count_graphs(impl, ref, shapes):
    """Compile `impl` (``fullgraph=True, dynamic=True``), run it across every
    cu_seqlens in `shapes`, assert each compiled result matches `ref`, and
    return the number of distinct graphs Dynamo built.

      1  -> compiled once and reused for every shape (no recompile)
      >1 -> Dynamo recompiled

    `fullgraph=True` makes a graph break a hard error, so a successful return
    additionally proves the helper traces into a single break-free graph.
    """
    import fla.utils as fu

    # tensor_cache memoises by tensor identity; turn it off so each call
    # really re-enters the compiled body instead of returning the cached
    # tensor from the first call.
    original = fu.FLA_DISABLE_TENSOR_CACHE
    fu.FLA_DISABLE_TENSOR_CACHE = True
    try:
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(impl, fullgraph=True, dynamic=True)
        for s in shapes:
            cu = torch.tensor(s, dtype=torch.int32)
            out = compiled(cu)
            assert torch.equal(out, ref(cu)), f"compiled output diverges on {s}"
        return counters["stats"].get("unique_graphs", 0)
    finally:
        fu.FLA_DISABLE_TENSOR_CACHE = original


@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=_HELPER_IDS)
def test_no_recompile_across_multisegment_shapes(name, impl, ref):
    """Core compile contract: fullgraph trace + a *single* graph reused across
    five distinct multi-segment shapes (segment count AND per-segment lengths
    both vary).

    Under the old `.tolist()` code this was impossible -- it graph-broke
    (so `fullgraph=True` would raise) and otherwise recompiled per shape.
    """
    n = _compile_and_count_graphs(impl, ref, MULTI_SEGMENT_SHAPES)
    assert n == 1, (
        f"{name}: expected 1 compiled graph across "
        f"{len(MULTI_SEGMENT_SHAPES)} shapes, got {n} -- Dynamo recompiled."
    )


@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=_HELPER_IDS)
def test_no_recompile_across_single_segment_shapes(name, impl, ref):
    """Single-segment inputs are internally stable too: varying the lone
    segment's length reuses one graph (no recompile within this class)."""
    n = _compile_and_count_graphs(impl, ref, SINGLE_SEGMENT_SHAPES)
    assert n == 1, (
        f"{name}: expected 1 compiled graph across "
        f"{len(SINGLE_SEGMENT_SHAPES)} single-segment shapes, got {n}."
    )


@pytest.mark.parametrize("name,impl,ref", HELPERS, ids=_HELPER_IDS)
def test_single_vs_multi_segment_specializes_once(name, impl, ref):
    """Known, bounded limitation -- documented here so a future change that
    widens or removes it is a deliberate, visible diff.

    A single-segment cu_seqlens (length 2) and a multi-segment one compile to
    *separate* graphs: `counts = torch.diff(cu_seqlens)` has size 1 for a
    single-segment input, and PyTorch's dynamic-shape engine always
    specializes size-0/size-1 dims (the "0/1 specialization"). So a workload
    that mixes batch-size-1 and batch-size->1 varlen batches pays exactly one
    extra compile -- 2 graphs total, NOT a per-batch recompile loop.
    """
    mixed = [[0, 64], [0, 5, 12, 20], [0, 32], [0, 7, 14, 21], [0, 100], [0, 3, 8]]
    n = _compile_and_count_graphs(impl, ref, mixed)
    assert n == 2, (
        f"{name}: expected exactly 2 compiled graphs (single- vs "
        f"multi-segment 0/1 specialization), got {n}."
    )
