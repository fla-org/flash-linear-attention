# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""Compile-contract tests for varlen index helpers."""

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

import fla.utils as fu
from fla.ops.utils.index import (
    prepare_chunk_indices,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)

# Shared chunk size for the `prepare_chunk_indices` wrapper below.
_CHUNK_SIZE = 4


# Parametrize the compile-contract tests over all varlen index helpers.
HELPERS = [
    ("prepare_chunk_indices", lambda cu, cu_cpu=None: prepare_chunk_indices(cu, _CHUNK_SIZE, cu_seqlens_cpu=cu_cpu)),
    ("prepare_position_ids", prepare_position_ids),
    ("prepare_sequence_ids", prepare_sequence_ids),
    ("prepare_token_indices", prepare_token_indices),
]
_HELPER_IDS = [h[0] for h in HELPERS]


# Vary both segment count and per-segment lengths.
MULTI_SEGMENT_SHAPES = [
    [0, 5, 12, 20],
    [0, 10, 30, 45, 80, 100],
    [0, 7, 14],
    [0, 3, 8, 16, 24, 40],
    [0, 1, 2, 3, 4, 5, 6, 7],
]
# Single-segment (length-2) cu_seqlens, i.e. one packed sequence.
SINGLE_SEGMENT_SHAPES = [[0, 16], [0, 64], [0, 100], [0, 7], [0, 999]]


def _compile_and_count_graphs(impl, shapes):
    """Return the number of distinct graphs built for `shapes`."""
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
            expected = impl(cu)
            out = compiled(cu)
            assert torch.equal(out, expected), f"compiled output diverges on {s}"
        return counters["stats"].get("unique_graphs", 0)
    finally:
        fu.FLA_DISABLE_TENSOR_CACHE = original


@pytest.mark.parametrize("name,impl", HELPERS, ids=_HELPER_IDS)
def test_cu_seqlens_cpu_equivalent_to_device_branch(name, impl):
    cu = torch.tensor([0, 5, 12, 20, 33, 47], dtype=torch.int32)
    cu_cpu = cu.cpu().clone()
    assert torch.equal(impl(cu), impl(cu, cu_cpu)), f"{name} differs with cu_seqlens_cpu"


@pytest.mark.parametrize("name,impl", HELPERS, ids=_HELPER_IDS)
def test_no_recompile_across_multisegment_shapes(name, impl):
    n = _compile_and_count_graphs(impl, MULTI_SEGMENT_SHAPES)
    assert n == 1, (
        f"{name}: expected 1 compiled graph across {len(MULTI_SEGMENT_SHAPES)} shapes, got {n} -- Dynamo recompiled."
    )


@pytest.mark.parametrize("name,impl", HELPERS, ids=_HELPER_IDS)
def test_no_recompile_across_single_segment_shapes(name, impl):
    n = _compile_and_count_graphs(impl, SINGLE_SEGMENT_SHAPES)
    assert n == 1, f"{name}: expected 1 compiled graph across {len(SINGLE_SEGMENT_SHAPES)} single-segment shapes, got {n}."


@pytest.mark.parametrize("name,impl", HELPERS, ids=_HELPER_IDS)
def test_single_vs_multi_segment_specializes_once(name, impl):
    mixed = [[0, 64], [0, 5, 12, 20], [0, 32], [0, 7, 14, 21], [0, 100], [0, 3, 8]]
    n = _compile_and_count_graphs(impl, mixed)
    assert n == 2, f"{name}: expected exactly 2 compiled graphs (single- vs multi-segment 0/1 specialization), got {n}."
