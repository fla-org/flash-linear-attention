# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils.graph import (
    host_chunk_statistics,
    normalize_graph_mode,
    route_graph_execution,
    static_chunk_capacity,
    validate_graph_capacity,
)
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_indices_static


@pytest.mark.parametrize(
    ("cu_seqlens", "expected"),
    [
        ([0, 0], (0, 0, 0)),
        ([0, 1], (1, 1, 1)),
        ([0, 63], (63, 1, 1)),
        ([0, 64], (64, 1, 1)),
        ([0, 65], (65, 1, 2)),
        ([0, 1, 64, 65], (65, 3, 3)),
        ([0, 64, 64, 129], (129, 2, 3)),
    ],
)
def test_host_chunk_statistics(cu_seqlens, expected):
    assert host_chunk_statistics(torch.tensor(cu_seqlens), 64) == expected


def test_static_metadata_matches_dynamic_for_empty_and_ragged_sequences():
    for offsets in ([0, 0, 0], [0, 1, 64, 65], [0, 63, 64, 129]):
        cu_seqlens = torch.tensor(offsets, dtype=torch.long)
        nt_max = static_chunk_capacity(129, len(offsets) - 1, 64)
        static_indices, static_offsets = prepare_chunk_indices_static(cu_seqlens, 64, nt_max)
        dynamic_indices = prepare_chunk_indices(cu_seqlens, 64)
        expected_offsets = torch.nn.functional.pad(
            torch.div(torch.diff(cu_seqlens) + 63, 64, rounding_mode="floor").cumsum(0), (1, 0)
        )
        assert static_indices.shape == (nt_max, 2)
        assert static_offsets.shape == (len(offsets),)
        torch.testing.assert_close(static_indices[:dynamic_indices.shape[0]], dynamic_indices)
        torch.testing.assert_close(static_offsets, expected_offsets)
        if dynamic_indices.shape[0] < nt_max:
            torch.testing.assert_close(
                static_indices[dynamic_indices.shape[0]:],
                torch.tensor([-1, 0], dtype=torch.long).expand(nt_max - dynamic_indices.shape[0], -1),
            )


@pytest.mark.parametrize("mode", [None, "eager", "force_graph", "auto"])
def test_normalize_graph_mode(mode):
    expected = "eager" if mode in (None, "eager") else mode
    assert normalize_graph_mode(mode) == expected
    if mode is None:
        assert normalize_graph_mode(mode, use_graph=True) == "force_graph"


def test_route_high_and_low_utilization():
    common = dict(
        actual_tokens=256,
        actual_sequences=4,
        t_max=256,
        n_max=4,
        nt_max=7,
        chunk_size=64,
    )
    high = route_graph_execution("auto", actual_nt=7, min_graph_utilization=0.75, **common)
    low = route_graph_execution("auto", actual_nt=1, min_graph_utilization=0.75, **common)
    assert high.selected_path == "graph"
    assert high.reason == "high_chunk_utilization"
    assert low.selected_path == "eager"
    assert low.reason == "low_chunk_utilization"


def test_route_missing_metadata_and_capacity_fallbacks():
    common = dict(t_max=256, n_max=4, nt_max=7, chunk_size=64)
    missing = route_graph_execution("auto", actual_tokens=None, actual_sequences=None, actual_nt=None, **common)
    assert missing.selected_path == "eager"
    assert missing.reason == "missing_host_metadata"

    over_tokens = route_graph_execution("auto", actual_tokens=257, actual_sequences=1, actual_nt=5, **common)
    over_sequences = route_graph_execution("auto", actual_tokens=128, actual_sequences=5, actual_nt=5, **common)
    assert over_tokens.reason == "tokens_exceed_t_max"
    assert over_sequences.reason == "sequences_exceed_n_max"

    with pytest.raises(ValueError, match="actual_tokens"):
        route_graph_execution("force_graph", actual_tokens=257, actual_sequences=1, actual_nt=5, **common)


def test_route_reports_known_overflow_before_missing_metadata():
    decision = route_graph_execution(
        "auto",
        actual_tokens=257,
        actual_sequences=None,
        actual_nt=None,
        t_max=256,
        n_max=4,
        nt_max=7,
        chunk_size=64,
    )
    assert decision.selected_path == "eager"
    assert decision.reason == "tokens_exceed_t_max"


def test_route_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="smaller than the required capacity"):
        route_graph_execution(
            "auto",
            actual_tokens=1,
            actual_sequences=1,
            actual_nt=1,
            t_max=256,
            n_max=4,
            nt_max=2,
            chunk_size=64,
        )


def test_static_metadata_rejects_underprovisioned_capacity():
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    with pytest.raises(ValueError, match="metadata requirement 2"):
        prepare_chunk_indices_static(cu_seqlens, 64, 1)

    with pytest.raises(ValueError, match="required capacity 3"):
        validate_graph_capacity(128, 2, 2, 64)


@pytest.mark.parametrize(
    ("input_tokens", "input_sequences", "reason"),
    [
        (128, 4, "input_token_shape_mismatch"),
        (256, 3, "input_sequence_shape_mismatch"),
    ],
)
def test_auto_falls_back_for_fixed_shape_mismatch(input_tokens, input_sequences, reason):
    decision = route_graph_execution(
        "auto",
        actual_tokens=256,
        actual_sequences=4,
        actual_nt=7,
        t_max=256,
        n_max=4,
        nt_max=7,
        min_graph_utilization=0.75,
        chunk_size=64,
        input_tokens=input_tokens,
        input_sequences=input_sequences,
    )
    assert decision.selected_path == "eager"
    assert decision.reason == reason


def test_force_graph_rejects_fixed_shape_mismatch():
    with pytest.raises(ValueError, match="input shape"):
        route_graph_execution(
            "force_graph",
            actual_tokens=128,
            actual_sequences=2,
            actual_nt=3,
            t_max=256,
            n_max=2,
            nt_max=5,
            chunk_size=64,
            input_tokens=128,
            input_sequences=2,
        )
