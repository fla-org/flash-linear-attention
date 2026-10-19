# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
from dataclasses import asdict

import pytest
import torch

from benchmarks.ops.kda import GRAPH, SCENARIOS, make_cu_seqlens, normalize_kda_shape
from benchmarks.utils.graph import GraphBenchmarkCase, benchmark_case
from benchmarks.utils.reporting import print_results


def test_kda_graph_registration_and_common_shape_defaults():
    assert "graph" in SCENARIOS
    config = GRAPH
    assert config.supported_modes == ("fwd", "fwdbwd")
    assert all({"B", "T", "H", "D"} <= shape.keys() for shape in config.default_shapes.values())


def test_normalize_kda_shape_adds_graph_axes_without_changing_common_axes():
    shape = normalize_kda_shape({"B": 1, "T": 128, "H": 2, "D": 64}, default_dtype="bfloat16")
    assert {name: shape[name] for name in ("B", "T", "H", "D")} == {
        "B": 1,
        "T": 128,
        "H": 2,
        "D": 64,
    }
    assert shape["N"] == 1
    assert shape["actual_T"] == 128
    assert shape["HV"] == 2
    assert shape["DV"] == 64
    assert shape["dtype"] == "bfloat16"


def test_normalize_kda_shape_allows_more_sequence_slots_than_tokens():
    shape = normalize_kda_shape({"B": 1, "T": 2, "H": 2, "D": 64, "N": 4})
    assert shape["T"] == 2
    assert shape["N"] == 4


def test_kda_sequence_profiles_are_valid_cumulative_offsets():
    for profile in ("balanced", "ragged", "empty_tail"):
        offsets = make_cu_seqlens(actual_t=17, num_seqs=3, profile=profile)
        assert len(offsets) == 4
        assert offsets[0] == 0
        assert offsets[-1] == 17
        assert offsets == sorted(offsets)


def test_kda_shape_rejects_dense_batch_and_invalid_fused_options():
    with pytest.raises(ValueError, match="B=1"):
        normalize_kda_shape({"B": 2, "T": 128, "H": 2, "D": 64})
    with pytest.raises(ValueError, match="require fused_options"):
        normalize_kda_shape({"B": 1, "T": 128, "H": 2, "D": 64, "safe_gate": True})


@pytest.mark.parametrize("mode", ["fwd", "fwdbwd"])
def test_benchmark_case_captures_once_validates_and_times_both_engines(mode):
    calls = []
    events = []

    def check_context():
        assert torch.is_grad_enabled() == (mode == "fwdbwd")
        assert not torch.is_inference_mode_enabled()

    def eager_step(*args):
        calls.append(("eager-step", args))
        events.append("eager")

    def graph_step(*args):
        calls.append(("graph-step", args))

    def run_once(step, args, collect):
        check_context()
        step(*args)
        return {"value": 1} if collect else {}

    comparisons = []
    case = GraphBenchmarkCase(
        eager_step=eager_step,
        graph_step=graph_step,
        capture_args=("capture",),
        eager_args=("eager",),
        graph_args=("live",),
        run_once=run_once,
        assert_close=lambda actual, reference: comparisons.append((actual, reference)),
    )

    captures = []

    def graph_factory(step, args):
        check_context()
        captures.append((step, args))
        events.append("capture")

        def replay(*live_args):
            calls.append(("graph-replay", live_args))

        return replay

    clock_values = iter(range(0, 15_000_000, 1_000_000))
    result = benchmark_case(
        case=case,
        mode=mode,
        engines=("eager", "graph"),
        warmup=1,
        iterations=2,
        graph_factory=graph_factory,
        synchronize=lambda: None,
        clock_ns=lambda: next(clock_values),
    )

    assert captures == [(graph_step, ("capture",))]
    assert comparisons == [({"value": 1}, {"value": 1})]
    assert result["validation"] == "passed"
    assert result["capture_ms"] == 1.0
    assert result["eager"]["p50_ms"] == 1.0
    assert result["graph"]["p95_ms"] == 1.0
    assert events.index("eager") < events.index("capture")
    assert any(kind == "graph-replay" and args == ("live",) for kind, args in calls)


def test_benchmark_case_rejects_reusing_capture_tensor_as_live_input():
    class FakeTensor:
        def data_ptr(self):
            return 1

    tensor = FakeTensor()
    case = GraphBenchmarkCase(
        eager_step=lambda *_: None,
        graph_step=lambda *_: None,
        capture_args=(tensor,),
        eager_args=(tensor,),
        graph_args=(tensor,),
        run_once=lambda *_: {},
        assert_close=lambda *_: None,
    )
    with pytest.raises(ValueError, match="reuses its capture address"):
        benchmark_case(
            case=case,
            mode="fwdbwd",
            engines=("graph",),
            warmup=0,
            iterations=1,
            graph_factory=lambda *_: None,
            synchronize=lambda: None,
        )


@pytest.mark.parametrize('extra_dimensions', [False, True])
def test_graph_workload_preserves_metrics_and_configuration(monkeypatch, capsys, extra_dimensions):
    from benchmarks.utils import graph

    row = {
        'op': 'chunk_kda', 'shape': 'custom', 'mode': 'fwdbwd',
        'B': 1, 'T': 128, 'H': 2, 'D': 64, 'N': 2, 'dtype': 'bfloat16', 'chunk_size': 32,
        'actual_T': 96 if extra_dimensions else 128,
        'HV': 4 if extra_dimensions else 2, 'DV': 32 if extra_dimensions else 64,
        'fused_options': True, 'sequence_profile': 'ragged', 'validation': 'passed',
        'eager': {'p50_ms': 2.0, 'p95_ms': 4.0},
        'graph': {'p50_ms': 1.0, 'p95_ms': 2.0}, 'capture_ms': 3.0,
    }
    monkeypatch.setattr(graph, 'benchmark_graph_op', lambda **kwargs: [row])
    parser = argparse.ArgumentParser()
    graph.add_graph_arguments(parser)
    results = graph.run_graph(args=parser.parse_args([]), config=GRAPH)
    assert len(results) == 2
    assert results[0].metrics == row['eager']
    assert results[1].metrics == {**row['graph'], 'capture_ms': 3.0}
    for result, engine in zip(results, ('eager', 'graph'), strict=True):
        assert result.name == 'chunk_kda'
        assert result.primary_metric == 'p50_ms'
        assert result.parameters == {
            **{key: value for key, value in row.items() if key not in ('op', 'eager', 'graph', 'capture_ms')},
            'engine': engine,
        }
    print_results(results=[asdict(result) for result in results])
    output = capsys.readouterr().out
    assert 'dtype=bfloat16' in output
    assert f"actual_T={row['actual_T']}" in output
    assert 'sequence_profile=ragged' in output
    assert 'capture_ms=3' in output
