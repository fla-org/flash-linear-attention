# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import sys
from types import SimpleNamespace

import pytest

from benchmarks.ops import graph_benchmark
from benchmarks.ops.graph_benchmark import benchmark_case, print_results
from benchmarks.ops.graph_registry import (
    GraphBenchmarkCase,
    _check_gdn_error,
    get_graph_op,
    list_graph_ops,
    make_cu_seqlens,
    normalize_gdn_shape,
    normalize_kda_shape,
    validate_graph_metadata,
)


def test_kda_graph_registration_and_common_shape_defaults():
    assert "chunk_kda" in list_graph_ops()
    config = get_graph_op("chunk_kda")
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


def test_gdn_registry_forward_backward():
    config = get_graph_op('chunk_gdn')
    assert config.supported_modes == ('fwd', 'fwdbwd')
    for shape in config.default_shapes.values():
        normalized = normalize_gdn_shape(shape)
        assert normalized['B'] == 1
        assert normalized['H'] == normalized['HV']
        assert normalized['D'] == normalized['DV']
        assert normalized['chunk_size'] == 64
    axes = ('B', 'T', 'H', 'D', 'N')
    gdn_shapes = {tuple(shape[key] for key in axes) for shape in config.default_shapes.values()}
    kda_shapes = {tuple(shape[key] for key in axes) for shape in get_graph_op('chunk_kda').default_shapes.values()}
    assert gdn_shapes == kda_shapes
    assert {shape['dtype'] for shape in config.default_shapes.values()} == {'float16', 'bfloat16'}


def test_gdn_normalization_does_not_define_operator_capabilities():
    shape = normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 256, 'chunk_size': 128, 'HV': 4})
    assert shape['D'] == 256 and shape['chunk_size'] == 128 and shape['HV'] == 4


@pytest.mark.parametrize(
    'override',
    [{'B': True}, {'T': 128.75}, {'D': 64.9}, {'N': 2.9}, {'actual_T': 127.5}],
    ids=['bool-batch', 'fractional-tokens', 'fractional-dim', 'fractional-sequences', 'fractional-active-tokens'],
)
def test_gdn_normalization_rejects_noninteger_dimensions(override):
    with pytest.raises(ValueError, match='must be a positive integer'):
        normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 64, **override})


@pytest.mark.parametrize(
    'extra',
    [{'workload_label': 'custom'}, {'T': '128', 'workload_label': 'custom'}, {'notes': {'run': 1}}],
    ids=['label', 'numeric-string', 'nested-annotation'],
)
def test_kda_normalization_preserves_annotations(extra):
    requested = {'B': 1, 'T': 128, 'H': 2, 'D': 64, **extra}
    normalized = normalize_kda_shape(requested)
    assert normalized['T'] == 128
    for name in set(extra) - {'T'}:
        assert normalized[name] == extra[name]
    with pytest.raises(ValueError, match='Unknown graph benchmark fields'):
        normalize_gdn_shape(requested)


def test_gdn_registry_preserves_partial_capacity():
    shape = normalize_gdn_shape({'B': 1, 'T': 256, 'actual_T': 129, 'H': 2, 'D': 64, 'N': 4})
    assert shape['T'] == 256 and shape['actual_T'] == 129 and shape['N'] == 4


def test_gdn_registry_rejects_unknown_mode_before_loading_runtime():
    with pytest.raises(ValueError, match='Unsupported GDN graph benchmark mode'):
        get_graph_op('chunk_gdn').build_case({'B': 1, 'T': 65, 'H': 1, 'D': 64}, 'bwd', 'npu', 0)


@pytest.mark.parametrize(
    'ci_env,abs_error,rms_error,passes',
    [('0', 0.1, 0.02, False), ('1', 0.1, 0.015, False), ('0', 0.1, 0.014, True), ('1', 1e-6, 1.0, True)],
    ids=['local-above-budget', 'ci-at-budget', 'local-below-budget', 'ci-absolute-shortcut'],
)
def test_gdn_error_budget_is_strict(monkeypatch, ci_env, abs_error, rms_error, passes):
    monkeypatch.setenv('FLA_CI_ENV', ci_env)
    if passes:
        _check_gdn_error('dg', abs_error, rms_error, 0.015)
    else:
        with pytest.raises(AssertionError):
            _check_gdn_error('dg', abs_error, rms_error, 0.015)


@pytest.mark.parametrize(
    'ci_env,abs_error,rms_error',
    [('0', float('nan'), 0.0), ('1', 0.0, float('inf'))],
    ids=['local-nan', 'ci-infinity'],
)
def test_gdn_error_budget_rejects_nonfinite_values(monkeypatch, ci_env, abs_error, rms_error):
    monkeypatch.setenv('FLA_CI_ENV', ci_env)
    with pytest.raises(AssertionError, match='Non-finite'):
        _check_gdn_error('dg', abs_error, rms_error, 0.015)


@pytest.mark.parametrize(
    'override',
    [
        {'N': 0}, {'T': 0}, {'actual_T': 257}, {'sequence_profile': 'unknown'}, {'mode': 'fwd'},
    ],
    ids=[
        'override-N-0',
        'override-T-0',
        'override-actual_T-257',
        'override-sequence-profile-unknown',
        'override-reserved-mode',
    ],
)
def test_gdn_registry_rejects_invalid_metadata(override):
    with pytest.raises(ValueError):
        normalize_gdn_shape({'B': 1, 'T': 256, 'H': 2, 'D': 64, **override})


@pytest.mark.parametrize(
    'override',
    [{'T': True}, {'T': 1.5}, {'dtype': None}, {'fused_options': 1}, {'actual_T': 129}],
    ids=['bool-dimension', 'float-dimension', 'null-dtype', 'integer-option', 'active-over-capacity'],
)
def test_graph_metadata_rejects_invalid_field_types_and_values(override):
    metadata = normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 64})
    with pytest.raises(ValueError):
        validate_graph_metadata({**metadata, **override})


def test_graph_metadata_has_a_serializable_schema():
    import json

    metadata = normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 64})
    assert json.loads(json.dumps(metadata)) == metadata
    with pytest.raises(ValueError, match='reserved'):
        validate_graph_metadata({**metadata, 'op': 'mislabelled'})


@pytest.mark.parametrize(
    'override,error,match',
    [
        ({'metadata': []}, TypeError, 'dictionary'),
        ({'metadata': {'op': 'mislabelled'}}, ValueError, 'reserved'),
        ({'metadata': {'shape': 'mislabelled'}}, ValueError, 'reserved'),
        ({'metadata': {'graph': {}}}, ValueError, 'reserved'),
    ],
    ids=['untyped-metadata', 'reserved-identity', 'reserved-dimension', 'reserved-timing'],
)
def test_benchmark_case_rejects_invalid_metadata(override, error, match):
    options = {'metadata': normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64}),
               **override}
    with pytest.raises(error, match=match):
        GraphBenchmarkCase(
            eager_step=lambda: None,
            graph_step=lambda: None,
            capture_args=(),
            eager_args=(),
            graph_args=(),
            run_once=lambda *_: {},
            assert_close=lambda *_: None,
            **options,
        )


@pytest.mark.parametrize("mode", ["fwd", "fwdbwd"])
def test_benchmark_case_captures_once_validates_and_times_both_engines(mode):
    torch = pytest.importorskip("torch")

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
        metadata={'B': 1, 'T': 128, 'H': 2, 'D': 64},
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


@pytest.mark.parametrize("graph_index", [0, 2])
def test_run_routes_graph_arguments(monkeypatch, graph_index):
    from benchmarks.ops import graph_benchmark, run

    arguments = ["--op", "chunk_kda", "--no-base", "--modes", "fwd", "fwdbwd", "--iterations", "2"]
    forwarded = []
    monkeypatch.setattr(graph_benchmark, "run_graph_benchmark", lambda argv: forwarded.append(argv))
    argv = arguments.copy()
    argv.insert(graph_index, "--graph")
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run.main()
    assert forwarded == [arguments]


@pytest.mark.parametrize("graph", [False, True])
def test_run_help_and_list(monkeypatch, capsys, graph):
    from benchmarks.ops import run

    prefix = ["run.py", "--graph"] if graph else ["run.py"]
    monkeypatch.setattr(sys, "argv", [*prefix, "--help"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert ("--iterations" if graph else "--graph") in help_text

    monkeypatch.setattr(sys, "argv", [*prefix, "--list"])
    run.main()
    assert "chunk_kda" in capsys.readouterr().out


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
        metadata={'B': 1, 'T': 128, 'H': 2, 'D': 64},
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


@pytest.mark.parametrize('requires_grad', [False, True], ids=['inference', 'training'])
def test_graph_safe_measurement_uses_independent_leaf_storage(requires_grad):
    torch = pytest.importorskip("torch")

    live = torch.ones(2, requires_grad=requires_grad)
    seen = []

    def graph_step(value):
        seen.append(value)
        return value

    def run_once(step, args, collect):
        value = step(*args)
        return {'o': value.detach().clone()} if collect else {}

    case = GraphBenchmarkCase(
        eager_step=lambda value: value,
        graph_step=graph_step,
        capture_args=(torch.zeros_like(live),),
        eager_args=(live.detach().clone(),),
        graph_args=(live,),
        run_once=run_once,
        assert_close=lambda actual, expected: torch.testing.assert_close(actual['o'], expected['o']),
        metadata={'B': 1, 'T': 128, 'H': 2, 'D': 64},
    )
    benchmark_case(
        case=case,
        mode='fwdbwd',
        engines=('graph',),
        warmup=0,
        iterations=2,
        graph_factory=lambda *_: lambda value: value,
        synchronize=lambda: None,
        measure_graph_safe=True,
    )
    assert len(seen) == 3
    assert all(value.data_ptr() != live.data_ptr() for value in seen)
    assert all(value.is_leaf and value.requires_grad == requires_grad for value in seen)
    assert all(value is seen[0] for value in seen)


@pytest.mark.parametrize(
    'replayed,uncaptured',
    [(5.1, 5.0), (5.0, 5.1), (5.01, 5.01)],
    ids=[
        'replayed-5.1-uncaptured-5.0',
        'replayed-5.0-uncaptured-5.1',
        'replayed-5.01-uncaptured-5.01',
    ],
)
def test_gdn_runner_checks_both_graph_paths_before_reporting(replayed, uncaptured):
    timings = []

    def run_once(step, args, collect):
        value = step()
        if not collect:
            timings.append(value)
        return {'dg': value} if collect else {}

    def compare(actual, reference):
        error = abs(actual['dg'] - reference['dg'])
        _check_gdn_error('dg', error, error / abs(reference['dg']), 0.015)

    case = GraphBenchmarkCase(
        eager_step=lambda: 5.0,
        graph_step=lambda: uncaptured,
        capture_args=(),
        graph_args=(),
        eager_args=(),
        run_once=run_once,
        assert_close=compare,
        metadata=normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64}),
    )
    kwargs = dict(
        case=case,
        mode='fwdbwd',
        engines=('graph',),
        warmup=0,
        iterations=2,
        graph_factory=lambda *_: lambda: replayed,
        synchronize=lambda: None,
        measure_graph_safe=True,
    )
    if 5.1 in (replayed, uncaptured):
        with pytest.raises(AssertionError, match='GDN dg'):
            benchmark_case(**kwargs)
        assert timings == []
    else:
        result = benchmark_case(**kwargs)
        assert result['validation'] == 'passed'
        assert result['graph_safe']['p50_ms'] >= 0
        assert result['graph']['p50_ms'] >= 0
        assert len(timings) == 4


@pytest.mark.parametrize('report_memory', [False, True], ids=['memory-off', 'memory-on'])
def test_gdn_memory_measurement_covers_selected_engines(monkeypatch, report_memory):
    calls = []

    def record(name, result=None):
        def call(*args, **kwargs):
            calls.append(name)
            return result
        return call

    npu = SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda _: None,
        reset_peak_memory_stats=record('reset'),
        memory_allocated=record('baseline', 100),
        max_memory_allocated=record('peak_allocated', 250),
        max_memory_reserved=record('peak_reserved', 400),
        synchronize=lambda: None,
        empty_cache=record('empty_cache'),
    )
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(npu=npu))
    metadata = normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64})
    case = SimpleNamespace(metadata={**metadata, 'workload_label': 'custom'})
    monkeypatch.setattr(graph_benchmark, 'get_graph_op', lambda _: SimpleNamespace(
        supported_modes=('fwdbwd',),
        build_case=record('build', case),
    ))
    monkeypatch.setattr(graph_benchmark, 'benchmark_case', record('benchmark', {'validation': 'not-run'}))
    result = graph_benchmark.benchmark_graph_op(
        'chunk_gdn',
        {'smoke': {'B': 1, 'T': 64, 'H': 1, 'D': 64}},
        ('fwdbwd',),
        ('eager',),
        'float16',
        'npu:0',
        0,
        1,
        progress=False,
        report_memory=report_memory,
    )
    if report_memory:
        assert calls == ['build', 'reset', 'baseline', 'benchmark', 'peak_allocated', 'peak_reserved', 'empty_cache']
        assert result[0]['memory'] == {'baseline_allocated_bytes': 100,
                                       'peak_allocated_bytes': 250, 'peak_reserved_bytes': 400}
    else:
        assert calls == ['build', 'benchmark', 'empty_cache']
        assert 'memory' not in result[0]
    assert result[0]['mode'] == 'fwdbwd'
    assert result[0]['T'] == 64
    assert result[0]['workload_label'] == 'custom'


def test_graph_safe_measurement_is_opt_in():
    calls = []

    def run_once(step, args, collect):
        calls.append(collect)
        return step()

    case = GraphBenchmarkCase(
        eager_step=lambda: 1,
        graph_step=lambda: 1,
        capture_args=(),
        graph_args=(),
        eager_args=(),
        run_once=run_once,
        assert_close=lambda actual, expected: None,
        metadata=normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64}),
    )
    result = benchmark_case(case, 'fwdbwd', ('graph',), 0, 2, graph_factory=lambda step, _: step, synchronize=lambda: None)
    assert 'graph_safe' not in result
    assert calls == [True, True, False, False]


@pytest.mark.parametrize('enabled', [False, True], ids=['diagnostics-off', 'diagnostics-on'])
def test_runner_forwards_diagnostics_to_the_worker(enabled):
    flags = ['--measure-graph-safe', '--report-memory'] if enabled else []
    _, args = graph_benchmark._parse_args(['--op', 'chunk_gdn', *flags])
    for engines in (('eager',), ('eager', 'graph')):
        command = graph_benchmark._worker_command('run.py', ['chunk_gdn'], {}, ('fwd',), engines, args, 'result.json')
        assert ('--measure-graph-safe' in command) == (enabled and 'graph' in engines)
        assert ('--report-memory' in command) == enabled


def test_result_table_reports_base_and_graph_speedups(capsys):
    current = [{
        "op": "chunk_kda",
        "shape": "small",
        "mode": "fwdbwd",
        "B": 1,
        "T": 128,
        "H": 2,
        "D": 64,
        "N": 2,
        "eager": {"p50_ms": 2.0, "p95_ms": 4.0},
        "graph": {"p50_ms": 1.0, "p95_ms": 2.0},
    }]
    baseline = [{
        **current[0],
        "eager": {"p50_ms": 3.0, "p95_ms": 6.0},
        "graph": None,
    }]
    info = {"git_label": "ascend_graph[abc]", "device": "Ascend", "torch": "2.9", "torch_npu": "2.9"}
    base_info = {"git_label": "main[def]"}

    print_results(current, info, baseline, base_info)
    output = capsys.readouterr().out
    assert "main[def] eager / ascend_graph[abc] eager / ascend_graph[abc] graph" in output
    header = next(line for line in output.splitlines() if line.startswith("mode"))
    assert header.split()[:3] == ["mode", "case", "stat"]
    assert "graph x    config" in header
    assert header.endswith("config")
    assert output.count("1.50x") == 2
    assert output.count("2.00x") == 2


@pytest.mark.parametrize("extra_dimensions", [False, True])
def test_result_table_reports_graph_specific_configuration(capsys, extra_dimensions):
    current = [{
        "op": "chunk_kda",
        "shape": "custom",
        "mode": "fwdbwd",
        "B": 1,
        "T": 128,
        "H": 2,
        "D": 64,
        "N": 2,
        "dtype": "bfloat16",
        "chunk_size": 32,
        "actual_T": 96 if extra_dimensions else 128,
        "HV": 4 if extra_dimensions else 2,
        "DV": 32 if extra_dimensions else 64,
        "fused_options": True,
        "sequence_profile": "ragged",
        "eager": {"p50_ms": 2.0, "p95_ms": 2.0},
        "graph": {"p50_ms": 1.0, "p95_ms": 1.0},
    }]
    info = {"git_label": "HEAD", "device": "Ascend", "torch": "2.9", "torch_npu": "2.9"}

    print_results(current, info)
    output = capsys.readouterr().out
    assert "dtype=bfloat16" in output
    assert "chunk_size=32" in output
    if extra_dimensions:
        assert "actual_T=96" in output
        assert "HV=4" in output
        assert "DV=32" in output
    else:
        assert "actual_T=" not in output
        assert "HV=" not in output
        assert "DV=" not in output
    assert "fused_options=True" in output
    assert "sequence_profile=ragged" in output
