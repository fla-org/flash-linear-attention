# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from benchmarks.ops import graph_benchmark
from benchmarks.ops.graph_benchmark import benchmark_case
from benchmarks.ops.graph_registry import (
    GraphBenchmarkCase,
    GraphBenchmarkMetadata,
    _check_gdn_error,
    get_graph_op,
    normalize_gdn_shape,
)


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


def test_gdn_normalization_does_not_define_operator_capabilities():
    shape = normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 256, 'chunk_size': 128, 'HV': 4})
    assert shape['D'] == 256 and shape['chunk_size'] == 128 and shape['HV'] == 4


@pytest.mark.parametrize(
    'override',
    [{'T': True}, {'T': 1.5}, {'dtype': None}, {'fused_options': 1}, {'actual_T': 129}],
    ids=['bool-dimension', 'float-dimension', 'null-dtype', 'integer-option', 'active-over-capacity'],
)
def test_graph_metadata_rejects_invalid_field_types_and_values(override):
    metadata = GraphBenchmarkMetadata(**normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 64}))
    with pytest.raises(ValueError):
        replace(metadata, **override)


def test_graph_metadata_has_a_serializable_schema():
    import json

    metadata = GraphBenchmarkMetadata(**normalize_gdn_shape({'B': 1, 'T': 128, 'H': 2, 'D': 64}))
    assert json.loads(json.dumps(asdict(metadata))) == asdict(metadata)
    with pytest.raises(TypeError):
        GraphBenchmarkMetadata(**{**asdict(metadata), 'op': 'mislabelled'})


def test_gdn_registry_preserves_partial_capacity():
    shape = normalize_gdn_shape({'B': 1, 'T': 256, 'actual_T': 129, 'H': 2, 'D': 64, 'N': 4})
    assert shape['T'] == 256 and shape['actual_T'] == 129 and shape['N'] == 4


def test_gdn_registry_rejects_unknown_mode_before_loading_runtime():
    with pytest.raises(ValueError, match='Unsupported GDN graph benchmark mode'):
        get_graph_op('chunk_gdn').build_case({'B': 1, 'T': 65, 'H': 1, 'D': 64}, 'bwd', 'npu', 0)


@pytest.mark.parametrize('ci_env', ['0', '1'], ids=['ci_env-0', 'ci_env-1'])
@pytest.mark.parametrize(
    'abs_error,rms_error,passes',
    [
        (0.1, 0.02, False), (0.1, 0.015, False), (0.1, 0.014, True),
        (1e-6, 1.0, True), (float('nan'), 0.0, False), (0.0, float('inf'), False),
    ],
    ids=[
        'abs_error-0.1-rms_error-0.02-passes-false',
        'abs_error-0.1-rms_error-0.015-passes-false',
        'abs_error-0.1-rms_error-0.014-passes-true',
        'abs_error-1e-06-rms_error-1.0-passes-true',
        'abs_error-float-nan-rms_error-0.0-passes-false',
        'abs_error-0.0-rms_error-float-inf-passes-false',
    ],
)
def test_gdn_error_budget_is_strict(monkeypatch, ci_env, abs_error, rms_error, passes):
    monkeypatch.setenv('FLA_CI_ENV', ci_env)
    if passes:
        _check_gdn_error('dg', abs_error, rms_error, 0.015)
    else:
        with pytest.raises(AssertionError):
            _check_gdn_error('dg', abs_error, rms_error, 0.015)


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
        metadata=GraphBenchmarkMetadata(**normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64})),
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
    metadata = GraphBenchmarkMetadata(**normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64}))
    case = SimpleNamespace(metadata=metadata)
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
        metadata=GraphBenchmarkMetadata(**normalize_gdn_shape({'B': 1, 'T': 64, 'H': 1, 'D': 64})),
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


def test_benchmark_case_requires_typed_metadata():
    with pytest.raises(TypeError, match='GraphBenchmarkMetadata'):
        GraphBenchmarkCase(
            eager_step=lambda: None,
            graph_step=lambda: None,
            capture_args=(),
            eager_args=(),
            graph_args=(),
            run_once=lambda *_: {},
            assert_close=lambda *_: None,
            metadata={},
        )
