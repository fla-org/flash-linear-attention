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
from benchmarks.ops.graph_benchmark import benchmark_case
from benchmarks.ops.graph_registry import GraphBenchmarkCase, _check_gdn_error, get_graph_op, normalize_gdn_shape


def test_gdn_registry_forward_backward():
    config = get_graph_op('chunk_gdn')
    assert config.supported_modes == ('fwd', 'fwdbwd')
    for shape in config.default_shapes.values():
        normalized = normalize_gdn_shape(shape)
        assert normalized['B'] == 1
        assert normalized['H'] == normalized['HV']
        assert normalized['D'] == normalized['DV']
        assert normalized['chunk_size'] == 64


@pytest.mark.parametrize(
    'override',
    [
        {'B': 2}, {'D': 32}, {'DV': 128}, {'HV': 4}, {'chunk_size': 32}, {'fused_options': True},
        {'disable_recompute': True}, {'dtype': 'float32'}, {'N': 0}, {'T': 0}, {'actual_T': 257},
    ],
    ids=[
        'override-B-2',
        'override-D-32',
        'override-DV-128',
        'override-HV-4',
        'override-chunk_size-32',
        'override-fused_options-True',
        'override-disable_recompute-True',
        'override-dtype-float32',
        'override-N-0',
        'override-T-0',
        'override-actual_T-257',
    ],
)
def test_gdn_registry_rejects_unsupported_shapes(override):
    with pytest.raises(ValueError):
        normalize_gdn_shape({'B': 1, 'T': 256, 'H': 2, 'D': 64, **override})


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
        measure_graph_safe=True,
    )
    kwargs = dict(
        case=case,
        mode='fwdbwd',
        engines=('graph',),
        warmup=0,
        iterations=2,
        graph_factory=lambda *_: lambda: replayed,
        synchronize=lambda: None,
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


def test_gdn_memory_measurement_covers_selected_engines(monkeypatch):
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
    case = SimpleNamespace(measure_graph_safe=True, metadata={})
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
    )
    assert calls == ['build', 'reset', 'baseline', 'benchmark', 'peak_allocated', 'peak_reserved', 'empty_cache']
    assert result[0]['memory'] == {'baseline_allocated_bytes': 100, 'peak_allocated_bytes': 250, 'peak_reserved_bytes': 400}


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
    )
    result = benchmark_case(case, 'fwdbwd', ('graph',), 0, 2, graph_factory=lambda step, _: step, synchronize=lambda: None)
    assert 'graph_safe' not in result
    assert calls == [True, True, False, False]
