# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import sys

import pytest
import torch
import triton

from benchmarks.ops import run
from fla import utils


@pytest.mark.parametrize('custom_shapes', [False, True])
@pytest.mark.parametrize('compile', [False, True])
def test_benchmark_shapes_and_compile(monkeypatch, custom_shapes, compile):
    shape = {'B': 1, 'T': 3, 'H': 1, 'D': 7, 'N': 2}
    defaults = {'default': {**shape, 'T': 5, 'N': 4}}
    monkeypatch.setattr(run.get_op('naive_hc_pre'), 'default_shapes', defaults)
    monkeypatch.setattr(utils, 'device_name', 'cpu')
    monkeypatch.setattr(run, '_device_synchronize', lambda device: None)
    events = []

    def compile_fn(fn, fullgraph):
        assert fullgraph
        events.append('compile')
        return fn

    def do_bench(fn, **kwargs):
        events.append('time')
        fn()
        return [1.0, 0.8, 1.2]

    monkeypatch.setattr(torch, 'compile', compile_fn)
    monkeypatch.setattr(triton.testing, 'do_bench', do_bench)
    results = run.benchmark_op(
        op_name='naive_hc_pre',
        shapes={'custom': shape} if custom_shapes else None,
        compile=compile,
    )
    expected = shape if custom_shapes else defaults['default']
    assert len(results) == 2
    assert [row['mode'] for row in results] == ['fwd', 'fwdbwd']
    assert all({key: row[key] for key in expected} == expected for row in results)
    assert all(row['op'] == ('naive_hc_pre[compile]' if compile else 'naive_hc_pre') for row in results)
    assert events == (['compile'] if compile else []) + ['time', 'time']


@pytest.mark.parametrize('backend', ['triton', 'gluon', 'tilelang'])
def test_benchmark_backend_overrides_inherited_flags(monkeypatch, backend):
    for name in ('FLA_GLUON', 'FLA_TILELANG', 'FLA_HC_GLUON', 'FLA_HC_TILELANG'):
        monkeypatch.setenv(name, '1')
    run.benchmark_op(op_name='hc_pre', shapes={}, backend=backend)
    assert os.environ['FLA_GLUON'] == '0'
    assert os.environ['FLA_TILELANG'] == '0'
    assert os.environ['FLA_HC_GLUON'] == str(int(backend == 'gluon'))
    assert os.environ['FLA_HC_TILELANG'] == str(int(backend == 'tilelang'))


@pytest.mark.parametrize('custom_shapes', [False, True])
def test_benchmark_worker_preserves_options(monkeypatch, custom_shapes):
    shapes = {'custom': {'B': 1, 'T': 3, 'H': 1, 'D': 7, 'N': 2}} if custom_shapes else None
    cmd = run._isolated_bench_cmd('run.py', ['naive_hc_pre'], shapes, ['fwd'], 'triton', 'unused.json', compile=True)
    del cmd[cmd.index('--json'):cmd.index('--json') + 2]
    monkeypatch.setattr(sys, 'argv', cmd[1:])
    monkeypatch.setenv('FLA_BENCH_WORKER', '1')
    monkeypatch.setattr(run, '_get_machine_info', lambda: {})
    calls = []

    def benchmark_op(op_name, shape_configs, **kwargs):
        calls.append((op_name, shape_configs, kwargs))
        return []

    monkeypatch.setattr(run, 'benchmark_op', benchmark_op)
    run.main()
    assert calls == [('naive_hc_pre', shapes, {'modes': ['fwd'], 'backend': 'triton', 'compile': True})]


@pytest.mark.parametrize('axis', ['L', 'N'])
def test_benchmark_table_distinguishes_residual_counts(capsys, axis):
    row = {'op': 'residual', 'mode': 'fwd', 'B': 1, 'T': 3, 'H': 1, 'D': 7, 'median_ms': 1.0}
    results = [{**row, axis: n} for n in (2, 4)]
    run.print_results_table(results=results, baseline=[{**r, 'median_ms': 2.0} for r in results])
    output = capsys.readouterr().out
    header = next(line for line in output.splitlines() if line.strip().startswith('fwd'))
    assert header.split()[:6] == ['fwd', axis, 'B', 'T', 'H', 'D']
    rows = [line.split() for line in output.splitlines() if 'residual' in line]
    assert [row[:5] for row in rows] == [['2', '1', '3', '1', '7'], ['4', '1', '3', '1', '7']]
    assert output.count('2.00x') == 2
