# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import triton

from benchmarks import run
from benchmarks.utils import distributed, environment
from benchmarks.utils.perf_report import run_module_benchmark
from benchmarks.utils.reporting import BenchmarkResult


@pytest.mark.parametrize('benchmark', ['operator_latency', 'conv_latency', 'training', 'layer_cp'])
@pytest.mark.parametrize('rank', [0, 1])
def test_workload_runner_records_results_and_rank(tmp_path, monkeypatch, benchmark, rank):
    output = tmp_path / 'results.json'
    monkeypatch.setenv('RANK', str(rank))
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setattr(environment, 'get_machine_info', lambda: {'git_label': 'test'})

    def add_arguments(parser):
        parser.add_argument('--tokens', type=int, required=True)

    def measure(args):
        return [BenchmarkResult(name='case', parameters={'T': args.tokens}, metrics={'median_ms': 1.25})]

    def import_module(name):
        assert name == f'benchmarks.workloads.{benchmark}'
        return SimpleNamespace(add_arguments=add_arguments, run=measure)

    monkeypatch.setattr(run.importlib, 'import_module', import_module)
    assert run.main([benchmark, '--tokens', '128', '--json', str(output)]) == 0
    path = output if rank == 0 else tmp_path / 'results.rank1.json'
    payload = json.loads(path.read_text())
    assert payload['benchmark'] == benchmark
    assert payload['machine_info'] == {'git_label': 'test', 'rank': rank, 'world_size': 2}
    assert payload['results'] == [{
        'name': 'case', 'parameters': {'T': 128}, 'metrics': {'median_ms': 1.25},
        'primary_metric': 'median_ms', 'higher_is_better': False,
        'benchmark': benchmark,
    }]
    if rank:
        assert not output.exists()


def test_list_does_not_import_workloads(monkeypatch, capsys):
    def import_module(name):
        raise AssertionError(f'Unexpected workload import: {name}')

    monkeypatch.setattr(run.importlib, 'import_module', import_module)
    assert run.main(['--list']) == 0
    output = capsys.readouterr().out
    assert all(f'| {name} |' in output for name in run.list_benchmarks())
    assert 'torchrun' in output and 'Ascend NPU' in output
    assert run.main(['*_cp', '--list']) == 0
    filtered = capsys.readouterr().out
    assert '| operator_cp |' in filtered and '| training |' not in filtered


@pytest.mark.parametrize(('benchmark', 'option'), [('kda_graph', '--iterations'), ('operator_latency', 'chunk_kda')])
def test_workload_help_includes_its_arguments(capsys, benchmark, option):
    with pytest.raises(SystemExit) as error:
        run.main([benchmark, '--help'])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert option in help_text and '--json' in help_text


def test_failed_workload_does_not_publish_results(tmp_path, monkeypatch):
    output = tmp_path / 'results.json'

    def measure(args):
        raise RuntimeError('measurement failed')

    workload = SimpleNamespace(add_arguments=lambda parser: None, run=measure)
    monkeypatch.setattr(run.importlib, 'import_module', lambda name: workload)
    with pytest.raises(RuntimeError, match='measurement failed'):
        run.main(['training', '--json', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('multiple', [False, True], ids=['single-report', 'report-list'])
def test_module_report_preserves_parameters_and_measurements(tmp_path, multiple):
    pytest.importorskip('pandas')
    pytest.importorskip('matplotlib')
    config = triton.testing.Benchmark(
        x_names=['T', 'dtype', 'softcap'],
        x_vals=[(64, torch.bfloat16, None), (128, torch.float32, 3.0)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['triton', 'torch'],
        ylabel='ms',
        plot_name='measurements',
        args={'packed': True},
    )

    @triton.testing.perf_report([config] if multiple else config)
    def measure(T, dtype, softcap, provider, packed):
        assert isinstance(dtype, torch.dtype) and packed
        elapsed = T / (64 if provider == 'triton' else 32)
        return elapsed, elapsed * 0.9, elapsed * 1.1

    results = run_module_benchmark(mark=measure, save_dir=str(tmp_path))
    records = json.loads(json.dumps([asdict(result) for result in results], allow_nan=False))
    assert [record['name'] for record in records] == ['triton', 'torch', 'triton', 'torch']
    assert [record['metrics']['median_ms'] for record in records] == [1.0, 2.0, 2.0, 4.0]
    assert records[0]['parameters'] == {'T': 64, 'dtype': 'torch.bfloat16', 'softcap': None, 'packed': True}
    assert records[2]['parameters'] == {'T': 128, 'dtype': 'torch.float32', 'softcap': 3.0, 'packed': True}
    assert (tmp_path / 'measurements.csv').is_file()


@pytest.mark.parametrize('fail', [False, True])
def test_distributed_context_cleans_up(monkeypatch, fail):
    events = []
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setattr(distributed.torch.cuda, 'set_device', lambda index: events.append(('device', index)))
    monkeypatch.setattr(distributed.dist, 'init_process_group', lambda backend: events.append(('init', backend)))
    monkeypatch.setattr(distributed.dist, 'destroy_process_group', lambda: events.append(('destroy',)))
    try:
        with distributed.distributed_context(backend='nccl'):
            events.append(('measure',))
            if fail:
                raise RuntimeError('measurement failed')
    except RuntimeError:
        assert fail
    assert events == [('device', 2), ('init', 'nccl'), ('measure',), ('destroy',)]


@pytest.mark.parametrize('invalid', [False, True])
def test_pattern_validates_all_arguments_before_measuring(tmp_path, monkeypatch, invalid):
    output = tmp_path / 'nested/results.json'
    monkeypatch.setattr(run, 'WORKLOADS', {'first': ('first', 'CPU'), 'second': ('second', 'CPU')})
    monkeypatch.setattr(environment, 'get_machine_info', lambda: {})
    planned = []
    measured = []

    def measure(args):
        assert planned == ['benchmarks.workloads.first', 'benchmarks.workloads.second']
        measured.append(args)
        return [BenchmarkResult(name='case', parameters={}, metrics={'median_ms': 1.0})]

    def import_module(name):
        def add_arguments(parser):
            planned.append(name)
            if invalid and name.endswith('second'):
                parser.add_argument('--required', required=True)
        return SimpleNamespace(add_arguments=add_arguments, run=measure)

    monkeypatch.setattr(run.importlib, 'import_module', import_module)
    if invalid:
        with pytest.raises(SystemExit) as error:
            run.main(['*', '--json', str(output)])
        assert error.value.code == 2
        assert not measured and not output.exists()
    else:
        assert run.main(['*', '--json', str(output)]) == 0
        results = json.loads(output.read_text())['results']
        assert [r['benchmark'] for r in results] == ['first', 'second']


def test_empty_selection_does_not_publish_results(tmp_path, monkeypatch):
    output = tmp_path / 'results.json'
    workload = SimpleNamespace(add_arguments=lambda parser: None, run=lambda args: [])
    monkeypatch.setattr(run.importlib, 'import_module', lambda name: workload)
    with pytest.raises(RuntimeError, match='No benchmark measurements'):
        run.main(['operator_latency', '--json', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('compile', [False, True])
def test_operator_task_runs_selected_variants(tmp_path, monkeypatch, compile):
    from benchmarks.workloads import operator_latency

    monkeypatch.setattr(environment, 'get_machine_info', lambda: {})
    calls = []

    def benchmark_op(op_name, shapes, modes, backend, compile):
        calls.append((op_name, modes, backend, compile))
        return [BenchmarkResult(name=op_name, parameters=shapes['small'], metrics={'median_ms': 1.0})]

    monkeypatch.setattr(operator_latency, 'benchmark_op', benchmark_op)
    output = tmp_path / 'results.json'
    assert run.main([
        'operator_latency', '--op', 'chunk_gdn', 'chunk_gdn2', '--modes', 'fwd', '--backend', 'triton',
        '--custom-shapes', '{"small": {"B": 1, "T": 64, "H": 2, "D": 16}}', '--json', str(output),
        *(['--compile'] if compile else []),
    ]) == 0
    assert calls == [('chunk_gdn', ['fwd'], 'triton', compile), ('chunk_gdn2', ['fwd'], 'triton', compile)]
    results = json.loads(output.read_text())['results']
    assert [result['benchmark'] for result in results] == ['operator_latency', 'operator_latency']
