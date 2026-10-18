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


@pytest.mark.parametrize('benchmark', ['ops.latency', 'modules.conv', 'models.training', 'cp.kda'])
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
        assert name == f'benchmarks.{benchmark}'
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
    }]
    if rank:
        assert not output.exists()


def test_list_does_not_import_workloads(monkeypatch, capsys):
    def import_module(name):
        raise AssertionError(f'Unexpected workload import: {name}')

    monkeypatch.setattr(run.importlib, 'import_module', import_module)
    assert run.main(['--list']) == 0
    assert set(capsys.readouterr().out.splitlines()) == set(run.list_benchmarks())


def test_workload_help_includes_its_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        run.main(['ops.graph', '--help'])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert '--iterations' in help_text and '--json' in help_text


def test_failed_workload_does_not_publish_results(tmp_path, monkeypatch):
    output = tmp_path / 'results.json'

    def measure(args):
        raise RuntimeError('measurement failed')

    workload = SimpleNamespace(add_arguments=lambda parser: None, run=measure)
    monkeypatch.setattr(run.importlib, 'import_module', lambda name: workload)
    with pytest.raises(RuntimeError, match='measurement failed'):
        run.main(['models.training', '--json', str(output)])
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
