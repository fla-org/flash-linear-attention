# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import csv
import json
from dataclasses import asdict

import pytest

from benchmarks import summarize
from benchmarks.utils.reporting import BenchmarkResult, compare_results


def test_summarize_multiple_artifacts_preserves_metrics_and_provenance(tmp_path):
    inputs = []
    for rank, (benchmark, metrics, primary_metric) in enumerate([
        ('operator_cp', {'median_ms': 1.25, 'p80_ms': 1.5}, 'median_ms'),
        ('training', {'tokens_per_second': 100.0, 'memory_gb': 2.5}, 'tokens_per_second'),
    ]):
        result = {
            **asdict(BenchmarkResult(
                name='kda', parameters={'T': 128, 'dtype': 'bfloat16'}, metrics=metrics,
                primary_metric=primary_metric, higher_is_better=primary_metric == 'tokens_per_second',
            )),
            'benchmark': benchmark,
        }
        path = tmp_path / f'input{rank}.json'
        path.write_text(json.dumps({
            'machine_info': {'git_label': f'revision{rank}', 'gpu_name': 'test GPU', 'rank': rank, 'world_size': 2},
            'results': [result],
        }))
        inputs.append(str(path))
    markdown = tmp_path / 'summary.md'
    assert summarize.main([*inputs, '--all-metrics', '--output', str(markdown)]) == 0
    text = markdown.read_text()
    assert 'tokens_per_second | 100' in text and 'p80_ms | 1.5' in text and 'memory_gb | 2.5' in text
    assert 'operator_cp' in text and 'training' in text
    assert '1/2' in text and 'revision1' in text and 'dtype=bfloat16' in text
    output = tmp_path / 'nested/summary.csv'
    assert summarize.main([*inputs, '--all-metrics', '--output', str(output)]) == 0
    with output.open() as file:
        rows = list(csv.DictReader(file))
    assert [(row['metric'], float(row['value'])) for row in rows] == [
        ('median_ms', 1.25), ('p80_ms', 1.5), ('tokens_per_second', 100.0), ('memory_gb', 2.5),
    ]
    assert [row['rank'] for row in rows] == ['0', '0', '1', '1']
    assert all(json.loads(row['parameters']) == {'T': 128, 'dtype': 'bfloat16'} for row in rows)


@pytest.mark.parametrize('legacy', [False, True], ids=['object-comparison', 'ci-comparison'])
def test_summarize_comparison_retains_threshold_and_direction(tmp_path, legacy):
    base = {**asdict(BenchmarkResult(name='chunk_kda', parameters={'T': 64}, metrics={'median_ms': 1.0})),
            'benchmark': 'operator_cp'}
    head = {**base, 'metrics': {'median_ms': 1.1}}
    if legacy:
        base, head = [{'op': row['name'], **row['parameters'], **row['metrics']} for row in (base, head)]
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps({
        'base_sha': 'base', 'head_sha': 'head', 'base_results': [base], 'head_results': [head], 'threshold': 20,
    }))
    output = tmp_path / 'summary.csv'
    assert summarize.main([str(path), '--output', str(output)]) == 0
    with output.open() as file:
        row, = csv.DictReader(file)
    assert row['base_sha'] == 'base' and row['head_sha'] == 'head'
    assert float(row['change_pct']) == pytest.approx(10)
    assert row['status'] == 'within threshold'
    assert json.loads(row['parameters']) == {'T': 64}


@pytest.mark.parametrize('invalid', ['empty', 'missing-case', 'nan'])
def test_summarize_rejects_invalid_artifact(tmp_path, invalid):
    base = asdict(BenchmarkResult(name='case', parameters={}, metrics={'median_ms': 1.0}))
    data = {'results': []}
    if invalid == 'missing-case':
        data = {'base_results': [base], 'head_results': [{**base, 'name': 'different'}]}
    elif invalid == 'nan':
        data = {'results': [{**base, 'metrics': {'median_ms': float('nan')}}]}
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data))
    output = tmp_path / 'summary.md'
    with pytest.raises(SystemExit) as error:
        summarize.main([str(path), '--output', str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_comparison_distinguishes_measurement_tasks():
    result = asdict(BenchmarkResult(name='triton', parameters={'T': 64}, metrics={'median_ms': 1.0}))
    results = [{**result, 'benchmark': benchmark} for benchmark in ['conv_latency', 'conv_cp', 'layernorm_latency']]
    changes = compare_results(base=results, head=results, threshold=5.0)
    assert len(changes) == 3


def test_summary_filters_cases_and_defaults_to_primary_metrics(tmp_path):
    path = tmp_path / 'results.json'
    records = [{
        **asdict(BenchmarkResult(name=name, parameters={'T': 64}, metrics={'median_ms': 1.0, 'p80_ms': 1.5})),
        'benchmark': benchmark,
    } for benchmark, name in [('operator_latency', 'chunk_kda'), ('operator_latency', 'chunk_gla'), ('conv_latency', 'triton')]]
    path.write_text(json.dumps({'results': records}))
    output = tmp_path / 'summary.csv'
    assert summarize.main([
        str(path), str(path), '--benchmark', 'operator_*', '--name', '*kda', '--output', str(output),
    ]) == 0
    with output.open() as file:
        row, = csv.DictReader(file)
    assert row['name'] == 'chunk_kda' and row['metric'] == 'median_ms'
    missing = tmp_path / 'missing.md'
    with pytest.raises(SystemExit) as error:
        summarize.main([str(path), '--name', 'absent', '--output', str(missing)])
    assert error.value.code == 2
    assert not missing.exists()
