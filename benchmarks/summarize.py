# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Summarize measurement and comparison artifacts without averaging different cases."""

import argparse
import csv
import fnmatch
import io
import json
import math
from pathlib import Path

from benchmarks.utils.reporting import compare_results, format_machine_info, format_parameters, format_table


def _normalize_results(results: list[dict]) -> list[dict]:
    normalized = []
    for result in results:
        if 'op' in result:
            metrics = {key: result[key] for key in ('median_ms', 'p20_ms', 'p80_ms') if key in result}
            result = {
                'benchmark': 'operator_latency',
                'name': result['op'],
                'parameters': {key: value for key, value in result.items() if key not in {'op', *metrics}},
                'metrics': metrics,
                'primary_metric': 'median_ms',
                'higher_is_better': False,
            }
        if not result['metrics'] or result['primary_metric'] not in result['metrics']:
            raise ValueError(f'Missing measurements for {result["name"]}')
        if any(not isinstance(value, int | float) or not math.isfinite(value) for value in result['metrics'].values()):
            raise ValueError(f'Non-finite measurements for {result["name"]}')
        normalized.append(result)
    if not normalized:
        raise ValueError('Artifact contains no measurements')
    return normalized


def load_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if 'results' in artifact:
        artifact['results'] = _normalize_results(results=artifact['results'])
    else:
        for key in ('base_results', 'head_results'):
            artifact[key] = _normalize_results(results=artifact[key])
        artifact['changes'] = compare_results(
            base=artifact['base_results'], head=artifact['head_results'], threshold=artifact.get('threshold', 5.0),
        )
    return artifact


def collect_rows(artifacts: list[tuple[Path, dict]], benchmark: str, name: str, all_metrics: bool) -> list[dict]:
    rows = []
    for path, artifact in artifacts:
        info = artifact.get('machine_info', {})
        metadata = {
            'source': str(path), 'revision': info.get('git_label', ''), 'gpu': info.get('gpu_name', ''),
            'rank': info.get('rank', 0), 'world_size': info.get('world_size', 1),
        }
        for result in artifact.get('results', artifact.get('changes', [])):
            task = result.get('benchmark', artifact.get('benchmark', ''))
            if not fnmatch.fnmatchcase(task, benchmark) or not fnmatch.fnmatchcase(result['name'], name):
                continue
            row = {**metadata, 'benchmark': task, 'name': result['name'], 'parameters': result['parameters']}
            if 'metrics' in result:
                metrics = result['metrics']
                if not all_metrics:
                    metrics = {result['primary_metric']: metrics[result['primary_metric']]}
                rows.extend({**row, 'metric': metric, 'value': value} for metric, value in metrics.items())
            else:
                rows.append({
                    **row, 'base_sha': artifact['base_sha'], 'head_sha': artifact['head_sha'],
                    **{key: result[key] for key in ('metric', 'base', 'head', 'change_pct')},
                    'status': 'regression' if result['regression'] else 'speedup' if result['speedup'] else 'within threshold',
                })
    if not rows:
        raise ValueError('No measurements match the benchmark and name filters')
    return rows


def summarize(artifacts: list[tuple[Path, dict]], rows: list[dict], output_format: str) -> str:
    if output_format == 'csv':
        output = io.StringIO()
        fields = [
            'source', 'revision', 'gpu', 'rank', 'world_size', 'benchmark', 'name', 'parameters',
            'metric', 'value', 'base_sha', 'head_sha', 'base', 'head', 'change_pct', 'status',
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, 'parameters': json.dumps(row['parameters'], sort_keys=True)} for row in rows)
        return output.getvalue()

    sources = {row['source'] for row in rows}
    source_ids = {}
    source_rows = []
    for path, artifact in artifacts:
        if str(path) not in sources or str(path) in source_ids:
            continue
        source_ids[str(path)] = len(source_ids) + 1
        info = artifact.get('machine_info', {})
        revision = info.get('git_label', 'unknown')
        if 'base_sha' in artifact:
            revision = f"{artifact['base_sha']} → {artifact['head_sha']}"
        source_rows.append([
            source_ids[str(path)], path, revision, format_machine_info(info),
            f"{info.get('rank', 0)}/{info.get('world_size', 1)}",
        ])
    sections = [format_table(headers=['Source', 'File', 'Revision', 'Environment', 'Rank'], rows=source_rows)]
    for benchmark in sorted({row['benchmark'] for row in rows}):
        sections.append(f'## {benchmark}')
        measurements, comparisons = [], []
        for row in rows:
            if row['benchmark'] != benchmark:
                continue
            cells = [source_ids[row['source']], row['name'], format_parameters(row['parameters']), row['metric']]
            if 'value' in row:
                measurements.append([*cells, f"{row['value']:.4g}"])
            else:
                comparisons.append([
                    *cells, f"{row['base']:.4g}", f"{row['head']:.4g}", f"{row['change_pct']:+.1f}%", row['status'],
                ])
        if measurements:
            sections.append(format_table(headers=['Source', 'Case', 'Parameters', 'Metric', 'Value'], rows=measurements))
        if comparisons:
            sections.append(format_table(
                headers=['Source', 'Case', 'Parameters', 'Metric', 'Base', 'Head', 'Change', 'Status'], rows=comparisons,
            ))
    return '\n\n'.join(sections) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='JSON artifacts from run or compare')
    parser.add_argument('--output', type=Path, help='Write the summary to a file instead of stdout')
    parser.add_argument('--format', choices=['markdown', 'csv'], help='Defaults to CSV for .csv output, otherwise Markdown')
    parser.add_argument('--benchmark', default='*', help='Filter by measurement task; accepts quoted patterns')
    parser.add_argument('--name', default='*', help='Filter by case name, such as chunk_kda; accepts quoted patterns')
    parser.add_argument(
        '--all-metrics',
        action='store_true',
        help='Include secondary measurements; comparisons use the primary metric',
    )
    args = parser.parse_args(args=argv)
    output_format = args.format or ('csv' if args.output and args.output.suffix == '.csv' else 'markdown')
    try:
        paths = dict.fromkeys(path.resolve() for path in args.inputs)
        artifacts = [(path, load_artifact(path=path)) for path in paths]
        rows = collect_rows(artifacts=artifacts, benchmark=args.benchmark, name=args.name, all_metrics=args.all_metrics)
        summary = summarize(artifacts=artifacts, rows=rows, output_format=output_format)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
    else:
        print(summary, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
