# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Summarize measurement and comparison artifacts without averaging different cases."""

import argparse
import csv
import io
import json
import math
from pathlib import Path

from benchmarks.utils.reporting import compare_results, format_comparison, format_machine_info, format_results


def _normalize_results(results: list[dict]) -> list[dict]:
    normalized = []
    for result in results:
        if 'op' in result:
            metrics = {key: result[key] for key in ('median_ms', 'p20_ms', 'p80_ms') if key in result}
            result = {
                'benchmark': 'ops',
                'scenario': 'latency',
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


def summarize(artifacts: list[tuple[Path, dict]], output_format: str) -> str:
    if output_format == 'markdown':
        sections = []
        for path, artifact in artifacts:
            info = artifact.get('machine_info', {})
            revision = info.get('git_label', 'unknown')
            metadata = f"Revision: {revision} | rank: {info.get('rank', 0)}/{info.get('world_size', 1)}"
            if 'results' in artifact:
                table = format_results(results=artifact['results'])
            else:
                metadata = f"Base: {artifact['base_sha']} | Head: {artifact['head_sha']} | {metadata}"
                table = format_comparison(changes=artifact['changes'])
            sections.append(f'## {path}\n\n{format_machine_info(info)}\n\n{metadata}\n\n{table}')
        return '\n\n'.join(sections) + '\n'

    output = io.StringIO()
    fields = [
        'source', 'revision', 'gpu', 'rank', 'world_size', 'benchmark', 'scenario', 'name', 'parameters',
        'metric', 'value', 'base_sha', 'head_sha', 'base', 'head', 'change_pct', 'status',
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for path, artifact in artifacts:
        info = artifact.get('machine_info', {})
        metadata = {
            'source': str(path), 'revision': info.get('git_label', ''), 'gpu': info.get('gpu_name', ''),
            'rank': info.get('rank', 0), 'world_size': info.get('world_size', 1),
        }
        records = artifact.get('results', artifact.get('changes', []))
        for result in records:
            row = {
                **metadata, 'benchmark': result.get('benchmark', ''), 'scenario': result.get('scenario', ''),
                'name': result['name'], 'parameters': json.dumps(result['parameters'], sort_keys=True),
            }
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    writer.writerow({**row, 'metric': metric, 'value': value})
            else:
                writer.writerow({
                    **row, 'base_sha': artifact['base_sha'], 'head_sha': artifact['head_sha'],
                    **{key: result[key] for key in ('metric', 'base', 'head', 'change_pct')},
                    'status': 'regression' if result['regression'] else 'speedup' if result['speedup'] else 'within threshold',
                })
    return output.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='JSON artifacts from run or compare')
    parser.add_argument('--output', type=Path, help='Write the summary to a file instead of stdout')
    parser.add_argument('--format', choices=['markdown', 'csv'], help='Defaults to CSV for .csv output, otherwise Markdown')
    args = parser.parse_args(args=argv)
    output_format = args.format or ('csv' if args.output and args.output.suffix == '.csv' else 'markdown')
    try:
        artifacts = [(path, load_artifact(path=path)) for path in args.inputs]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    summary = summarize(artifacts=artifacts, output_format=output_format)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
    else:
        print(summary, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
