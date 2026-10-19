# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """One measured case, with units in metric names and an explicit comparison metric."""

    name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    primary_metric: str = 'median_ms'
    higher_is_better: bool = False


def _result_key(result: dict) -> tuple[str, str, str, str]:
    return result.get('benchmark', ''), result.get('scenario', ''), result['name'], json.dumps(result['parameters'], sort_keys=True)


def format_table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return str(value).replace('|', '&#124;').replace('\n', ' ')

    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines.extend('| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows)
    return '\n'.join(lines)


def format_parameters(parameters: dict) -> str:
    return ', '.join(f'{key}={value}' for key, value in parameters.items())


def format_results(results: list[dict]) -> str:
    rows = [[
        result.get('benchmark', ''), result.get('scenario', ''), result['name'],
        format_parameters(result['parameters']),
        ', '.join(f'{key}={value:.4g}' for key, value in result['metrics'].items()),
    ] for result in results]
    return format_table(headers=['Object', 'Scenario', 'Case', 'Parameters', 'Measurements'], rows=rows)


def print_results(results: list[dict]) -> None:
    print(format_results(results=results))


def format_comparison(changes: list[dict]) -> str:
    rows = [[
        change.get('benchmark', ''), change.get('scenario', ''), change['name'],
        format_parameters(change['parameters']), change['metric'],
        f"{change['base']:.4g}", f"{change['head']:.4g}", f"{change['change_pct']:+.1f}%",
        'regression' if change['regression'] else 'speedup' if change['speedup'] else 'within threshold',
    ] for change in changes]
    return format_table(headers=['Object', 'Scenario', 'Case', 'Parameters', 'Metric', 'Base', 'Head', 'Change', 'Status'], rows=rows)


def compare_results(base: list[dict], head: list[dict], threshold: float) -> list[dict]:
    """Compare matching cases; positive change denotes a regression for either metric direction."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and non-negative')
    base_by_key = {_result_key(result): result for result in base}
    head_by_key = {_result_key(result): result for result in head}
    if len(base_by_key) != len(base) or len(head_by_key) != len(head):
        raise ValueError('Duplicate benchmark case names and parameters')
    if base_by_key.keys() != head_by_key.keys():
        raise ValueError('Base and head must return the same benchmark cases')

    changes = []
    for key, current in head_by_key.items():
        previous = base_by_key[key]
        metric = current['primary_metric']
        higher_is_better = current['higher_is_better']
        if metric != previous['primary_metric'] or higher_is_better != previous['higher_is_better']:
            raise ValueError(f'Incompatible comparison metric for {current["name"]}')
        old, new = previous['metrics'][metric], current['metrics'][metric]
        if not math.isfinite(old) or not math.isfinite(new) or old <= 0 or new <= 0:
            raise ValueError(f'Comparison metric must be positive for {current["name"]}')
        change = (new - old) / old * (-100 if higher_is_better else 100)
        changes.append({
            'benchmark': current.get('benchmark', ''),
            'scenario': current.get('scenario', ''),
            'name': current['name'],
            'parameters': current['parameters'],
            'metric': metric,
            'base': old,
            'head': new,
            'change_pct': change,
            'regression': change > threshold,
            'speedup': change < -threshold,
        })
    return changes


def format_machine_info(info: dict) -> str:
    gpu = info.get('gpu_name', 'N/A')
    pytorch = info.get('pytorch_version', 'N/A')
    platform = info.get('device_platform', 'N/A')
    if platform == 'cuda':
        backend = f"CUDA {info.get('cuda_version', 'N/A')}"
    elif platform == 'npu':
        backend = f"NPU: CANN {info.get('cann_version', 'N/A')}"
    else:
        backend = platform.upper() if platform != 'N/A' else 'N/A'
    return f"Machine: {gpu} | {backend} | PyTorch {pytorch}"


def get_result_path(path: Path, rank: int) -> Path:
    return path if rank == 0 else path.with_stem(f'{path.stem}.rank{rank}')
