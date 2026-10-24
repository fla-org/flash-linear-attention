# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run a named measurement task and save results for comparison or summary."""

import argparse
import fnmatch
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.utils.reporting import format_machine_info, format_table, get_result_path, print_results
from benchmarks.workloads import WORKLOADS


def list_benchmarks(pattern: str = '*') -> list[str]:
    return sorted(name for name in WORKLOADS if fnmatch.fnmatchcase(name, pattern))


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(description=__doc__, add_help=False)
    common.add_argument('benchmark', nargs='?', help='Measurement task or quoted pattern; use --list to see available tasks')
    common.add_argument(
        '--list',
        action='store_true',
        help='Show tasks, measurements, and required devices without loading them',
    )
    common.add_argument('--json', type=Path, help='Write measurements and machine metadata to JSON')
    selected, _ = common.parse_known_args(args=argv)
    names = list_benchmarks(pattern=selected.benchmark or '*')
    if selected.list:
        common.parse_args(args=argv)
        print(format_table(
            headers=['Benchmark', 'Measurement', 'Execution'],
            rows=[[name, *WORKLOADS[name]] for name in names],
        ))
        return 0
    if selected.benchmark is None:
        argparse.ArgumentParser(parents=[common], description=__doc__).parse_args(args=argv)
        common.error('benchmark is required unless --list is used')
    if not names:
        common.error(f'No benchmark matches {selected.benchmark!r}; use --list to see available tasks')

    plans = []
    for name in names:
        workload = importlib.import_module(f'benchmarks.workloads.{name}')
        parser = argparse.ArgumentParser(parents=[common], description=f'{name}: {WORKLOADS[name][0]} ({WORKLOADS[name][1]})')
        workload.add_arguments(parser)
        plans.append((name, workload, parser.parse_args(args=argv)))

    results = []
    for name, workload, args in plans:
        measured = workload.run(args)
        if not measured:
            raise RuntimeError(f'No benchmark measurements were produced by {name}')
        results.extend({**asdict(result), 'benchmark': name} for result in measured)

    from benchmarks.utils.environment import get_machine_info

    rank = int(os.environ.get('RANK', '0'))
    machine_info = get_machine_info()
    machine_info.update(rank=rank, world_size=int(os.environ.get('WORLD_SIZE', '1')))
    if rank == 0:
        print(format_machine_info(machine_info))
        print_results(results=results)
    if selected.json:
        output = get_result_path(path=selected.json, rank=rank)
        payload = {'benchmark': selected.benchmark, 'machine_info': machine_info, 'results': results}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        print(f'Results saved to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
