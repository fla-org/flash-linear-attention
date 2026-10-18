# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run a benchmark workload and record its measurements and environment."""

import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.utils.reporting import format_machine_info, get_result_path, print_results


def list_benchmarks() -> list[str]:
    root = Path(__file__).parent
    return sorted(
        f'{category}.{path.stem}'
        for category in ('ops', 'modules', 'models', 'cp')
        for path in (root / category).glob('*.py')
        if path.stem != '__init__'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('benchmark', nargs='?', choices=list_benchmarks())
    parser.add_argument('--list', action='store_true', help='List available workloads')
    parser.add_argument('--json', type=Path, help='Write measurements and machine metadata to JSON')
    selected, _ = parser.parse_known_args(args=argv)
    parser.add_argument('-h', '--help', action='help')
    if selected.list:
        print('\n'.join(list_benchmarks()))
        return 0
    if selected.benchmark is None:
        parser.parse_args(args=argv)
        parser.error('benchmark is required unless --list is used')

    workload = importlib.import_module(f'benchmarks.{selected.benchmark}')
    workload.add_arguments(parser)
    args = parser.parse_args(args=argv)
    results = [asdict(result) for result in workload.run(args)]

    from benchmarks.utils.environment import get_machine_info

    rank = int(os.environ.get('RANK', '0'))
    machine_info = get_machine_info()
    machine_info.update(rank=rank, world_size=int(os.environ.get('WORLD_SIZE', '1')))
    if rank == 0:
        print(format_machine_info(machine_info))
        print_results(results=results)
    if args.json:
        output = get_result_path(path=args.json, rank=rank)
        payload = {'benchmark': args.benchmark, 'machine_info': machine_info, 'results': results}
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        print(f'Results saved to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
