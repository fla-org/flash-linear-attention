# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Measure selected benchmark objects in a supported execution scenario."""

import argparse
import fnmatch
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
        for category in ('ops', 'modules', 'layers', 'models')
        for path in (root / category).glob('*.py')
        if path.stem != '__init__'
    )


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(description=__doc__, add_help=False)
    common.add_argument('benchmark', nargs='?', help='Object name or quoted pattern, such as ops.kda or ops.*')
    common.add_argument('--scenario', help="Execution scenario; defaults to the object's first scenario")
    common.add_argument('--list', action='store_true', help='List benchmark objects without importing their dependencies')
    common.add_argument('--json', type=Path, help='Write measurements and machine metadata to JSON')
    selected, _ = common.parse_known_args(args=argv)
    if selected.list:
        print('\n'.join(list_benchmarks()))
        return 0
    if selected.benchmark is None:
        argparse.ArgumentParser(parents=[common], description=__doc__).parse_args(args=argv)
        common.error('benchmark is required unless --list is used')
    names = [name for name in list_benchmarks() if fnmatch.fnmatchcase(name, selected.benchmark)]
    if not names:
        common.error(f'No benchmark matches {selected.benchmark!r}; use --list to see available objects')

    plans = []
    for name in names:
        workload = importlib.import_module(f'benchmarks.{name}')
        scenario = selected.scenario or workload.SCENARIOS[0]
        if scenario not in workload.SCENARIOS:
            if name == selected.benchmark:
                common.error(f'{name} supports scenarios: {", ".join(workload.SCENARIOS)}')
            continue
        parser = argparse.ArgumentParser(
            parents=[common], description=f'{name}; supported scenarios: {", ".join(workload.SCENARIOS)}',
        )
        workload.add_arguments(parser, scenario)
        args = parser.parse_args(args=argv)
        args.scenario = scenario
        plans.append((name, workload, args))
    if not plans:
        common.error(f'No matching object supports scenario {selected.scenario!r}')

    results = []
    for name, workload, args in plans:
        for result in workload.run(args):
            results.append({**asdict(result), 'benchmark': name, 'scenario': args.scenario})
    if not results:
        raise RuntimeError('No benchmark measurements were produced')

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
