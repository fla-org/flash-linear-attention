# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Prewarm GDN compilation caches across devices and collect coverage evidence."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    shapes: dict[str, dict[str, Any]]
    graph: bool = False


def partition_workloads(ordinary, graph, devices):
    if len(devices) == 1:
        return {devices[0]: [Workload('ordinary', ordinary), Workload('graph', graph, graph=True)]}
    # keep D128 ordinary cases together and graph cases of each dtype together to reuse compilation.
    return {
        devices[0]: [Workload('ordinary-d128', {name: s for name, s in ordinary.items() if s['D'] == 128})],
        devices[1]: [Workload('ordinary-other', {name: s for name, s in ordinary.items() if s['D'] != 128})],
        devices[2]: [Workload('graph-fp16', {name: s for name, s in graph.items() if s['dtype'] == 'float16'}, graph=True)],
        devices[3]: [Workload('graph-bf16', {name: s for name, s in graph.items() if s['dtype'] == 'bfloat16'}, graph=True)],
    }


def _prewarm_worker(device, workloads, output_dir):
    env = {**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': device}
    for workload in workloads:
        command = [sys.executable, '-m', 'benchmarks.ops.run', '--op', 'chunk_gdn']
        if workload.graph:
            command += ['--graph', '--no-base', '--modes', 'fwd', 'fwdbwd',
                        '--measure-graph-safe', '--report-memory', '--warmup', '3', '--iterations', '10']
        else:
            command += ['--base', '']
        command += ['--custom-shapes', json.dumps(workload.shapes), '--json', str(output_dir / f'{workload.name}.json')]
        started = time.monotonic()
        log = output_dir / f'{workload.name}.log'
        with log.open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'{workload.name}: device={device}, elapsed={time.monotonic() - started:.2f}s, log={log}', flush=True)


def prewarm(ordinary, graph, devices, output_dir):
    plans = partition_workloads(ordinary, graph, devices)
    workloads = [workload for plan in plans.values() for workload in plan]
    for expected, is_graph in ((ordinary, False), (graph, True)):
        assigned = [name for w in workloads if w.graph == is_graph for name in w.shapes]
        if sorted(assigned) != sorted(expected):
            raise ValueError('GDN prewarm partition must assign each registered shape exactly once')
    output_dir.mkdir()
    started = time.monotonic()
    # the executor waits for all running workers even when one raises.
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(_prewarm_worker, device, [w for w in plan if w.shapes], output_dir)
                   for device, plan in plans.items()]
        for future in futures:
            future.result()

    ordinary_rows, graph_rows = [], []
    for workload in workloads:
        if not workload.shapes:
            continue
        result = json.loads((output_dir / f'{workload.name}.json').read_text())
        if workload.graph:
            graph_rows.extend(result['current']['results'])
        else:
            ordinary_rows.extend(result['results'])
    # shard timings are prewarm diagnostics. The formal benchmark runs in a separate workflow step.
    (output_dir / 'benchmark_chunk_gdn.json').write_text(json.dumps({'results': ordinary_rows, 'prewarm': True}))
    (output_dir / 'benchmark_gdn_graph.json').write_text(json.dumps({'current': {'results': graph_rows}, 'prewarm': True}))
    print(f'GDN prewarm: {len(devices)} device(s), elapsed={time.monotonic() - started:.2f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if len(args.devices) not in (1, 4) or len(set(args.devices)) != len(args.devices):
        parser.error('--devices must contain one or four distinct device IDs')

    from benchmarks.ops.graph_registry import get_graph_op
    from benchmarks.ops.registry import SHAPE_CONFIGS, get_op

    ordinary = get_op('chunk_gdn').default_shapes or SHAPE_CONFIGS
    graph = get_graph_op('chunk_gdn').default_shapes
    prewarm(ordinary, graph, args.devices, args.output_dir.resolve())


if __name__ == '__main__':
    main()
