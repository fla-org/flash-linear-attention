# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run operator benchmarks locally or against a git baseline. See benchmarks/README.md."""

import argparse
import json
import logging
import os
import sys

from benchmarks.ops.utils.discovery import find_affected_op_names, get_changed_files
from benchmarks.ops.utils.registry import SHAPE_CONFIGS, get_op, list_ops
from benchmarks.ops.utils.runner import benchmark_op, run_benchmarks
from benchmarks.utils.environment import get_git_label, get_machine_info
from benchmarks.utils.reporting import format_machine_info, print_results_table

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmark runner for flash-linear-attention ops',
    )
    parser.add_argument(
        '--op', nargs='+', default=None,
        help='Op name(s) to benchmark, or "all"',
    )
    parser.add_argument(
        '--backend', default=None,
        help="Op backend to select, e.g. 'triton' or 'gluon'. Toggles the backend's dispatch env "
             "var for ops that declare one (see OpConfig.backend_env); ignored otherwise.",
    )
    parser.add_argument(
        '--custom-shapes', default=None,
        help='JSON string to override default shapes, '
             'e.g. \'{"my": {"B":1,"T":2048,"H":16,"D":128}}\'',
    )
    parser.add_argument(
        '--modes', nargs='+', default=['fwd', 'fwdbwd'],
        choices=['fwd', 'fwdbwd'],
        help='Benchmark modes (default: fwd fwdbwd)',
    )
    parser.add_argument(
        '--json', dest='json_file', default=None,
        help='Output file path for JSON results',
    )
    parser.add_argument(
        '--base', default=None,
        help='Git ref for the baseline (old) column, e.g. "main" or "HEAD~3". '
             'Auto-detected as "main" when on a feature branch. '
             'HEAD and the baseline each run in an isolated subprocess.',
    )
    parser.add_argument(
        '--no-base', action='store_true',
        help='Skip baseline comparison. Nested worker processes pass this '
             'so they never spawn another compare.',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all registered ops and exit',
    )
    parser.add_argument(
        '--from-diff', action='store_true',
        help='Select ops from git diff (use with --diff-base / --diff-head, not with --op)',
    )
    parser.add_argument(
        '--diff-base', default='main',
        help='Base ref for --from-diff (default: main)',
    )
    parser.add_argument(
        '--diff-head', default='HEAD',
        help='Head ref for --from-diff (default: HEAD)',
    )
    args = parser.parse_args()

    if args.list:
        ops = list_ops()
        print(f"Registered ops ({len(ops)}):")
        for name in ops:
            cfg = get_op(name)
            print(f"  {name:30s}  [{cfg.category}]  {cfg.import_path}")
        return

    if args.from_diff:
        if args.op is not None:
            parser.error('--from-diff cannot be used with --op')
        changed = get_changed_files(base=args.diff_base, head=args.diff_head)
        op_names = find_affected_op_names(changed_files=changed)
        if not op_names:
            print('No affected ops for this diff.', file=sys.stderr)
            return
    elif args.op is None:
        parser.error("--op is required unless --from-diff (use --list to see available ops)")

    if not args.from_diff:
        op_names = list_ops() if args.op == ['all'] else args.op
    shape_configs = json.loads(args.custom_shapes) if args.custom_shapes else SHAPE_CONFIGS

    is_worker = os.environ.get('FLA_BENCH_WORKER') == '1'
    print(f"Git: {get_git_label()}")
    print(f"Shapes: {len(shape_configs)} configs")
    print(f"Ops: {op_names}")

    if is_worker:
        machine_info = get_machine_info()
        print(format_machine_info(machine_info))
        all_results = []
        for op_name in op_names:
            try:
                all_results.extend(
                    benchmark_op(op_name=op_name, shapes=shape_configs, modes=args.modes, backend=args.backend),
                )
            except Exception as e:
                logger.error(f"Failed to benchmark {op_name}: {e}")
        baseline, baseline_info = None, None
    else:
        all_results, machine_info = run_benchmarks(
            op_names=op_names, shapes=shape_configs, modes=args.modes, backend=args.backend,
        )
        all_results = all_results or []
        machine_info = machine_info or {'git_label': get_git_label()}

        git_label = machine_info.get('git_label', 'unknown')
        current_branch = git_label.split('[')[0] if '[' in git_label else git_label
        base_ref = None if args.no_base else args.base
        if base_ref is None and not args.no_base and current_branch not in ('main', 'master', 'unknown'):
            base_ref = 'main'
        if base_ref == '':
            base_ref = None

        baseline, baseline_info = None, None
        if base_ref:
            baseline, baseline_info = run_benchmarks(
                op_names=op_names,
                shapes=shape_configs,
                modes=args.modes,
                backend=args.backend,
                ref=base_ref,
            )

    mode_order = {'fwd': 0, 'fwdbwd': 1}
    all_results.sort(key=lambda r: (
        mode_order.get(r['mode'], 9),
        r.get('L', 0),
        r['B'], r['T'], r['H'], r['D'], r['op'],
    ))

    # workers already print their own results when no baseline is requested.
    if is_worker or baseline:
        print_results_table(results=all_results, machine_info=machine_info, baseline=baseline, baseline_info=baseline_info)

    if args.json_file:
        output = {'machine_info': machine_info, 'results': all_results}
        with open(args.json_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json_file}")

    return all_results


if __name__ == "__main__":
    main()
