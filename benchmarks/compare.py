# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare workload results across revisions, with affected-operator selection for CI."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmarks.configs.operators import SHAPE_CONFIGS
from benchmarks.utils.discovery import find_affected_op_names, get_changed_files
from benchmarks.utils.reporting import compare_results, format_comparison, get_result_path
from benchmarks.utils.revisions import get_commit_sha, run_benchmark, run_operator_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Compare benchmarks between two commits")
    parser.add_argument("--base", default="main", help="Base commit/branch/tag (default: main)")
    parser.add_argument("--head", default="HEAD", help="Head commit/branch/tag (default: HEAD)")
    parser.add_argument("--benchmark-ops", nargs="*", help="Op names to benchmark. If omitted, auto-detect from diff.")
    parser.add_argument('--benchmark', help='Measurement task or quoted pattern to compare instead of CI operator discovery')
    parser.add_argument("--threshold", type=float, default=5.0, help="Regression threshold in percent (default: 5.0)")
    parser.add_argument("--output", help="Save comparison results to JSON file")
    parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Do not exit with non-zero code on performance regression",
    )
    parser.add_argument("--plot", default=None, help="Save comparison visualization plots to this directory")
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Workload arguments after --')
    args = parser.parse_args()

    if args.benchmark and args.benchmark_ops:
        parser.error('--benchmark and --benchmark-ops are mutually exclusive')
    if args.arguments and not args.benchmark:
        parser.error('workload arguments require --benchmark')
    benchmark = args.benchmark or 'operator_latency'
    if args.plot and args.benchmark:
        parser.error('--plot supports the default CI operator comparison')

    if args.plot and not args.output:
        print(
            "Warning: --plot requires --output to save comparison JSON first; "
            "no plots will be generated.",
            file=sys.stderr,
        )

    head_sha = get_commit_sha(ref=args.head)
    base_sha = get_commit_sha(ref=args.base)
    print(f"Comparing: {base_sha} (base) vs {head_sha} (head)")

    if args.benchmark:
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    elif args.benchmark_ops:
        op_names = args.benchmark_ops
    else:
        changed_files = get_changed_files(base=base_sha, head=head_sha)
        if not changed_files:
            print("No changed .py files found between commits.")
            return 0
        print(f"Changed files ({len(changed_files)}):")
        for f in changed_files:
            print(f"  {f}")

        op_names = find_affected_op_names(changed_files=changed_files)
        if not op_names:
            print("No affected ops found for the changed files.")
            return 0

    if not args.benchmark:
        print(f"\nOps to benchmark ({len(op_names)}):")
        for name in op_names:
            print(f"  {name}")

    def measure(ref):
        if args.benchmark:
            return run_benchmark(module='benchmarks.run', arguments=[benchmark, *arguments], ref=ref)
        return run_operator_benchmarks(op_names=op_names, shapes=SHAPE_CONFIGS, modes=['fwd', 'fwdbwd'], ref=ref)

    head_results, machine_info = measure(ref=head_sha)
    cooldown = int(os.environ.get('FLA_BENCH_COOLDOWN_SEC', '0'))
    if cooldown > 0:
        time.sleep(cooldown)
    base_results, _ = measure(ref=base_sha)

    if not base_results or not head_results:
        print("Warning: missing results from one or both commits.", file=sys.stderr)
        return 1

    changes = compare_results(base=base_results, head=head_results, threshold=args.threshold)
    print(format_comparison(changes=changes))
    regressions = [change for change in changes if change['regression']]
    speedups = [change for change in changes if change['speedup']]
    exit_code = int(bool(regressions))

    if args.output:
        comparison = {
            'threshold': args.threshold,
            'base_sha': base_sha,
            'head_sha': head_sha,
            'machine_info': machine_info,
            'base_results': base_results,
            'head_results': head_results,
            'regressions': regressions,
            'speedups': speedups,
            'has_regression': len(regressions) > 0,
            'has_speedup': len(speedups) > 0,
        }
        if not args.benchmark:
            # the PR-comment workflow reads the established operator artifact on the default branch.
            for key in ('base_results', 'head_results'):
                comparison[key] = [
                    {'op': result['name'], **result['parameters'], **result['metrics']}
                    for result in comparison[key]
                ]
            for key in ('regressions', 'speedups'):
                comparison[key] = [
                    {'op': change['name'], **change['parameters'], 'base_ms': change['base'],
                     'head_ms': change['head'], 'change_pct': change['change_pct']}
                    for change in comparison[key]
                ]
        else:
            comparison.update(benchmark=benchmark, changes=changes, threshold=args.threshold)
        output = get_result_path(path=Path(args.output), rank=int(os.environ.get('RANK', '0')))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
        print(f"\nFull results saved to {output}")

        if args.plot:
            from benchmarks.visualize import visualize_comparison
            plot_paths = visualize_comparison(
                output,
                save_dir=args.plot,
                threshold=args.threshold,
            )
            for p in plot_paths:
                print(f"  Plot saved: {p}")

    if args.no_fail_on_regression and regressions:
        print("\n  --no-fail-on-regression set, exiting with code 0 despite regressions.")
        return 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
