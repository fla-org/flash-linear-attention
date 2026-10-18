# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare affected operator workloads for CI and local regression checks."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from benchmarks.ops.utils.discovery import find_affected_op_names, get_changed_files
from benchmarks.ops.utils.registry import SHAPE_CONFIGS
from benchmarks.ops.utils.runner import run_benchmarks
from benchmarks.utils.reporting import print_comparison
from benchmarks.utils.revisions import get_commit_sha


def clear_triton_cache():
    """Clear triton autotune cache to avoid cross-commit contamination."""
    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton")) / "cache"
    if cache_dir.exists():
        print(f"  Clearing triton cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmarks between two commits")
    parser.add_argument("--base", default="main", help="Base commit/branch/tag (default: main)")
    parser.add_argument("--head", default="HEAD", help="Head commit/branch/tag (default: HEAD)")
    parser.add_argument("--benchmark-ops", nargs="*", help="Op names to benchmark. If omitted, auto-detect from diff.")
    parser.add_argument("--threshold", type=float, default=5.0, help="Regression threshold in percent (default: 5.0)")
    parser.add_argument("--output", help="Save comparison results to JSON file")
    parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Do not exit with non-zero code on performance regression",
    )
    parser.add_argument("--plot", default=None, help="Save comparison visualization plots to this directory")
    args = parser.parse_args()

    if args.plot and not args.output:
        print(
            "Warning: --plot requires --output to save comparison JSON first; "
            "no plots will be generated.",
            file=sys.stderr,
        )

    head_sha = get_commit_sha(ref=args.head)
    base_sha = get_commit_sha(ref=args.base)
    print(f"Comparing: {base_sha} (base) vs {head_sha} (head)")

    if args.benchmark_ops:
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

    print(f"\nOps to benchmark ({len(op_names)}):")
    for name in op_names:
        print(f"  {name}")

    clear_triton_cache()
    head_results, machine_info = run_benchmarks(
        op_names=op_names,
        shapes=SHAPE_CONFIGS,
        modes=['fwd', 'fwdbwd'],
        ref=head_sha,
    )
    cooldown = int(os.environ.get('FLA_BENCH_COOLDOWN_SEC', '0'))
    if cooldown > 0:
        time.sleep(cooldown)
    clear_triton_cache()
    base_results, _ = run_benchmarks(op_names=op_names, shapes=SHAPE_CONFIGS, modes=['fwd', 'fwdbwd'], ref=base_sha)

    if not base_results or not head_results:
        print("Warning: missing results from one or both commits.", file=sys.stderr)
        return 1

    exit_code, regressions, speedups = print_comparison(
        base_results=base_results,
        head_results=head_results,
        base_sha=base_sha,
        head_sha=head_sha,
        threshold=args.threshold,
        machine_info=machine_info,
    )

    if args.output:
        comparison = {
            'base_sha': base_sha,
            'head_sha': head_sha,
            'machine_info': machine_info,
            'base_results': base_results,
            'head_results': head_results,
            'regressions': [
                {
                    'op': key[0], 'mode': key[1], 'B': key[2], 'T': key[3],
                    'H': key[4], 'D': key[5],
                    'base_ms': base_ms, 'head_ms': head_ms, 'change_pct': change_pct
                }
                for key, base_ms, head_ms, change_pct in regressions
            ],
            'speedups': [
                {
                    'op': key[0], 'mode': key[1], 'B': key[2], 'T': key[3],
                    'H': key[4], 'D': key[5],
                    'base_ms': base_ms, 'head_ms': head_ms, 'change_pct': change_pct
                }
                for key, base_ms, head_ms, change_pct in speedups
            ],
            'has_regression': len(regressions) > 0,
            'has_speedup': len(speedups) > 0,
        }
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nFull results saved to {args.output}")

        if args.plot:
            from benchmarks.visualize import visualize_comparison
            plot_paths = visualize_comparison(
                args.output,
                save_dir=args.plot,
                threshold=args.threshold,
            )
            for p in plot_paths:
                print(f"  Plot saved: {p}")

    if args.no_fail_on_regression:
        print("\n  --no-fail-on-regression set, exiting with code 0 despite regressions.")
        return 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
