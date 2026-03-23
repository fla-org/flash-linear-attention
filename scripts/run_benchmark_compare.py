#!/usr/bin/env python
"""
Compare benchmark results between two git commits using the unified runner.

Uses a "temp copy" strategy: copies the benchmark runner to a temp directory
before any git checkout, then runs benchmarks from the temp copy at both
commits.  The runner imports fla.ops.* via importlib, so after `pip install -e .`
at each commit, Python resolves those imports to the commit-specific kernel code.

Usage:
    # Auto-detect ops from diff (defaults: --base main --head HEAD)
    python scripts/run_benchmark_compare.py

    # Specify ops directly
    python scripts/run_benchmark_compare.py --benchmark-ops chunk_gated_delta_rule chunk_kda

    # Compare against a different base
    python scripts/run_benchmark_compare.py --base feature-branch --head HEAD

    # Custom threshold
    python scripts/run_benchmark_compare.py --threshold 10.0
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_OP_DIR_RE = re.compile(r'^fla/ops/([^/]+)/')

TRITON_CACHE_DIR = Path(os.environ.get(
    'TRITON_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.triton')
))

# Files to copy to the temp directory for cross-commit benchmarking
RUNNER_FILES = [
    'benchmarks/ops/registry.py',
    'benchmarks/ops/run.py',
]


def run_cmd(cmd, **kwargs):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), **kwargs,
    )
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_commit_sha(ref: str) -> str:
    return run_cmd(["git", "rev-parse", "--short", ref])


def get_changed_files(base: str, head: str) -> list[str]:
    out = run_cmd(["git", "diff", "--name-only", base, head, "--", "*.py"])
    return [f for f in out.split('\n') if f]


def find_affected_op_names(changed_files: list[str]) -> list[str]:
    """Map changed file paths to registered op names.

    Parses fla/ops/<dir>/ from each path and looks up the registry.
    Changes in fla/ops/common/ or fla/ops/utils/ affect ALL ops.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'benchmarks' / 'ops'))
        from registry import _REGISTRY
    except ImportError:
        print("Warning: could not import registry", file=sys.stderr)
        return []

    # Reverse map: dir_name -> [op_names]
    import_map: dict[str, list[str]] = {}
    for name, cfg in _REGISTRY.items():
        parts = cfg.import_path.split('.')
        if len(parts) >= 3 and parts[0] == 'fla' and parts[1] == 'ops':
            import_map.setdefault(parts[2], []).append(name)

    affected_dirs: set[str] = set()
    common_changed = False
    for fpath in changed_files:
        if not fpath.endswith('.py'):
            continue
        if fpath.startswith('fla/ops/common/') or fpath.startswith('fla/ops/utils/'):
            common_changed = True
            continue
        m = _OP_DIR_RE.match(fpath)
        if m:
            affected_dirs.add(m.group(1))

    if common_changed:
        return sorted(_REGISTRY.keys())

    op_names: set[str] = set()
    for d in affected_dirs:
        if d in import_map:
            op_names.update(import_map[d])
    return sorted(op_names)


def run_unified_benchmark(
    runner_path: str,
    op_names: list[str],
    output_json: str,
) -> bool:
    """Run the unified runner from a temp directory."""
    cmd = [
        sys.executable, runner_path,
        '--op', *op_names,
        '--json', output_json,
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"  Warning: benchmark failed: {result.stderr[:500]}", file=sys.stderr)
        return False
    return True


def clear_triton_cache():
    """Clear triton autotune cache to avoid cross-commit contamination."""
    cache_dir = TRITON_CACHE_DIR / "cache"
    if cache_dir.exists():
        print(f"  Clearing triton cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


def checkout_and_install(ref: str, clear_cache: bool = True):
    """Checkout a ref and reinstall the package."""
    print(f"\n  Checking out {ref}...")
    run_cmd(["git", "checkout", ref])
    print("  Installing package...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )
    if clear_cache:
        clear_triton_cache()


def copy_runner_to_temp() -> str:
    """Copy benchmark runner files to a temp directory for cross-commit use."""
    tmpdir = tempfile.mkdtemp(prefix="fla_bench_")
    for relpath in RUNNER_FILES:
        src = PROJECT_ROOT / relpath
        if src.exists():
            dst = os.path.join(tmpdir, os.path.basename(relpath))
            shutil.copy2(str(src), dst)
        else:
            print(f"  Warning: {relpath} not found, skipping copy", file=sys.stderr)
    return tmpdir


def print_comparison(base_results: list[dict], head_results: list[dict],
                     base_sha: str, head_sha: str, threshold: float,
                     machine_info: dict | None = None):
    """Print a comparison table with speedup ratios and detect regressions."""

    def make_key(r):
        return (r['op'], r['mode'], r['B'], r['T'], r['H'], r['D'])

    head_map = {make_key(r): r for r in head_results}
    base_map = {make_key(r): r for r in base_results}

    all_keys = sorted(set(list(head_map.keys()) + list(base_map.keys())))

    # Header
    print(f"\n{'=' * 105}")
    if machine_info:
        gpu = machine_info.get('gpu_name', 'N/A')
        cuda = machine_info.get('cuda_version', 'N/A')
        pytorch = machine_info.get('pytorch_version', 'N/A')
        print(f"  Machine: {gpu} | CUDA {cuda} | PyTorch {pytorch}")
    print(f"  Old (base): {base_sha} | New (head): {head_sha}")
    print(f"{'=' * 105}")
    print(f"  {'op':<28s} {'mode':<7s} {'shape':<22s} "
          f"{'old(ms)':>9s} {'new(ms)':>9s} {'speedup':>8s} {'change':>8s}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 22} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8}")

    regressions = []
    for key in all_keys:
        op, mode, B, T, H, D = key
        base_r = base_map.get(key)
        head_r = head_map.get(key)

        shape_str = f"B={B} T={T} H={H} D={D}"

        if base_r and head_r:
            base_ms = base_r['median_ms']
            head_ms = head_r['median_ms']
            change_pct = (head_ms - base_ms) / base_ms * 100
            speedup = base_ms / head_ms if head_ms > 0 else float('inf')
            sign = '+' if change_pct > 0 else ''
            marker = ''
            if change_pct > threshold:
                marker = ' <<< REGRESSION'
            elif change_pct < -threshold:
                marker = ' SPEEDUP'
            print(f"  {op:<28s} {mode:<7s} {shape_str:<22s} "
                  f"{base_ms:>9.3f} {head_ms:>9.3f} {speedup:>7.2f}x "
                  f"{sign}{change_pct:>6.1f}%{marker}")
            if change_pct > threshold:
                regressions.append((key, base_ms, head_ms, change_pct))
        elif head_r:
            print(f"  {op:<28s} {mode:<7s} {shape_str:<22s} "
                  f"{'N/A':>9s} {head_r['median_ms']:>9.3f} {'':>8s} {'new':>8s}")
        elif base_r:
            print(f"  {op:<28s} {mode:<7s} {shape_str:<22s} "
                  f"{base_r['median_ms']:>9.3f} {'N/A':>9s} {'':>8s} {'removed':>8s}")

    print(f"{'=' * 105}")

    if regressions:
        print(f"\n  WARNING: {len(regressions)} regression(s) detected (>{threshold}% slower):")
        for key, base_ms, head_ms, pct in regressions:
            print(f"    {key}: {base_ms:.3f} -> {head_ms:.3f} ms (+{pct:.1f}%)")
        return 1
    else:
        print(f"\n  No regressions detected (threshold: {threshold}%).")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Compare benchmarks between two commits")
    parser.add_argument("--base", default="main", help="Base commit/branch/tag (default: main)")
    parser.add_argument("--head", default="HEAD", help="Head commit/branch/tag (default: HEAD)")
    parser.add_argument("--benchmark-ops", nargs="*",
                        help="Op names to benchmark. If omitted, auto-detect from diff.")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold in percent (default: 5.0)")
    parser.add_argument("--output", help="Save comparison results to JSON file")
    args = parser.parse_args()

    # Resolve commit SHAs
    head_sha = get_commit_sha(args.head)
    base_sha = get_commit_sha(args.base)
    print(f"Comparing: {base_sha} (base) vs {head_sha} (head)")

    # Determine which ops to benchmark
    if args.benchmark_ops:
        op_names = args.benchmark_ops
    else:
        changed_files = get_changed_files(args.base, args.head)
        if not changed_files:
            print("No changed .py files found between commits.")
            sys.exit(0)
        print(f"Changed files ({len(changed_files)}):")
        for f in changed_files:
            print(f"  {f}")

        op_names = find_affected_op_names(changed_files)
        if not op_names:
            print("No affected ops found for the changed files.")
            sys.exit(0)

    print(f"\nOps to benchmark ({len(op_names)}):")
    for name in op_names:
        print(f"  {name}")

    # Get current branch/ref to restore later
    current_ref = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if current_ref == "HEAD":
        current_ref = run_cmd(["git", "rev-parse", "HEAD"])

    tmpdir = tempfile.mkdtemp(prefix="fla_bench_")
    runner_tmpdir = copy_runner_to_temp()
    runner_path = os.path.join(runner_tmpdir, 'run.py')
    print(f"Runner copied to: {runner_tmpdir}")

    machine_info = None

    try:
        # Step 1: HEAD
        print(f"\n{'=' * 60}")
        print(f"  Running benchmarks at HEAD ({head_sha})")
        print(f"{'=' * 60}")
        checkout_and_install(args.head)

        head_json = os.path.join(tmpdir, "head.json")
        run_unified_benchmark(runner_path, op_names, head_json)

        head_results = []
        if os.path.exists(head_json):
            with open(head_json) as f:
                data = json.load(f)
                head_results = data.get('results', [])
                machine_info = data.get('machine_info')

        # Step 2: BASE
        print(f"\n{'=' * 60}")
        print(f"  Running benchmarks at BASE ({base_sha})")
        print(f"{'=' * 60}")
        checkout_and_install(args.base)

        base_json = os.path.join(tmpdir, "base.json")
        run_unified_benchmark(runner_path, op_names, base_json)

        base_results = []
        if os.path.exists(base_json):
            with open(base_json) as f:
                data = json.load(f)
                base_results = data.get('results', [])

        # Step 3: Restore original ref
        print(f"\n  Restoring {current_ref}...")
        checkout_and_install(current_ref, clear_cache=False)

        # Step 4: Print comparison
        if not base_results or not head_results:
            print("Warning: missing results from one or both commits.", file=sys.stderr)
            sys.exit(1)

        exit_code = print_comparison(
            base_results, head_results,
            base_sha, head_sha, args.threshold,
            machine_info=machine_info,
        )

        # Optionally save full results
        if args.output:
            comparison = {
                'base_sha': base_sha,
                'head_sha': head_sha,
                'machine_info': machine_info,
                'base_results': base_results,
                'head_results': head_results,
            }
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nFull results saved to {args.output}")

        sys.exit(exit_code)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(runner_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
