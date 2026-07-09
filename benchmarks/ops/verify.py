# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Correctness-gated benchmark driver for the kernel optimization loop.

This ties the two halves of an optimization iteration into one reproducible command:
it runs the op's **frozen pytest** as a correctness gate, and only if the gate is green does it measure performance.
A speedup is never reported on a red gate.
See ``.agents/skills/fla-optimization-loop/SKILL.md`` for the discipline this driver supports.

Why a separate driver and not just ``run.py``?
``run.py`` measures latency but does not check correctness;
the pytest suite checks correctness (forward AND backward, under NaN memory poisoning) but does not measure latency.
During an optimization loop you need both, every iteration,
and you must never let a fast kernel that silently broke gradients look like a win.
This driver enforces that ordering.
It reuses ``registry.py`` (inputs) and ``run.py`` (timing);
the only correctness logic is "run the unmodified test file via pytest" —
the test is a black box this tool may select from but never edit.

Usage::

    # Gate (full pytest) + benchmark vs. main
    python -m benchmarks.ops.verify --op chunk_gla --base main

    # Fast signal: gate on a shape subset (pytest -k selection, test unchanged)
    python -m benchmarks.ops.verify --op chunk_gla --gate-k T15 --modes fwd

    # Gate + benchmark + torch profiler trace
    python -m benchmarks.ops.verify --op fused_attnres --profile

    # Point the gate at an explicit test file (when the derived path is wrong)
    python -m benchmarks.ops.verify --op chunk_gdn --test-file tests/ops/test_gdn.py

    # Skip the gate entirely (loud warning; speedup is then unverified)
    python -m benchmarks.ops.verify --op chunk_gla --no-gate

    # List registered ops
    python -m benchmarks.ops.verify --list

``--gate-k`` is a pytest ``-k`` *selection* for fast iteration — it narrows which parametrized cases run,
it does not modify the test.
The full file is the promotion gate; a subset is only a signal.
Promote a candidate only on a full green gate.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch

# Import sibling modules. Works both as a package (python -m benchmarks.ops.verify)
# and standalone, matching run.py's import strategy.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import SHAPE_CONFIGS, get_op, list_ops  # noqa: E402
from run import (  # noqa: E402
    _bench_at_ref,
    _find_project_root,
    _get_machine_info,
    benchmark_op,
    print_results_table,
)


def derive_test_file(op_name: str) -> str:
    """Derive the gate test path from the registered op.

    Priority: the op's explicit ``test_file`` field, else ``tests/ops/test_<last-segment-of-import_path>.py``.
    The returned path is repo-root-relative; the caller checks existence
    (some ops need an explicit override because the op directory name and test name differ,
    e.g. ``fla.ops.gated_delta_rule`` is tested by ``test_gdn.py``).
    """
    config = get_op(op_name)
    if config.test_file:
        return config.test_file
    leaf = config.import_path.rstrip('.').split('.')[-1]
    return os.path.join('tests', 'ops', f'test_{leaf}.py')


def run_gate(op_name: str, test_file: str | None, gate_k: str | None) -> bool:
    """Run the frozen pytest correctness gate for *op_name*.

    Returns True iff pytest exits 0.
    The test file is run unmodified; ``gate_k`` only narrows selection via pytest ``-k``.
    Raises FileNotFoundError if the resolved test file does not exist (caller should surface the override hint).
    """
    root = _find_project_root()
    rel = test_file or derive_test_file(op_name)
    abs_path = rel if os.path.isabs(rel) else os.path.join(root, rel)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(rel)

    cmd = [sys.executable, '-m', 'pytest', abs_path, '-q']
    if gate_k:
        cmd += ['-k', gate_k]
    scope = f" (-k {gate_k})" if gate_k else ' (full)'
    print(f"\n{'=' * 78}\n  Correctness gate: {rel}{scope}\n{'=' * 78}")
    result = subprocess.run(cmd, cwd=root)
    return result.returncode == 0


def profile_op(op_name: str, modes: list[str]) -> None:
    """Dump a torch.profiler trace for one fwd(+bwd) call into profile/<op>-opt/.

    A general (no-ncu) profiling entry point — the torch profiler the user already reaches for,
    wired to the same registry inputs.
    Detailed Nsight Compute collection lives in the fla-nvidia-performance skill.
    """
    import importlib

    from registry import generate_inputs
    from torch.profiler import ProfilerActivity, profile

    config = get_op(op_name)
    mod = importlib.import_module(config.import_path)
    op_fn = getattr(mod, config.func_name or config.name)

    shapes = config.default_shapes or SHAPE_CONFIGS
    shape = next(iter(shapes.values()))
    B, T, H, D = shape['B'], shape['T'], shape['H'], shape['D']
    extra = {k: v for k, v in shape.items() if k not in ('B', 'T', 'H', 'D')}
    inputs = generate_inputs(config, B, T, H, D, dtype=torch.bfloat16, device='cuda', **extra)

    do_bwd = 'fwdbwd' in modes and not config.skip_backward
    out = op_fn(**inputs, **config.extra_kwargs)
    out_tensor = out[0] if config.output_is_tuple else out
    do = torch.randn_like(out_tensor)

    def step():
        result = op_fn(**inputs, **config.extra_kwargs)
        t = result[0] if config.output_is_tuple else result
        if do_bwd:
            t.backward(do)

    for _ in range(3):  # warm autotune before profiling
        step()
    torch.cuda.synchronize()

    out_dir = os.path.join(_find_project_root(), 'profile', f'{op_name}-opt')
    os.makedirs(out_dir, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        step()
        torch.cuda.synchronize()

    trace = os.path.join(out_dir, f'{op_name}_trace.json')
    prof.export_chrome_trace(trace)
    print(f"\n  Top CUDA ops for {op_name} @ {B}x{T}x{H}x{D} "
          f"({'fwd+bwd' if do_bwd else 'fwd'}):")
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
    print(f"  Chrome trace: {trace}")


def main():
    parser = argparse.ArgumentParser(
        description='Correctness-gated benchmark driver for the optimization loop',
    )
    parser.add_argument('--op', help='Registered op name (see --list)')
    parser.add_argument(
        '--kind', choices=['op', 'layer'], default='op',
        help='Target kind. Only "op" is implemented; "layer" is a reserved '
             'extension point. Default: op.',
    )
    parser.add_argument(
        '--base', default=None,
        help='Git ref for the baseline column (e.g. "main"); reuses run.py worktree compare.',
    )
    parser.add_argument(
        '--modes', nargs='+', default=['fwd', 'fwdbwd'], choices=['fwd', 'fwdbwd'],
        help='Benchmark modes. Default: fwd fwdbwd.',
    )
    parser.add_argument(
        '--gate-k', default=None,
        help='pytest -k selection for a fast signal gate (test file is NOT modified). '
             'Promote only on a full (no -k) green gate.',
    )
    parser.add_argument(
        '--test-file', default=None,
        help='Explicit gate test path (repo-root relative). Overrides the derived path.',
    )
    parser.add_argument(
        '--no-gate', action='store_true',
        help='Skip the correctness gate. Speedup is then UNVERIFIED — never promote on this.',
    )
    parser.add_argument('--profile', action='store_true', help='Dump a torch.profiler trace.')
    parser.add_argument('--list', action='store_true', help='List registered ops and exit.')
    args = parser.parse_args()

    if args.list:
        ops = list_ops()
        print(f"Registered ops ({len(ops)}):")
        for name in ops:
            cfg = get_op(name)
            print(f"  {name:30s}  [{cfg.category}]  {cfg.import_path}")
        return 0

    if args.kind == 'layer':
        raise NotImplementedError(
            "verify.py currently covers ops only; the registry has no layer targets yet. "
            "Layer support is a reserved extension point — register layer configs in "
            "registry.py and add a layer input/reference path before using --kind layer."
        )

    if not args.op:
        parser.error('--op is required (use --list to see available ops)')

    # 1. Correctness gate (the frozen test, run unmodified). Never benchmark on a red gate.
    if args.no_gate:
        print("\n  *** WARNING: --no-gate set. Correctness is NOT verified; any "
              "speedup below is meaningless for promotion. ***")
    else:
        try:
            passed = run_gate(args.op, args.test_file, args.gate_k)
        except FileNotFoundError as e:
            print(
                f"\n  Gate test not found: {e}\n"
                f"  The op directory name may differ from its test name. Pass "
                f"--test-file tests/ops/test_<name>.py, or set test_file=... in "
                f"registry.py for '{args.op}'.",
                file=sys.stderr,
            )
            return 2
        if not passed:
            print("\n  GATE FAILED — correctness regressed. Not benchmarking. "
                  "Fix correctness before measuring speed.", file=sys.stderr)
            return 1
        print("\n  GATE PASSED.")

    # 2. Performance measurement (reuses run.py). Baseline compare via git worktree.
    machine_info = _get_machine_info()
    print(f"\n  Machine: {machine_info.get('gpu_name', 'N/A')} | "
          f"CUDA {machine_info.get('cuda_version', 'N/A')} | "
          f"PyTorch {machine_info.get('pytorch_version', 'N/A')} | "
          f"Triton {machine_info.get('triton_version', 'N/A')} | "
          f"{machine_info.get('git_label', 'unknown')}")

    results = benchmark_op(args.op, SHAPE_CONFIGS, modes=args.modes)
    baseline, baseline_info = (None, None)
    if args.base:
        baseline, baseline_info = _bench_at_ref(args.base, [args.op], SHAPE_CONFIGS, args.modes)
    print_results_table(results, machine_info, baseline=baseline, baseline_info=baseline_info)

    # 3. Optional profiling.
    if args.profile:
        profile_op(args.op, args.modes)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
