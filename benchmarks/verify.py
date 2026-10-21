# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run the selected correctness tests before operator timing and optional profiling."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from benchmarks.configs.operators import SHAPE_CONFIGS, get_op, list_ops
from benchmarks.utils.environment import synchronize_device
from benchmarks.utils.reporting import compare_results, format_comparison, format_machine_info, print_results
from benchmarks.utils.revisions import find_project_root, run_operator_benchmarks


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
    root = find_project_root()
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

    import torch
    from torch.profiler import ProfilerActivity, profile

    from benchmarks.utils.inputs import generate_inputs
    from fla.utils import device_name

    config = get_op(op_name)
    mod = importlib.import_module(config.import_path)
    op_fn = getattr(mod, config.func_name or config.name)

    shapes = config.default_shapes or SHAPE_CONFIGS
    shape = next(iter(shapes.values()))
    B, T, H, D = shape['B'], shape['T'], shape['H'], shape['D']
    extra = {k: v for k, v in shape.items() if k not in ('B', 'T', 'H', 'D')}
    inputs = generate_inputs(config=config, B=B, T=T, H=H, D=D, dtype=torch.bfloat16, device=device_name, **extra)

    do_bwd = 'fwdbwd' in modes and not config.skip_backward
    out = op_fn(**inputs, **config.extra_kwargs)
    out_tensor = out[0] if config.output_is_tuple else out
    do = torch.randn_like(out_tensor)

    def step():
        result = op_fn(**inputs, **config.extra_kwargs)
        t = result[0] if config.output_is_tuple else result
        if do_bwd:
            t.backward(do)

    # warm autotune before profiling.
    for _ in range(3):
        step()
    synchronize_device(device_name)

    out_dir = os.path.join(find_project_root(), 'profile', f'{op_name}-opt')
    os.makedirs(out_dir, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    sort_by = 'self_cpu_time_total'
    if device_name == 'cuda':
        activities.append(ProfilerActivity.CUDA)
        sort_by = 'self_cuda_time_total'
    with profile(activities=activities, record_shapes=True) as prof:
        step()
        synchronize_device(device_name)

    trace = os.path.join(out_dir, f'{op_name}_trace.json')
    prof.export_chrome_trace(trace)
    print(f"\n  Top ops for {op_name} @ {B}x{T}x{H}x{D} "
          f"({'fwd+bwd' if do_bwd else 'fwd'}):")
    print(prof.key_averages().table(sort_by=sort_by, row_limit=10))
    print(f"  Chrome trace: {trace}")


def main():
    parser = argparse.ArgumentParser(
        description='Correctness-gated benchmark driver for the optimization loop',
    )
    parser.add_argument('--op', help='Registered op name (see --list)')
    parser.add_argument(
        '--base', default=None,
        help='Git ref for the baseline column (e.g. "main"); uses an isolated revision worker.',
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

    if not args.op:
        parser.error('--op is required (use --list to see available ops)')

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
                f"configs/operators.py for '{args.op}'.",
                file=sys.stderr,
            )
            return 2
        if not passed:
            print("\n  GATE FAILED — correctness regressed. Not benchmarking. "
                  "Fix correctness before measuring speed.", file=sys.stderr)
            return 1
        print("\n  GATE PASSED.")

    results, machine_info = run_operator_benchmarks(op_names=[args.op], shapes=None, modes=args.modes)
    results = results or []
    machine_info = machine_info or {}
    print(f"\n  {format_machine_info(machine_info)} | "
          f"Triton {machine_info.get('triton_version', 'N/A')} | "
          f"{machine_info.get('git_label', 'unknown')}")

    baseline = None
    if args.base:
        baseline, _ = run_operator_benchmarks(op_names=[args.op], shapes=None, modes=args.modes, ref=args.base)
    if baseline is not None:
        print(format_comparison(changes=compare_results(base=baseline, head=results, threshold=5.0)))
    else:
        print_results(results=results)

    if args.profile:
        profile_op(args.op, args.modes)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
