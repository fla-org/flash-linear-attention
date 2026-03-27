# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Unified CLI benchmark runner for all registered ops.

Usage::

    # Benchmark one op (uses all default shape configs)
    python -m benchmarks.ops.run --op chunk_gla

    # Multiple ops
    python -m benchmarks.ops.run --op chunk_gla chunk_kda

    # All registered ops
    python -m benchmarks.ops.run --op all

    # Forward only
    python -m benchmarks.ops.run --op chunk_gla --modes fwd

    # Compare against main (auto git worktree, no stash needed)
    python -m benchmarks.ops.run --op chunk_gla --base main

    # Compare against any branch/tag/commit
    python -m benchmarks.ops.run --op chunk_gla --base HEAD~3

    # Save results to JSON
    python -m benchmarks.ops.run --op chunk_gla --json results.json

    # Custom shape (overrides default SHAPE_CONFIGS)
    python -m benchmarks.ops.run --op chunk_gla \\
        --custom-shapes '{"test": {"B": 2, "T": 4096, "H": 32, "D": 128}}'

    # List all registered ops
    python -m benchmarks.ops.run --list

Results are cached per-branch in ``.bench_cache/<branch>.json``.  When run on
a feature branch, the ``main`` branch's cached results are loaded as the
baseline automatically.  Column headers display ``branch[commit](ms)``
(branch names longer than 8 characters are truncated with ``...``).

Sample output::

    ==========================================================================================================================
      Machine: NVIDIA GB200 | CUDA 12.9 | PyTorch 2.9.0+cu129.msh
    ==========================================================================================================================
      op                           mode       B      T    H    D  fuse-gdn...[51141dbc](ms)        main[7978c0bd](ms)  speedup
      ---------------------------- ------- ---- ------ ---- ----  -------------------------  -------------------------  -------
      chunk_gated_delta_rule       fwd        1   8192   96  128                      1.268                     1.506    1.19x
      chunk_gated_delta_rule       fwd        2  16384   16  128                      0.988                     1.152    1.17x
      chunk_gated_delta_rule       fwd        4   2048   16  128                      0.518                     0.582    1.12x
      chunk_gated_delta_rule       fwd        4   4096   64  128                      1.600                     1.934    1.21x
      chunk_gated_delta_rule       fwd        8    512    4   64                      0.524                     0.570    1.09x
      chunk_gated_delta_rule       fwd        8   2048   32  256                      1.887                     2.034    1.08x

      chunk_gated_delta_rule       fwdbwd     1   8192   96  128                      4.818                     5.065    1.05x
      chunk_gated_delta_rule       fwdbwd     2  16384   16  128                      4.001                     4.168    1.04x
      chunk_gated_delta_rule       fwdbwd     4   2048   16  128                      1.648                     1.682    1.02x
      chunk_gated_delta_rule       fwdbwd     4   4096   64  128                      6.062                     6.398    1.06x
      chunk_gated_delta_rule       fwdbwd     8    512    4   64                      1.609                     1.674    1.04x
      chunk_gated_delta_rule       fwdbwd     8   2048   32  256                      8.838                     8.963    1.01x
    ==========================================================================================================================

Registering a new op
====================
All op definitions live in ``registry.py``.  To add a new op:

1. Pick shape helpers for each input tensor (defined in registry.py)::

       shape_BTHD  -> (B, T, H, D)     most q/k/v/g tensors
       shape_BTH   -> (B, T, H)         per-head scalars (gates, beta)
       shape_BTD   -> (B, T, H*D)       flattened hidden dim (HGRN)
       shape_HD    -> (H, D)            per-head vectors (RWKV u)
       shape_H     -> (H,)              per-head scalars

2. Pick transforms to map randn into the right value range::

       logsigmoid        -> negative values (log-space gates)
       sigmoid_transform -> (0, 1) range (beta)
       logsigmoid_clamp  -> logsigmoid clamped to >= -5

3. Call register_op() in registry.py::

       register_op(OpConfig(
           name='chunk_my_op',
           import_path='fla.ops.my_op',
           inputs={
               'q': TensorSpec(shape_BTHD),
               'k': TensorSpec(shape_BTHD),
               'v': TensorSpec(shape_BTHD),
               'g': TensorSpec(shape_BTH, transform=logsigmoid),
           },
           extra_kwargs={'use_some_flag': True},
           category='my_group',
       ))

4. Verify:  ``python -m benchmarks.ops.run --list``

Special cases:

- Non-standard param init:  use ``post_init`` callback (see _rwkv7_post_init)
- Op only supports certain D: set ``dim_constraints={'D': [64, 128]}``
- Op has no backward:        set ``skip_backward=True``
- Output is a plain tensor:  set ``output_is_tuple=False``

Benchmark methodology
=====================
1. **Warmup**: For each (op, shape), run fwd+bwd 5 times to trigger all
   triton autotuning.  All shapes are warmed up before any timing begins.
2. **Timing**: ``triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])``
   gives median, p20, p80 in milliseconds.
3. Input tensors (including gate transforms like logsigmoid) are prepared
   **before** timing — only the op call itself is measured.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile

import torch

# Import registry — works both as a package (python -m benchmarks.ops.run)
# and standalone (python /tmp/fla_bench_xxx/run.py) for cross-commit use.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import (  # noqa: E402
    SHAPE_CONFIGS,
    OpConfig,
    generate_inputs,
    get_op,
    list_ops,
)

logger = logging.getLogger(__name__)


def _import_op(config: OpConfig):
    """Dynamically import the op function from the installed fla package."""
    mod = importlib.import_module(config.import_path)
    attr = config.func_name or config.name
    fn = getattr(mod, attr, None)
    if fn is None:
        raise ImportError(
            f"Cannot find '{attr}' in module '{config.import_path}'. "
            f"Available: {[x for x in dir(mod) if not x.startswith('_')]}"
        )
    return fn


def _get_git_label() -> str:
    """Return 'branch[short_sha]', e.g. 'main[abc1234]'."""
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return f"{branch}[{sha}]"
    except Exception:
        return 'unknown'


def _get_machine_info() -> dict:
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda or 'N/A',
        'git_label': _get_git_label(),
    }
    try:
        import triton
        info['triton_version'] = triton.__version__
    except Exception:
        info['triton_version'] = 'N/A'

    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
    else:
        info['gpu_name'] = 'N/A'
        info['gpu_count'] = 0
        info['gpu_memory_gb'] = 0
    return info


def _warmup_autotune(fn, n=5):
    """Run *fn* multiple times so triton autotuning is fully cached."""
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def benchmark_op(
    op_name: str,
    shapes: dict[str, dict[str, int]],
    modes: list[str] | None = None,
) -> list[dict]:
    """Benchmark a single op across all *shapes* and *modes*.

    Returns a list of result dicts (one per shape x mode).
    """
    import triton

    if modes is None:
        modes = ['fwd', 'fwdbwd']

    config = get_op(op_name)
    op_fn = _import_op(config)

    if config.skip_backward and 'fwdbwd' in modes:
        modes = [m for m in modes if m != 'fwdbwd']

    # Filter shapes by dim_constraints
    valid_shapes = {}
    for shape_name, shape_dict in shapes.items():
        if config.dim_constraints:
            skip = False
            for dim_name, allowed in config.dim_constraints.items():
                if shape_dict.get(dim_name) not in allowed:
                    logger.info(
                        f"Skipping {op_name} @ {shape_name}: "
                        f"{dim_name}={shape_dict.get(dim_name)} not in {allowed}"
                    )
                    skip = True
                    break
            if skip:
                continue
        valid_shapes[shape_name] = shape_dict

    if not valid_shapes:
        logger.warning(f"No compatible shapes for {op_name}, skipping.")
        return []

    device = 'cuda'
    dtype = torch.bfloat16

    # Phase 1: warmup ALL shapes before timing ANY
    print(f"\n  [{op_name}] Warming up {len(valid_shapes)} shape(s)...")
    failed_shapes = set()
    for shape_name, shape_dict in valid_shapes.items():
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        try:
            inputs = generate_inputs(config, B, T, H, D, dtype=dtype, device=device)
            out = op_fn(**inputs, **config.extra_kwargs)
            out_tensor = out[0] if config.output_is_tuple else out
            do = torch.randn_like(out_tensor)

            def _fwdbwd_fn(inputs=inputs, do=do):
                result = op_fn(**inputs, **config.extra_kwargs)
                t = result[0] if config.output_is_tuple else result
                t.backward(do)

            _warmup_autotune(_fwdbwd_fn)
        except Exception as e:
            logger.warning(f"Warmup failed for {op_name} @ {shape_name}: {e}")
            failed_shapes.add(shape_name)

    for name in failed_shapes:
        del valid_shapes[name]
    print(f"  [{op_name}] Warmup done.")

    # Phase 2: timing
    results = []
    for shape_name, shape_dict in list(valid_shapes.items()):
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        try:
            inputs = generate_inputs(config, B, T, H, D, dtype=dtype, device=device)
        except Exception as e:
            logger.warning(f"Input generation failed for {op_name} @ {shape_name}: {e}")
            continue

        out = op_fn(**inputs, **config.extra_kwargs)
        out_tensor = out[0] if config.output_is_tuple else out
        do = torch.randn_like(out_tensor)

        for mode in modes:
            if mode == 'fwd':
                def fn(inputs=inputs):
                    return op_fn(**inputs, **config.extra_kwargs)
            else:
                def fn(inputs=inputs, do=do):
                    result = op_fn(**inputs, **config.extra_kwargs)
                    t = result[0] if config.output_is_tuple else result
                    t.backward(do)

            try:
                ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
            except Exception as e:
                logger.warning(f"Bench failed for {op_name} {mode} @ {shape_name}: {e}")
                continue

            results.append({
                'op': op_name,
                'mode': mode,
                'B': B, 'T': T, 'H': H, 'D': D,
                'median_ms': ms[0],
                'p20_ms': ms[1],
                'p80_ms': ms[2],
            })

    return results


def _make_result_key(r):
    return (r['op'], r['mode'], r['B'], r['T'], r['H'], r['D'])


def _truncate_branch(name: str, max_len: int = 8) -> str:
    """Truncate a branch name to *max_len* chars, adding ``...`` if needed."""
    if len(name) > max_len:
        return name[:max_len] + '...'
    return name


def _make_col_headers(old_git: str, new_git: str) -> tuple[str, str]:
    """Build equal-width column headers like ``main    [abc1234](ms)``.

    Both branch names are truncated to 8 chars (with ``...``), then
    right-padded to the same width so the ``[commit](ms)`` parts align.
    """
    def _branch(git_label):
        return git_label.split('[')[0] if '[' in git_label else git_label

    def _suffix(git_label):
        if '[' in git_label:
            return '[' + git_label.split('[', 1)[1] + '(ms)'
        return '(ms)'

    old_br = _truncate_branch(_branch(old_git))
    new_br = _truncate_branch(_branch(new_git))
    br_w = max(len(old_br), len(new_br))
    return (f"{old_br:>{br_w}s}{_suffix(old_git)}",
            f"{new_br:>{br_w}s}{_suffix(new_git)}")


def print_results_table(results: list[dict], machine_info: dict | None = None,
                        baseline: list[dict] | None = None,
                        baseline_info: dict | None = None):
    """Print old / new / speedup comparison table.

    Column order: ``mode  B  T  H  D  op  old(ms)  new(ms)  speedup``.
    The mode column (``fwd`` / ``fwdbwd``) is shown when it changes.
    A dash separator (not covering the mode column) and column header
    are repeated before each shape group.
    """
    if not results:
        print("\n  No results to display.")
        return

    has_baseline = baseline is not None and len(baseline) > 0
    old_map = {_make_result_key(r): r for r in baseline} if has_baseline else {}

    new_git = machine_info.get('git_label', 'new') if machine_info else 'new'

    # mode_w = 2 (indent) + 7 (mode field) + 1 (space) = 10 chars before B column
    mode_pad = ' ' * 10

    if has_baseline:
        old_git = baseline_info.get('git_label', 'main') if baseline_info else 'main'
        old_hdr, new_hdr = _make_col_headers(old_git, new_git)
        col_w = max(len(old_hdr), len(new_hdr), 10)
        inner_w = 4 + 1 + 6 + 1 + 4 + 1 + 4 + 2 + 28 + 2 + col_w + 1 + col_w + 1 + 8
        inner_hdr = (f"{'B':>4s} {'T':>6s} {'H':>4s} {'D':>4s}  {'op':<28s}"
                     f"  {old_hdr:>{col_w}s} {new_hdr:>{col_w}s} {'speedup':>8s}")
    else:
        new_hdr = _truncate_branch(new_git.split('[')[0]) if '[' in new_git else new_git
        suffix = '[' + new_git.split('[', 1)[1] + '(ms)' if '[' in new_git else '(ms)'
        new_hdr = new_hdr + suffix
        col_w = max(len(new_hdr), 10)
        inner_w = 4 + 1 + 6 + 1 + 4 + 1 + 4 + 2 + 28 + 2 + col_w
        inner_hdr = (f"{'B':>4s} {'T':>6s} {'H':>4s} {'D':>4s}  {'op':<28s}"
                     f"  {new_hdr:>{col_w}s}")

    width = 10 + inner_w
    sep = '=' * width
    dash_line = mode_pad + '-' * inner_w

    print(f"\n{sep}")
    if machine_info:
        gpu = machine_info.get('gpu_name', 'N/A')
        cuda = machine_info.get('cuda_version', 'N/A')
        pytorch = machine_info.get('pytorch_version', 'N/A')
        print(f"  Machine: {gpu} | CUDA {cuda} | PyTorch {pytorch}")

    prev_shape = None
    prev_mode = None
    for r in results:
        cur_shape = (r['B'], r['T'], r['H'], r['D'])
        cur_mode = r['mode']

        # Show mode label + column header when mode changes; just a dash line between shapes
        if cur_mode != prev_mode:
            print(sep)
            print(f"  {cur_mode:<7s} {inner_hdr}")
            print(dash_line)
        elif cur_shape != prev_shape:
            print(dash_line)

        # Show B/T/H/D on first row of each shape group
        if cur_mode != prev_mode or cur_shape != prev_shape:
            shape_str = f"{r['B']:>4d} {r['T']:>6d} {r['H']:>4d} {r['D']:>4d}"
        else:
            shape_str = f"{'':>4s} {'':>6s} {'':>4s} {'':>4s}"

        prev_shape = cur_shape
        prev_mode = cur_mode

        new_ms = r['median_ms']
        prefix = f"{mode_pad}{shape_str}  {r['op']:<28s}"
        if has_baseline:
            old_r = old_map.get(_make_result_key(r))
            if old_r:
                old_ms = old_r['median_ms']
                speedup = old_ms / new_ms if new_ms > 0 else float('inf')
                print(f"{prefix}  {old_ms:>{col_w}.3f} {new_ms:>{col_w}.3f} {speedup:>7.2f}x")
            else:
                print(f"{prefix}  {'-':>{col_w}s} {new_ms:>{col_w}.3f} {'-':>8s}")
        else:
            print(f"{prefix}  {new_ms:>{col_w}.3f}")

    print(sep)


def _find_project_root() -> str:
    """Walk up from this file to find the git root."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != '/':
        if os.path.isdir(os.path.join(d, '.git')):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def _bench_at_ref(ref, op_names, shape_configs, modes):
    """Run benchmarks at a git ref using a temporary worktree.

    Returns (results_list, machine_info_dict) or (None, None) on failure.
    Does NOT touch the current working tree.
    """
    project_root = _find_project_root()
    tmpdir = tempfile.mkdtemp(prefix=f'fla_bench_{ref}_')
    worktree_dir = os.path.join(tmpdir, 'worktree')

    # Copy runner files to temp (constant across branches)
    runner_dir = os.path.join(tmpdir, 'runner')
    os.makedirs(runner_dir)
    for fname in ('run.py', 'registry.py'):
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(runner_dir, fname))

    try:
        print(f"\n  Benchmarking at '{ref}' (via git worktree)...")
        subprocess.run(
            ['git', 'worktree', 'add', worktree_dir, ref],
            cwd=project_root, capture_output=True, check=True,
        )
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.', '-q'],
            cwd=worktree_dir, capture_output=True,
        )

        runner = os.path.join(runner_dir, 'run.py')
        out_json = os.path.join(tmpdir, 'results.json')
        cmd = [sys.executable, runner, '--op', *op_names,
               '--custom-shapes', json.dumps(shape_configs),
               '--modes', *modes, '--json', out_json]
        subprocess.run(cmd, cwd=worktree_dir)

        if os.path.exists(out_json):
            with open(out_json) as f:
                data = json.load(f)
            return data.get('results', []), data.get('machine_info')
        return None, None
    except Exception as e:
        logger.warning(f"Failed to benchmark at '{ref}': {e}")
        return None, None
    finally:
        subprocess.run(
            ['git', 'worktree', 'remove', '--force', worktree_dir],
            cwd=project_root, capture_output=True,
        )
        # Reinstall current branch's fla
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.', '-q'],
            cwd=project_root, capture_output=True,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmark runner for flash-linear-attention ops',
    )
    parser.add_argument(
        '--op', nargs='+', default=None,
        help='Op name(s) to benchmark, or "all"',
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
             'Auto-detected as "main" when on a feature branch.',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all registered ops and exit',
    )
    args = parser.parse_args()

    if args.list:
        ops = list_ops()
        print(f"Registered ops ({len(ops)}):")
        for name in ops:
            cfg = get_op(name)
            print(f"  {name:30s}  [{cfg.category}]  {cfg.import_path}")
        return

    if args.op is None:
        parser.error("--op is required (use --list to see available ops)")

    op_names = list_ops() if args.op == ['all'] else args.op
    shape_configs = json.loads(args.custom_shapes) if args.custom_shapes else SHAPE_CONFIGS

    machine_info = _get_machine_info()
    print(f"Machine: {machine_info.get('gpu_name', 'N/A')} | "
          f"CUDA {machine_info.get('cuda_version', 'N/A')} | "
          f"PyTorch {machine_info.get('pytorch_version', 'N/A')}")
    print(f"Shapes: {len(shape_configs)} configs")
    print(f"Ops: {op_names}")

    all_results = []
    for op_name in op_names:
        try:
            all_results.extend(benchmark_op(op_name, shape_configs, modes=args.modes))
        except Exception as e:
            logger.error(f"Failed to benchmark {op_name}: {e}")

    # Determine baseline ref: explicit --base, or auto-detect main if on a feature branch.
    git_label = machine_info.get('git_label', 'unknown')
    current_branch = git_label.split('[')[0] if '[' in git_label else git_label
    base_ref = args.base
    if base_ref is None and current_branch not in ('main', 'master', 'unknown'):
        base_ref = 'main'

    baseline, baseline_info = None, None
    if base_ref:
        baseline, baseline_info = _bench_at_ref(
            base_ref, op_names, shape_configs, args.modes)

    # Sort by (mode, B, T, H, D, op) so the table groups by mode first
    mode_order = {'fwd': 0, 'fwdbwd': 1}
    all_results.sort(key=lambda r: (mode_order.get(r['mode'], 9), r['B'], r['T'], r['H'], r['D'], r['op']))

    print_results_table(all_results, machine_info, baseline=baseline, baseline_info=baseline_info)

    if args.json_file:
        output = {'machine_info': machine_info, 'results': all_results}
        with open(args.json_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json_file}")

    return all_results


if __name__ == '__main__':
    main()
