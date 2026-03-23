# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Unified CLI benchmark runner for all registered ops.

Running benchmarks
==================

    # Benchmark one op (uses all 6 default shape configs)
    python -m benchmarks.ops.run --op chunk_gla

    # Multiple ops
    python -m benchmarks.ops.run --op chunk_gla chunk_kda

    # All registered ops
    python -m benchmarks.ops.run --op all

    # Forward only
    python -m benchmarks.ops.run --op chunk_gla --modes fwd

    # Save results to JSON
    python -m benchmarks.ops.run --op chunk_gla --json results.json

    # Custom shape (overrides default SHAPE_CONFIGS)
    python -m benchmarks.ops.run --op chunk_gla \
        --custom-shapes '{"test": {"B": 2, "T": 4096, "H": 32, "D": 128}}'

    # List all registered ops
    python -m benchmarks.ops.run --list

Cross-commit comparison
=======================
Compare performance between two git commits (defaults: main vs current branch):

    python scripts/run_benchmark_compare.py
    python scripts/run_benchmark_compare.py --benchmark-ops chunk_gla chunk_kda
    python scripts/run_benchmark_compare.py --base HEAD~3

The compare script copies run.py + registry.py to a temp dir, then does
``git checkout`` + ``pip install -e .`` at each commit.  run.py imports
fla.ops.* via importlib, so it picks up whatever kernel version is installed.

Registering a new op
====================
All op definitions live in ``registry.py``.  To add a new op:

1. Pick shape helpers for each input tensor (defined in registry.py):

       shape_BTHD  -> (B, T, H, D)     most q/k/v/g tensors
       shape_BTH   -> (B, T, H)         per-head scalars (gates, beta)
       shape_BTD   -> (B, T, H*D)       flattened hidden dim (HGRN)
       shape_HD    -> (H, D)            per-head vectors (RWKV u)
       shape_H     -> (H,)              per-head scalars

2. Pick transforms to map randn into the right value range:

       logsigmoid        -> negative values (log-space gates)
       sigmoid_transform -> (0, 1) range (beta)
       logsigmoid_clamp  -> logsigmoid clamped to >= -5

3. Call register_op() in registry.py:

       register_op(OpConfig(
           name='chunk_my_op',                    # function name in the module
           import_path='fla.ops.my_op',           # module for importlib.import_module()
           inputs={
               'q': TensorSpec(shape_BTHD),
               'k': TensorSpec(shape_BTHD),
               'v': TensorSpec(shape_BTHD),
               'g': TensorSpec(shape_BTH, transform=logsigmoid),
           },
           extra_kwargs={'use_some_flag': True},   # non-tensor kwargs passed to the op
           category='my_group',                    # label for --list output
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
import socket
import sys

import torch

# ---------------------------------------------------------------------------
# Import registry: works from both package and standalone contexts.
# When run as `python /tmp/fla_bench_xxx/run.py`, registry.py is in the same
# directory.  When run as `python -m benchmarks.ops.run`, normal relative
# imports would work but we keep the sys.path approach for uniformity.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import (  # noqa: E402
    SHAPE_CONFIGS,
    OpConfig,
    generate_inputs,
    get_op,
    list_ops,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Op loader
# ---------------------------------------------------------------------------


def _import_op(config: OpConfig):
    """Dynamically import the op function from the installed fla package."""
    mod = importlib.import_module(config.import_path)
    fn = getattr(mod, config.name, None)
    if fn is None:
        raise ImportError(
            f"Cannot find '{config.name}' in module '{config.import_path}'. "
            f"Available: {[x for x in dir(mod) if not x.startswith('_')]}"
        )
    return fn


# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------


def _get_machine_info() -> dict:
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda or 'N/A',
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


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup_autotune(fn, n=5):
    """Run fn multiple times to trigger all triton autotuning before timing.

    triton.testing.do_bench has its own warmup, but its first fn() call
    triggers autotuning for every kernel in the call graph. Autotuning
    explores many configs and is orders of magnitude slower than a normal
    call. Calling fn here guarantees all kernel configs are cached before
    do_bench starts.
    """
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def benchmark_op(
    op_name: str,
    shapes: dict[str, dict[str, int]],
    modes: list[str] | None = None,
) -> list[dict]:
    """Benchmark a single op across all shapes and modes.

    Returns list of result dicts with timing info.
    """
    import triton

    if modes is None:
        modes = ['fwd', 'fwdbwd']

    config = get_op(op_name)
    op_fn = _import_op(config)

    if config.skip_backward and 'fwdbwd' in modes:
        modes = [m for m in modes if m != 'fwdbwd']

    # Filter shapes by dim_constraints (fast, no GPU allocation)
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

    # Phase 1: warmup ALL shapes before timing ANY.
    # Collect shapes that fail warmup so we can skip them in timing.
    print(f"\n  [{op_name}] Warming up {len(valid_shapes)} shape(s)...")
    failed_shapes = set()
    for shape_name, shape_dict in valid_shapes.items():
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        try:
            inputs = generate_inputs(config, B, T, H, D, dtype=dtype, device=device)

            # Run fwd once to get output shape for backward grad tensor
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
            continue

    for name in failed_shapes:
        del valid_shapes[name]
    print(f"  [{op_name}] Warmup done.")

    # Phase 2: actual timing
    results = []
    for shape_name, shape_dict in list(valid_shapes.items()):
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        try:
            inputs = generate_inputs(config, B, T, H, D, dtype=dtype, device=device)
        except Exception as e:
            logger.warning(f"Input generation failed for {op_name} @ {shape_name}: {e}")
            continue

        # Get output shape for backward grad tensor
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
                logger.warning(
                    f"Bench failed for {op_name} {mode} @ {shape_name}: {e}"
                )
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


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def print_results_table(results: list[dict], machine_info: dict | None = None):
    """Print benchmark results as an aligned table."""
    if not results:
        print("\n  No results to display.")
        return

    print(f"\n{'=' * 95}")
    if machine_info:
        gpu = machine_info.get('gpu_name', 'N/A')
        cuda = machine_info.get('cuda_version', 'N/A')
        pytorch = machine_info.get('pytorch_version', 'N/A')
        print(f"  Machine: {gpu} | CUDA {cuda} | PyTorch {pytorch}")
    print(f"{'=' * 95}")
    print(f"  {'op':<28s} {'mode':<7s} {'shape':<24s} "
          f"{'median(ms)':>10s} {'p20(ms)':>10s} {'p80(ms)':>10s}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 10}")

    for r in results:
        shape_str = f"B={r['B']} T={r['T']} H={r['H']} D={r['D']}"
        print(f"  {r['op']:<28s} {r['mode']:<7s} {shape_str:<24s} "
              f"{r['median_ms']:>10.3f} {r['p20_ms']:>10.3f} {r['p80_ms']:>10.3f}")

    print(f"{'=' * 95}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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

    # Determine ops
    if args.op is None:
        parser.error("--op is required (use --list to see available ops)")

    if args.op == ['all']:
        op_names = list_ops()
    else:
        op_names = args.op

    # Determine shapes
    if args.custom_shapes:
        shape_configs = json.loads(args.custom_shapes)
    else:
        shape_configs = SHAPE_CONFIGS

    # Run
    machine_info = _get_machine_info()
    print(f"Machine: {machine_info.get('gpu_name', 'N/A')} | "
          f"CUDA {machine_info.get('cuda_version', 'N/A')} | "
          f"PyTorch {machine_info.get('pytorch_version', 'N/A')}")
    print(f"Shapes: {len(shape_configs)} configs")
    print(f"Ops: {op_names}")

    all_results = []
    for op_name in op_names:
        try:
            results = benchmark_op(op_name, shape_configs, modes=args.modes)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to benchmark {op_name}: {e}")
            continue

    print_results_table(all_results, machine_info)

    # Save JSON
    if args.json_file:
        output = {
            'machine_info': machine_info,
            'results': all_results,
        }
        with open(args.json_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json_file}")

    return all_results


if __name__ == '__main__':
    main()
