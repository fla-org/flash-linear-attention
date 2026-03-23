# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Unified CLI benchmark runner for all registered ops.

Works both as a package module and as a standalone script from a temp directory:
    python -m benchmarks.ops.run --op chunk_gla --shapes ci --json out.json
    python /tmp/fla_bench_xxx/run.py --op chunk_gla --shapes ci --json out.json

The runner imports fla.ops.* via importlib at call time, so it resolves to
whatever fla version is currently installed. This is critical for cross-commit
comparisons where the runner is copied to a temp dir and the fla package is
reinstalled at each commit.
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
    OpConfig,
    generate_inputs,
    get_op,
    get_shape_configs,
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

            result = {
                'op': op_name,
                'mode': mode,
                'B': B, 'T': T, 'H': H, 'D': D,
                'median_ms': ms[0],
                'p20_ms': ms[1],
                'p80_ms': ms[2],
            }
            results.append(result)
            print(
                f"  {op_name:30s} {mode:6s} B={B} T={T:5d} H={H:2d} D={D:3d}: "
                f"{ms[0]:8.3f} ms (p20={ms[1]:.3f}, p80={ms[2]:.3f})"
            )

    return results


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
        '--shapes', default='ci', choices=['ci', 'full', 'custom'],
        help='Shape config preset (default: ci)',
    )
    parser.add_argument(
        '--custom-shapes', default=None,
        help='JSON string for custom shape dict, e.g. \'{"my": {"B":1,"T":2048,"H":16,"D":128}}\'',
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
    if args.shapes == 'custom':
        if args.custom_shapes is None:
            parser.error("--custom-shapes is required when --shapes=custom")
        shape_configs = json.loads(args.custom_shapes)
    else:
        shape_configs = get_shape_configs(args.shapes)

    # Run
    machine_info = _get_machine_info()
    print(f"Machine: {machine_info.get('gpu_name', 'N/A')} | "
          f"CUDA {machine_info.get('cuda_version', 'N/A')} | "
          f"PyTorch {machine_info.get('pytorch_version', 'N/A')}")
    print(f"Shapes: {args.shapes} ({len(shape_configs)} configs)")
    print(f"Ops: {op_names}")

    all_results = []
    for op_name in op_names:
        try:
            results = benchmark_op(op_name, shape_configs, modes=args.modes)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to benchmark {op_name}: {e}")
            continue

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
