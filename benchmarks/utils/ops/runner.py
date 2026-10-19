# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Warmup and latency measurement for registered operators."""

from __future__ import annotations

import importlib
import json
import logging
import os

import torch

from benchmarks.utils.environment import synchronize_device
from benchmarks.utils.ops.inputs import generate_inputs
from benchmarks.utils.ops.registry import SHAPE_CONFIGS, OpConfig, get_op, list_ops
from benchmarks.utils.reporting import BenchmarkResult
from benchmarks.utils.revisions import run_benchmark

logger = logging.getLogger(__name__)


def run_benchmarks(
    op_names: list[str],
    shapes: dict[str, dict[str, int]],
    modes: list[str],
    backend: str | None = None,
    ref: str | None = None,
) -> tuple[list[dict], dict]:
    arguments = ['ops.*', '--scenario', 'latency', '--op', *op_names, '--custom-shapes', json.dumps(shapes), '--modes', *modes]
    if backend is not None:
        arguments.extend(['--backend', backend])
    return run_benchmark(module='benchmarks.run', arguments=arguments, ref=ref)


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


def _warmup_iterations() -> int:
    """Extra per-shape forward+backward iterations before timing (not Triton do_bench warmup)."""
    return max(1, int(os.environ.get('FLA_BENCH_OP_WARMUP_ITERS', '5')))


def _timing_options():
    """Triton ``do_bench`` uses warmup/rep in *milliseconds* of timed execution (see Triton docs)."""
    warmup_ms = int(os.environ.get('FLA_BENCH_WARMUP_MS', '25'))
    rep_ms = int(os.environ.get('FLA_BENCH_REP_MS', '100'))
    return {'warmup': max(1, warmup_ms), 'rep': max(1, rep_ms)}


def _warmup_autotune(fn, iterations: int | None = None, device: str | None = None):
    """Run *fn* multiple times so triton autotuning is fully cached."""
    if iterations is None:
        iterations = _warmup_iterations()
    for _ in range(iterations):
        fn()
    synchronize_device(device)


def benchmark_op(
    op_name: str,
    shapes: dict[str, dict[str, int]],
    modes: list[str] | None = None,
    backend: str | None = None,
) -> list[BenchmarkResult]:
    """Return one latency record per supported shape and execution mode."""
    import triton

    from fla.utils import device_name

    if modes is None:
        modes = ['fwd', 'fwdbwd']

    config = get_op(op_name)
    op_fn = _import_op(config)

    call_kwargs = dict(config.extra_kwargs)
    op_label = op_name
    backend_env = config.backend_env or {}
    if backend is not None:
        # explicit benchmark selection must not inherit a global Gluon opt-in.
        os.environ['FLA_GLUON'] = '0'
    if backend and backend != 'triton':
        env = backend_env.get(backend)
        if env is not None:
            os.environ[env] = '1'
            op_label = f"{op_name}[{backend}]"
        else:
            logger.info(f"Op '{op_name}' has no '{backend}' backend; running the default path")
    elif backend == 'triton':
        for env in backend_env.values():
            os.environ[env] = '0'

    if config.skip_backward and 'fwdbwd' in modes:
        modes = [m for m in modes if m != 'fwdbwd']

    if config.default_shapes is not None:
        shapes = config.default_shapes

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

    dtype = torch.bfloat16

    # warm every shape first so later compilation does not interrupt the timed sweep.
    print(f"\n  [{op_name}] Warming up {len(valid_shapes)} shape(s)...")
    failed_shapes = set()
    for shape_name, shape_dict in valid_shapes.items():
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        extra_shape_kw = {k: v for k, v in shape_dict.items() if k not in ('B', 'T', 'H', 'D')}
        try:
            inputs = generate_inputs(config=config, B=B, T=T, H=H, D=D, dtype=dtype, device=device_name, **extra_shape_kw)
            out = op_fn(**inputs, **call_kwargs)
            out_tensor = out[0] if config.output_is_tuple else out
            do = torch.randn_like(out_tensor)

            def _fwdbwd_fn(inputs=inputs, do=do):
                result = op_fn(**inputs, **call_kwargs)
                if not config.skip_backward:
                    t = result[0] if config.output_is_tuple else result
                    t.backward(do)

            _warmup_autotune(fn=_fwdbwd_fn, device=device_name)
        except Exception as e:
            logger.warning(f"Warmup failed for {op_name} @ {shape_name}: {e}")
            failed_shapes.add(shape_name)

    for name in failed_shapes:
        del valid_shapes[name]
    print(f"  [{op_name}] Warmup done.")

    results = []
    for shape_name, shape_dict in list(valid_shapes.items()):
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        extra_shape_kw = {k: v for k, v in shape_dict.items() if k not in ('B', 'T', 'H', 'D')}
        try:
            inputs = generate_inputs(config=config, B=B, T=T, H=H, D=D, dtype=dtype, device=device_name, **extra_shape_kw)
        except Exception as e:
            logger.warning(f"Input generation failed for {op_name} @ {shape_name}: {e}")
            continue

        out = op_fn(**inputs, **call_kwargs)
        out_tensor = out[0] if config.output_is_tuple else out
        do = torch.randn_like(out_tensor)

        for mode in modes:
            if mode == 'fwd':
                def fn(inputs=inputs):
                    return op_fn(**inputs, **call_kwargs)
            else:
                def fn(inputs=inputs, do=do):
                    result = op_fn(**inputs, **call_kwargs)
                    t = result[0] if config.output_is_tuple else result
                    t.backward(do)

            try:
                ms = triton.testing.do_bench(
                    fn, quantiles=[0.5, 0.2, 0.8], **_timing_options()
                )
            except Exception as e:
                logger.warning(f"Bench failed for {op_name} {mode} @ {shape_name}: {e}")
                continue

            results.append(BenchmarkResult(
                name=op_label,
                parameters={'mode': mode, 'B': B, 'T': T, 'H': H, 'D': D, **extra_shape_kw},
                metrics={'median_ms': ms[0], 'p20_ms': ms[1], 'p80_ms': ms[2]},
            ))

    return results


def add_operator_arguments(parser, configs):
    parser.add_argument(
        '--op', nargs='+', choices=list_ops(), metavar='OP',
        help=f'Select operator variants; this object provides: {", ".join(config.name for config in configs)}',
    )
    parser.add_argument('--backend', help='Registered backend override, such as triton or gluon')
    parser.add_argument('--custom-shapes', help='JSON object keyed by shape name, with B/T/H/D and optional operator axes')
    parser.add_argument('--modes', nargs='+', choices=['fwd', 'fwdbwd'], default=['fwd', 'fwdbwd'])


def run_operator(args, configs):
    shapes = json.loads(args.custom_shapes) if args.custom_shapes else SHAPE_CONFIGS
    results = []
    for config in configs:
        if args.op is None or config.name in args.op:
            results.extend(benchmark_op(op_name=config.name, shapes=shapes, modes=args.modes, backend=args.backend))
    return results
