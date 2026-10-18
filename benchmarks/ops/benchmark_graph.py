# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare eager execution and graph replay latency on Ascend."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import platform
import socket
import statistics
import sys
import time
from typing import Any

from benchmarks.ops.utils.graph_registry import GraphBenchmarkCase, get_graph_op, list_graph_ops
from benchmarks.utils.environment import get_git_label
from benchmarks.utils.reporting import print_graph_results
from benchmarks.utils.revisions import run_benchmark

logger = logging.getLogger(__name__)
_WORKER_ENV = "FLA_GRAPH_BENCH_WORKER"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def _measure(
    invocation,
    synchronize,
    warmup: int,
    iterations: int,
    clock_ns=time.perf_counter_ns,
) -> dict[str, float]:
    for _ in range(warmup):
        invocation()
        synchronize()

    samples = []
    for _ in range(iterations):
        start = clock_ns()
        invocation()
        synchronize()
        samples.append((clock_ns() - start) / 1e6)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
    }


def benchmark_case(
    case: GraphBenchmarkCase,
    mode: str,
    engines: tuple[str, ...],
    warmup: int,
    iterations: int,
    graph_factory,
    synchronize,
    clock_ns=time.perf_counter_ns,
) -> dict[str, Any]:
    """Benchmark one materialized case; dependencies are injectable for unit tests."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    unknown = set(engines) - {"eager", "graph"}
    if unknown:
        raise ValueError(f"Unknown benchmark engines: {sorted(unknown)}")

    result: dict[str, Any] = {"validation": "not-run"}
    inference_context = contextlib.nullcontext
    if mode == "fwd":
        import torch

        # inference tensors in shared NPU graph RNG state cannot be updated by later training captures.
        inference_context = torch.no_grad

    # measure eager before capture to preserve the lifecycle of an eager-only baseline.
    if "eager" in engines:
        def eager_invocation():
            with inference_context():
                case.run_once(case.eager_step, case.eager_args, False)

        result["eager"] = _measure(eager_invocation, synchronize, warmup, iterations, clock_ns)

    graphed_step = None
    if "graph" in engines:
        if len(case.capture_args) != len(case.graph_args):
            raise ValueError("capture_args and graph_args must have the same length")
        for index, (captured, live) in enumerate(zip(case.capture_args, case.graph_args)):
            captured_data_ptr = getattr(captured, "data_ptr", None)
            live_data_ptr = getattr(live, "data_ptr", None)
            if callable(captured_data_ptr) and callable(live_data_ptr) and captured_data_ptr() == live_data_ptr():
                raise ValueError(
                    f"Graph argument {index} reuses its capture address; "
                    "provide a distinct live tensor so update cost is measured"
                )
        capture_start = clock_ns()
        with inference_context():
            graphed_step = graph_factory(case.graph_step, case.capture_args)
        synchronize()
        result["capture_ms"] = (clock_ns() - capture_start) / 1e6

        # validate replay against eager execution outside the timed region.
        with inference_context():
            reference = case.run_once(case.eager_step, case.eager_args, True)
            actual = case.run_once(graphed_step, case.graph_args, True)
        synchronize()
        case.assert_close(actual, reference)
        result["validation"] = "passed"

        def graph_invocation():
            with inference_context():
                case.run_once(graphed_step, case.graph_args, False)

        result["graph"] = _measure(graph_invocation, synchronize, warmup, iterations, clock_ns)

    return result


def benchmark_graph_op(
    op_name: str,
    shapes: dict[str, dict[str, Any]],
    modes: tuple[str, ...],
    engines: tuple[str, ...],
    default_dtype: str,
    device: str,
    warmup: int,
    iterations: int,
    seed: int = 2026,
    progress: bool = True,
) -> list[dict[str, Any]]:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Graph benchmark requires an available Ascend NPU")
    if "graph" in engines and not hasattr(torch.npu, "make_graphed_callables"):
        raise RuntimeError("torch.npu.make_graphed_callables is required for graph benchmarks")
    torch.npu.set_device(device)

    config = get_graph_op(op_name)
    unsupported = set(modes) - set(config.supported_modes)
    if unsupported:
        raise ValueError(f"{op_name} does not support graph benchmark modes: {sorted(unsupported)}")

    def graph_factory(step, capture_args):
        return torch.npu.make_graphed_callables(step, capture_args, allow_unused_input=True)

    results = []
    for shape_index, (shape_name, raw_shape) in enumerate(shapes.items()):
        shape = dict(raw_shape)
        shape.setdefault("dtype", default_dtype)
        for mode in modes:
            if progress:
                print(f"  [{op_name}] {shape_name} {mode}: building case", flush=True)
            case = config.build_case(shape, mode, device, seed + shape_index * 10)
            measured = benchmark_case(
                case=case,
                mode=mode,
                engines=engines,
                warmup=warmup,
                iterations=iterations,
                graph_factory=graph_factory,
                synchronize=torch.npu.synchronize,
            )
            results.append({
                "op": op_name,
                "shape": shape_name,
                "mode": mode,
                **shape,
                **case.metadata,
                **measured,
            })
            del case
            torch.npu.empty_cache()
    return results


def _machine_info(device: str) -> dict[str, Any]:
    import torch
    import torch_npu

    torch.npu.set_device(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": torch.npu.get_device_name(torch.npu.current_device()),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cann": getattr(torch.version, "cann", None) or "N/A",
        "git_label": get_git_label(),
    }


def _worker_command(
    op_names: list[str],
    shapes: dict[str, dict[str, Any]],
    modes: tuple[str, ...],
    engines: tuple[str, ...],
    args,
) -> list[str]:
    return [
        "--worker",
        "--op",
        *op_names,
        "--custom-shapes",
        json.dumps(shapes),
        "--modes",
        *modes,
        "--engines",
        *engines,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
    ]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", nargs="+", help="Registered graph op name(s), or 'all'")
    parser.add_argument("--base", default=None, help="Git ref used for the eager baseline, for example origin/main")
    parser.add_argument("--no-base", action="store_true", help="Skip the eager baseline")
    parser.add_argument("--custom-shapes", help="JSON object keyed by shape name; B/T/H/D are required")
    parser.add_argument("--modes", nargs="+", choices=("fwd", "fwdbwd"), default=["fwdbwd"])
    parser.add_argument("--engines", nargs="+", choices=("eager", "graph"), default=["eager", "graph"])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", dest="json_file")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.base and args.no_base:
        parser.error("--base and --no-base are mutually exclusive")
    return parser, args


def main(argv=None):
    parser, args = _parse_args(argv)
    if args.list:
        for name in list_graph_ops():
            config = get_graph_op(name)
            print(f"{name:30s} [{config.category}] modes={','.join(config.supported_modes)}")
        return 0
    if not args.op:
        parser.error("--op is required unless --list is used")

    op_names = list_graph_ops() if args.op == ["all"] else args.op
    for op_name in op_names:
        get_graph_op(op_name)
    if args.custom_shapes:
        shapes = json.loads(args.custom_shapes)
    elif len(op_names) == 1:
        shapes = get_graph_op(op_names[0]).default_shapes
    else:
        parser.error("--custom-shapes is required when benchmarking multiple graph ops")
    if not isinstance(shapes, dict) or not shapes:
        parser.error("--custom-shapes must be a non-empty JSON object")

    modes = tuple(args.modes)
    engines = tuple(dict.fromkeys(args.engines))
    is_worker = args.worker or os.environ.get(_WORKER_ENV) == "1"
    if is_worker:
        info = _machine_info(args.device)
        results = []
        for op_name in op_names:
            results.extend(benchmark_graph_op(
                op_name=op_name,
                shapes=shapes,
                modes=modes,
                engines=engines,
                default_dtype=args.dtype,
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
            ))
        payload = {"machine_info": info, "results": results}
        if args.json_file:
            with open(args.json_file, "w") as file:
                json.dump(payload, file, indent=2)
        else:
            print(json.dumps(payload, indent=2))
        return 0

    current, current_info = run_benchmark(
        module='benchmarks.ops.benchmark_graph',
        arguments=_worker_command(op_names=op_names, shapes=shapes, modes=modes, engines=engines, args=args),
    )
    baseline = baseline_info = None
    base_ref = None if args.no_base else args.base
    if base_ref is None and not args.no_base:
        branch = current_info["git_label"].split("[", 1)[0]
        if branch not in ("main", "master", "unknown"):
            base_ref = "origin/main"
    if base_ref:
        baseline, baseline_info = run_benchmark(
            module='benchmarks.ops.benchmark_graph',
            arguments=_worker_command(op_names=op_names, shapes=shapes, modes=modes, engines=('eager',), args=args),
            ref=base_ref,
        )
    print_graph_results(current=current, current_info=current_info, baseline=baseline, baseline_info=baseline_info)

    if args.json_file:
        payload = {
            "current": {"machine_info": current_info, "results": current},
            "baseline": None if baseline is None else {"machine_info": baseline_info, "results": baseline},
        }
        with open(args.json_file, "w") as file:
            json.dump(payload, file, indent=2)
        print(f"\nResults saved to {args.json_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
