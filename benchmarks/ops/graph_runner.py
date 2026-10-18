# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Eager execution, graph capture, validation, and replay timing."""

from __future__ import annotations

import contextlib
import math
import statistics
import time
from typing import Any

from benchmarks.ops.graph_registry import GraphBenchmarkCase, get_graph_op


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
