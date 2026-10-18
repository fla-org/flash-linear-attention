# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Command-line orchestration for Ascend eager and graph benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
from typing import Any

from benchmarks.comparison import run_benchmark
from benchmarks.environment import get_git_label
from benchmarks.ops.graph_registry import get_graph_op, list_graph_ops
from benchmarks.ops.graph_runner import benchmark_graph_op
from benchmarks.reporting import print_graph_results

logger = logging.getLogger(__name__)
_WORKER_ENV = "FLA_GRAPH_BENCH_WORKER"


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
        "--graph",
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


def run_graph_benchmark(argv=None):
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
        module='benchmarks.ops.run',
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
            module='benchmarks.ops.run',
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
