# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Generic Ascend NPU profiling collector.

Op-agnostic: wrap any callable that runs the workload (fwd / bwd / both).
Does not import fla or any specific operator.

Examples
--------
# As a library
from profile_npu import profile_callable

def workload():
    y = op(x)
    y.backward(dy)

trace_dir = profile_callable(workload, name="my_op", aic_metrics="PipeUtilization")

# CLI (inline python defining `workload`)
python profile_npu.py --name my_op --metrics PipeUtilization --exec '
import torch
from my_mod import op
from fla.utils import device
x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
def workload():
    y = op(x); y.backward(torch.ones_like(y))
'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable


def _resolve_aic_metrics(name: str):
    import torch_npu

    metrics = torch_npu.profiler.AiCMetrics
    aliases = {
        "pipe": "PipeUtilization",
        "pipeutilization": "PipeUtilization",
        "ub": "MemoryUB",
        "memoryub": "MemoryUB",
        "memory": "Memory",
        "l2": "L2Cache",
        "l2cache": "L2Cache",
        "arith": "ArithmeticUtilization",
        "arithmeticutilization": "ArithmeticUtilization",
        "memaccess": "MemoryAccess",
        "memoryaccess": "MemoryAccess",
        "memoryl0": "MemoryL0",
        "conflict": "ResourceConflictRatio",
        "resourceconflictratio": "ResourceConflictRatio",
        "none": "AiCoreNone",
        "aicorenone": "AiCoreNone",
    }
    key = name.replace("_", "").replace("-", "").lower()
    attr = aliases.get(key, name)
    if not hasattr(metrics, attr):
        available = [x for x in dir(metrics) if not x.startswith("_")]
        raise ValueError(f"Unknown aic_metrics={name!r}. Available: {available}")
    return getattr(metrics, attr)


def profile_callable(
    fn: Callable[[], None],
    *,
    name: str = "npu_op",
    out_dir: str | None = None,
    aic_metrics: str = "PipeUtilization",
    profiler_level: str = "Level1",
    wait: int = 0,
    warmup: int = 1,
    active: int = 1,
    repeat: int = 1,
    skip_first: int = 0,
    steps: int | None = None,
    record_shapes: bool = True,
    with_modules: bool = True,
) -> str:
    """Run ``fn`` under torch_npu.profiler and return the trace directory.

    Schedule semantics (torch_npu): for each repeat, the profiler advances
    wait → warmup → active steps. Call ``prof.step()`` once per iteration.
    Default loop count is ``(wait + warmup + active) * repeat`` (plus skip_first).
    """
    import torch_npu

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    trace_dir = os.path.join(out_dir, f"{name}_profiling_{time.time()}")

    level = getattr(torch_npu.profiler.ProfilerLevel, profiler_level)
    metrics = _resolve_aic_metrics(aic_metrics)
    if steps is None:
        steps = skip_first + (wait + warmup + active) * repeat

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=level,
        aic_metrics=metrics,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        experimental_config=experimental_config,
        record_shapes=record_shapes,
        with_modules=with_modules,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(steps):
            fn()
            prof.step()

    print(f"[profile_npu] trace written to {trace_dir}")
    print(f"[profile_npu] aic_metrics={aic_metrics} level={profiler_level} steps={steps}")
    return trace_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generic Ascend NPU profiler. Provide a Python snippet that defines workload().",
    )
    parser.add_argument("--name", default="npu_op", help="Trace directory prefix")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Parent directory for traces (default: cwd)",
    )
    parser.add_argument(
        "--metrics",
        default="PipeUtilization",
        help="AiCMetrics name or alias: PipeUtilization|MemoryUB|Memory|L2Cache|...",
    )
    parser.add_argument("--level", default="Level1", help="ProfilerLevel: Level0|Level1|Level2")
    parser.add_argument("--wait", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="Override loop iterations")
    parser.add_argument(
        "--exec",
        dest="exec_src",
        default=None,
        help="Python source that defines workload(). Runs in an isolated globals dict.",
    )
    parser.add_argument(
        "--exec-file",
        default=None,
        help="Path to a .py file that defines workload().",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="After profiling, run analyze_profile on the trace",
    )
    parser.add_argument(
        "--kernel-filter",
        default=None,
        help="Substring filter for kernel Name when analyzing",
    )
    args = parser.parse_args(argv)

    if not args.exec_src and not args.exec_file:
        parser.error("Provide --exec or --exec-file that defines workload()")

    g: dict = {"__name__": "__profile_workload__"}
    if args.exec_file:
        with open(args.exec_file, encoding="utf-8") as f:
            src = f.read()
        # Allow relative imports from the file's directory.
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.exec_file)))
        exec(compile(src, args.exec_file, "exec"), g, g)
    else:
        exec(compile(args.exec_src, "<--exec>", "exec"), g, g)

    if "workload" not in g or not callable(g["workload"]):
        raise SystemExit("Snippet must define a callable workload()")

    trace_dir = profile_callable(
        g["workload"],
        name=args.name,
        out_dir=args.out_dir,
        aic_metrics=args.metrics,
        profiler_level=args.level,
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=args.repeat,
        skip_first=args.skip_first,
        steps=args.steps,
    )

    if args.analyze:
        # Late import so this file stays usable without analyze_profile on PYTHONPATH.
        from analyze_profile import summarize_trace

        summarize_trace(trace_dir, kernel_filter=args.kernel_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
