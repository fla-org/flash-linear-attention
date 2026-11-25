# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Measure complete graph-call throughput from concurrent CUPTI GPU activities."""

import statistics

import torch
from torch.profiler import ProfilerActivity, profile

CALLS_PER_GRAPH = 16
LAST_REPORT = None


def union_duration(intervals):
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


@torch.inference_mode()
def do_bench(fn, *, warmup=5, rep=3, backend="cupti"):
    global LAST_REPORT
    if backend != "cupti":
        raise ValueError("This timer requires CUPTI activity tracing")
    LAST_REPORT = None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [fn() for _ in range(CALLS_PER_GRAPH)]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rep):
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            graph.replay()
            torch.cuda.synchronize()
        events = [
            event
            for event in profiler.events()
            if event.device_type.name == "CUDA" and event.time_range.end > event.time_range.start
        ]
        if not events:
            raise RuntimeError("CUPTI did not return GPU activities; no fallback timer is used")
        intervals = [(event.time_range.start, event.time_range.end) for event in events]
        first = min(start for start, _ in intervals)
        last = max(stop for _, stop in intervals)
        busy = union_duration(intervals)
        summed = sum(stop - start for start, stop in intervals)
        samples.append(
            {
                "calls": CALLS_PER_GRAPH,
                "activity_count": len(events),
                "span_us": last - first,
                "per_call_us": (last - first) / CALLS_PER_GRAPH,
                "busy_union_us": busy,
                "summed_activity_us": summed,
                "overlap_us": max(0.0, summed - busy),
                "activities": [
                    {
                        "name": event.name,
                        "start_us": event.time_range.start - first,
                        "duration_us": event.time_range.end - event.time_range.start,
                    }
                    for event in sorted(events, key=lambda event: event.time_range.start)
                ],
            }
        )
    assert len(outputs) == CALLS_PER_GRAPH
    LAST_REPORT = {
        "backend": "cupti_concurrent_gpu_activity_span",
        "calls_per_graph": CALLS_PER_GRAPH,
        "warmup_calls": warmup,
        "warmup_graph_replays": 3,
        "samples": samples,
        "median_us": statistics.median(sample["per_call_us"] for sample in samples),
    }
    return LAST_REPORT["median_us"] / 1000


def last_report():
    return LAST_REPORT
