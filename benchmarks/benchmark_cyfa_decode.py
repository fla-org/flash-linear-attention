# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Measure host-visible decode latency and allocated memory on one GPU."""

import argparse
import gc
import importlib
import json
import statistics
import time
from pathlib import Path

import torch

from fla.ops.cyfa import fused_recurrent_cyfa


def _inputs(batch, heads, key_dim, value_dim, slots, dtype, varlen):
    tokens = batch if varlen else 1
    leading = 1 if varlen else batch
    return {
        "q": torch.randn(leading, tokens, heads, key_dim, device="cuda", dtype=dtype),
        "k": torch.randn(leading, tokens, heads, key_dim, device="cuda", dtype=dtype),
        "v": torch.randn(leading, tokens, heads, value_dim, device="cuda", dtype=dtype),
        "g": -0.1 * torch.rand(leading, tokens, heads, device="cuda"),
        "delta": torch.rand(leading, tokens, heads, device="cuda"),
        "beta": torch.rand(leading, tokens, heads, device="cuda"),
        "readout": torch.eye(slots - 1, device="cuda").expand(heads, -1, -1).contiguous(),
        "q_norm_weight": torch.ones(key_dim, device="cuda"),
        "k_norm_weight": torch.ones(key_dim, device="cuda"),
        "q_norm_eps": 1e-6,
        "k_norm_eps": 1e-6,
        "initial_state": (
            torch.randn(batch, heads, key_dim, slots, device="cuda"),
            torch.randn(batch, heads, slots, value_dim, device="cuda"),
            17 + torch.rand(batch, heads, device="cuda"),
        ),
        "output_final_state": True,
        "cu_seqlens": torch.arange(batch + 1, device="cuda", dtype=torch.int32) if varlen else None,
    }


@torch.no_grad()
def _measure(config, mode, disable_pdl):
    torch.manual_seed(42)
    torch.compiler.reset()
    module = importlib.import_module("fla.ops.cyfa.fused_recurrent")
    native_pdl = module._device_supports_pdl
    if disable_pdl:
        module._device_supports_pdl = lambda index: False
    try:
        inputs = _inputs(*config)
        run = torch.compile(fused_recurrent_cyfa) if mode == "compiled" else fused_recurrent_cyfa
        for _ in range(20):
            output = run(**inputs)
        del output
        torch.cuda.synchronize()
        wall, device = [], []
        steps, repeats = 100, 9
        for _ in range(repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            begin.record()
            for _ in range(steps):
                output = run(**inputs)
            end.record()
            end.synchronize()
            wall.append((time.perf_counter_ns() - started) / steps / 1000)
            device.append(begin.elapsed_time(end) * 1000 / steps)
            del output
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        output = run(**inputs)
        torch.cuda.synchronize()
        live_delta = torch.cuda.memory_allocated() - before
        peak_delta = torch.cuda.max_memory_allocated() - before
        state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in inputs["initial_state"][:2])
        for tensor in (output[0], *output[1]):
            assert torch.isfinite(tensor).all()
        record = {
            "shape": config[:5],
            "dtype": str(config[5]),
            "varlen": config[6],
            "mode": mode,
            "disable_pdl": disable_pdl,
            "wall_us_median": statistics.median(wall),
            "wall_us_samples": wall,
            "cuda_event_us_median": statistics.median(device),
            "cuda_event_us_samples": device,
            "live_allocated_delta_bytes": live_delta,
            "peak_allocated_delta_bytes": peak_delta,
            "kv_state_bytes": state_bytes,
            "steps_per_sample": steps,
            "samples": repeats,
            "method": "Repeated public-op calls with fixed state references; no CUDA graphs. "
            "The buggy baseline mutates K/V in eager mode and may return stale compiled state. "
            "Timing quantifies this fix's cost, not correct-generation throughput.",
        }
        print("BENCHMARK " + json.dumps(record), flush=True)
        return record
    finally:
        module._device_supports_pdl = native_pdl


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = [
        (1, 2, 64, 64, 64, torch.float32, False),
        (1, 2, 64, 64, 64, torch.bfloat16, False),
        (1, 16, 64, 64, 128, torch.bfloat16, False),
        (8, 16, 64, 64, 128, torch.bfloat16, False),
        (1, 2, 64, 64, 64, torch.bfloat16, True),
        (8, 16, 64, 64, 128, torch.bfloat16, True),
    ]
    results = []
    for config in configs:
        for disable_pdl in (False, True):
            results.append(_measure(config, "eager", disable_pdl))
        if not config[6] and config[0] == 1 and config[5] == torch.bfloat16:
            results.append(_measure(config, "compiled", False))
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
