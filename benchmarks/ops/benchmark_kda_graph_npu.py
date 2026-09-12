# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Measure eager vs NPUGraph KDA training latency on Ascend NPUs."""

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F
import torch_npu

from fla.ops.kda import chunk_kda


def make_inputs(seed, tokens, heads, dim, num_seqs, dtype, device):
    generator = torch.Generator(device).manual_seed(seed)
    inputs = (
        torch.randn(1, tokens, heads, dim, dtype=dtype, device=device, generator=generator),
        F.normalize(
            torch.randn(1, tokens, heads, dim, dtype=torch.float32, device=device, generator=generator),
            dim=-1,
        ).to(dtype),
        torch.randn(1, tokens, heads, dim, dtype=dtype, device=device, generator=generator),
        F.logsigmoid(
            torch.randn(1, tokens, heads, dim, dtype=torch.float32, device=device, generator=generator),
        ),
        torch.rand(1, tokens, heads, dtype=dtype, device=device, generator=generator),
        torch.randn(num_seqs, heads, dim, dim, dtype=torch.float32, device=device, generator=generator),
    )
    return tuple(tensor.requires_grad_() for tensor in inputs)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def measure(step, inputs, cu_seqlens, do, dht, warmup, iterations):
    def run_once():
        for tensor in inputs:
            if tensor.grad is not None:
                tensor.grad.zero_()
        o, ht = step(*inputs, cu_seqlens)
        ((o * do).sum() + (ht * dht).sum()).backward()
        torch.npu.synchronize()

    for _ in range(warmup):
        run_once()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        run_once()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This benchmark requires an Ascend NPU")
    if args.tokens < args.num_seqs:
        raise ValueError("tokens must be greater than or equal to num-seqs")

    torch.npu.set_device(args.device)
    dtype = torch.float16
    offsets = [i * args.tokens // args.num_seqs for i in range(args.num_seqs)] + [args.tokens]
    cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=args.device)
    generator = torch.Generator(args.device).manual_seed(2026)
    do = torch.randn(
        1,
        args.tokens,
        args.heads,
        args.dim,
        dtype=dtype,
        device=args.device,
        generator=generator,
    )
    dht = torch.randn(
        args.num_seqs,
        args.heads,
        args.dim,
        args.dim,
        dtype=torch.float32,
        device=args.device,
        generator=generator,
    )

    def eager_step(q, k, v, g, beta, h0, cu):
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
        )

    def graph_step(q, k, v, g, beta, h0, cu):
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            use_graph=True,
            max_num_seqs=args.num_seqs,
        )

    graph_inputs = make_inputs(0, args.tokens, args.heads, args.dim, args.num_seqs, dtype, args.device)
    graphed_step = torch.npu.make_graphed_callables(
        graph_step,
        graph_inputs + (cu_seqlens,),
        allow_unused_input=True,
    )
    eager_inputs = make_inputs(0, args.tokens, args.heads, args.dim, args.num_seqs, dtype, args.device)

    eager = measure(eager_step, eager_inputs, cu_seqlens, do, dht, args.warmup, args.iterations)
    graph = measure(graphed_step, graph_inputs, cu_seqlens, do, dht, args.warmup, args.iterations)
    result = {
        "config": {
            "tokens": args.tokens,
            "heads": args.heads,
            "dim": args.dim,
            "num_seqs": args.num_seqs,
            "dtype": str(dtype),
            "device": torch.npu.get_device_name(torch.npu.current_device()),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "eager_live": eager,
        "graph_update_replay": graph,
        "p50_speedup": eager["p50_ms"] / graph["p50_ms"],
        "p95_speedup": eager["p95_ms"] / graph["p95_ms"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
