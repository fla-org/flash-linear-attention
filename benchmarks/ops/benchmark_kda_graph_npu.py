# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

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


def make_inputs(seed, tokens, heads, value_heads, key_dim, value_dim, num_seqs, dtype, device, fused_options):
    generator = torch.Generator(device).manual_seed(seed)
    state_shape = (num_seqs, value_heads, value_dim, key_dim) if fused_options else (
        num_seqs,
        value_heads,
        key_dim,
        value_dim,
    )
    inputs = (
        torch.randn(1, tokens, heads, key_dim, dtype=dtype, device=device, generator=generator),
        F.normalize(
            torch.randn(1, tokens, heads, key_dim, dtype=torch.float32, device=device, generator=generator),
            dim=-1,
        ).to(dtype),
        torch.randn(1, tokens, value_heads, value_dim, dtype=dtype, device=device, generator=generator),
        torch.randn(1, tokens, value_heads, key_dim, dtype=dtype, device=device, generator=generator)
        if fused_options
        else F.logsigmoid(
            torch.randn(1, tokens, value_heads, key_dim, dtype=torch.float32, device=device, generator=generator),
        ),
        torch.randn(1, tokens, value_heads, dtype=dtype, device=device, generator=generator)
        if fused_options
        else torch.rand(1, tokens, value_heads, dtype=dtype, device=device, generator=generator),
        torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator),
        torch.randn(value_heads, dtype=torch.float32, device=device, generator=generator),
        torch.randn(value_heads * key_dim, dtype=torch.float32, device=device, generator=generator),
    )
    return tuple(tensor.requires_grad_() for tensor in inputs)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def run_once(step, inputs, cu_seqlens, do, dht):
    for tensor in inputs:
        if tensor.grad is not None:
            tensor.grad.zero_()
    o, ht = step(*inputs, cu_seqlens)
    ((o * do).sum() + (ht * dht).sum()).backward()
    torch.npu.synchronize()
    return (o, ht, *(tensor.grad for tensor in inputs if tensor.grad is not None))


def measure(step, inputs, cu_seqlens, do, dht, warmup, iterations):

    for _ in range(warmup):
        run_once(step, inputs, cu_seqlens, do, dht)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        run_once(step, inputs, cu_seqlens, do, dht)
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
    parser.add_argument("--value-heads", type=int)
    parser.add_argument("--value-dim", type=int)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--chunk-size", choices=(32, 64), type=int, default=64)
    parser.add_argument("--fused-options", action="store_true")
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--allow-neg-eigval", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This benchmark requires an Ascend NPU")
    if args.tokens < args.num_seqs:
        raise ValueError("tokens must be greater than or equal to num-seqs")
    if (args.safe_gate or args.allow_neg_eigval) and not args.fused_options:
        raise ValueError("safe-gate and allow-neg-eigval require fused-options")

    torch.npu.set_device(args.device)
    dtype = getattr(torch, args.dtype)
    value_heads = args.value_heads or args.heads
    value_dim = args.value_dim or args.dim
    if value_heads % args.heads:
        raise ValueError("value-heads must be divisible by heads")
    offsets = [i * args.tokens // args.num_seqs for i in range(args.num_seqs)] + [args.tokens]
    cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=args.device)
    generator = torch.Generator(args.device).manual_seed(2026)
    do = torch.randn(
        1,
        args.tokens,
        value_heads,
        value_dim,
        dtype=dtype,
        device=args.device,
        generator=generator,
    )
    state_shape = (args.num_seqs, value_heads, value_dim, args.dim) if args.fused_options else (
        args.num_seqs,
        value_heads,
        args.dim,
        value_dim,
    )
    dht = torch.randn(
        *state_shape,
        dtype=torch.float32,
        device=args.device,
        generator=generator,
    )

    op_options = {
        "use_qk_l2norm_in_kernel": args.fused_options,
        "use_gate_in_kernel": args.fused_options,
        "use_beta_sigmoid_in_kernel": args.fused_options,
        "allow_neg_eigval": args.allow_neg_eigval,
        "safe_gate": args.safe_gate,
        "lower_bound": -5.0 if args.safe_gate else None,
        "disable_recompute": args.disable_recompute,
        "state_v_first": args.fused_options,
        "chunk_size": args.chunk_size,
    }

    def eager_step(q, k, v, g, beta, h0, A_log, dt_bias, cu):
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            A_log=A_log if args.fused_options else None,
            dt_bias=dt_bias if args.fused_options else None,
            **op_options,
        )

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu):
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
            A_log=A_log if args.fused_options else None,
            dt_bias=dt_bias if args.fused_options else None,
            **op_options,
        )

    input_args = (
        args.tokens,
        args.heads,
        value_heads,
        args.dim,
        value_dim,
        args.num_seqs,
        dtype,
        args.device,
        args.fused_options,
    )
    capture_inputs = make_inputs(0, *input_args)
    capture_offsets = [*range(args.num_seqs), args.tokens]
    capture_cu_seqlens = torch.tensor(capture_offsets, dtype=torch.long, device=args.device)
    graphed_step = torch.npu.make_graphed_callables(
        graph_step,
        capture_inputs + (capture_cu_seqlens,),
        allow_unused_input=True,
    )
    graph_inputs = make_inputs(0, *input_args)
    graph_cu_seqlens = cu_seqlens.clone()
    assert all(captured.data_ptr() != live.data_ptr() for captured, live in zip(capture_inputs, graph_inputs))
    assert capture_cu_seqlens.data_ptr() != graph_cu_seqlens.data_ptr()
    eager_inputs = make_inputs(0, *input_args)

    eager_result = run_once(eager_step, eager_inputs, cu_seqlens, do, dht)
    graph_result = run_once(graphed_step, graph_inputs, graph_cu_seqlens, do, dht)
    for actual, reference in zip(graph_result, eager_result):
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)

    eager = measure(eager_step, eager_inputs, cu_seqlens, do, dht, args.warmup, args.iterations)
    graph = measure(graphed_step, graph_inputs, graph_cu_seqlens, do, dht, args.warmup, args.iterations)
    result = {
        "config": {
            "tokens": args.tokens,
            "heads": args.heads,
            "value_heads": value_heads,
            "key_dim": args.dim,
            "value_dim": value_dim,
            "num_seqs": args.num_seqs,
            "dtype": str(dtype),
            "chunk_size": args.chunk_size,
            "fused_options": args.fused_options,
            "safe_gate": args.safe_gate,
            "allow_neg_eigval": args.allow_neg_eigval,
            "disable_recompute": args.disable_recompute,
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
