# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.nn.functional import (
    logsigmoid,
)

from benchmarks.utils.distributed import all_gather, distributed_context
from benchmarks.utils.graph import GraphBenchmarkCase, GraphOpConfig
from benchmarks.utils.ops.inputs import (
    SIMPLE_QKV,
    TensorSpec,
    shape_bth,
    shape_bthd,
    sigmoid_transform,
)
from benchmarks.utils.ops.registry import OpConfig
from benchmarks.utils.ops.runner import add_operator_arguments, run_operator
from benchmarks.utils.reporting import BenchmarkResult

if TYPE_CHECKING:
    from collections.abc import Callable


OPERATORS = (
    OpConfig(
        name='chunk_kda',
        import_path='fla.ops.kda',
        inputs={
            **SIMPLE_QKV,
            'g': TensorSpec(shape_fn=shape_bthd, transform=logsigmoid),
            'beta': TensorSpec(shape_fn=shape_bth, transform=sigmoid_transform),
        },
        extra_kwargs={'use_qk_l2norm_in_kernel': True, 'safe_gate': True, 'lower_bound': -5},
        category='gate_beta',
    ),
)


def _as_bool(shape: dict[str, Any], name: str, default: bool = False) -> bool:
    value = shape.get(name, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def normalize_kda_shape(shape: dict[str, Any], default_dtype: str = "float16") -> dict[str, Any]:
    """Validate a KDA graph shape while preserving the common B/T/H/D axes."""

    missing = [name for name in ("B", "T", "H", "D") if name not in shape]
    if missing:
        raise ValueError(f"KDA graph shape is missing required dimensions: {missing}")

    normalized = dict(shape)
    for name in ("B", "T", "H", "D"):
        normalized[name] = int(normalized[name])
    normalized["N"] = int(normalized.get("N", normalized["B"]))
    normalized["actual_T"] = int(normalized.get("actual_T", normalized["T"]))
    normalized["HV"] = int(normalized.get("HV", normalized["H"]))
    normalized["DV"] = int(normalized.get("DV", normalized["D"]))
    normalized["chunk_size"] = int(normalized.get("chunk_size", 64))
    normalized["dtype"] = str(normalized.get("dtype", default_dtype))
    normalized["sequence_profile"] = str(normalized.get("sequence_profile", "balanced"))

    for name in ("fused_options", "safe_gate", "allow_neg_eigval", "disable_recompute"):
        normalized[name] = _as_bool(normalized, name)

    if normalized["B"] != 1:
        raise ValueError("KDA graph benchmark requires B=1 for flattened variable-length inputs")
    if min(normalized[name] for name in ("T", "H", "D", "N", "HV", "DV")) <= 0:
        raise ValueError("T, H, D, N, HV, and DV must all be positive")
    if not 0 < normalized["actual_T"] <= normalized["T"]:
        raise ValueError("actual_T must satisfy 0 < actual_T <= T")
    if normalized["HV"] % normalized["H"]:
        raise ValueError("HV must be divisible by H")
    if normalized["chunk_size"] not in (32, 64):
        raise ValueError("chunk_size must be 32 or 64")
    if normalized["dtype"] not in ("float16", "bfloat16"):
        raise ValueError("dtype must be float16 or bfloat16")
    if normalized["sequence_profile"] not in ("balanced", "ragged", "empty_tail"):
        raise ValueError("sequence_profile must be balanced, ragged, or empty_tail")
    if (normalized["safe_gate"] or normalized["allow_neg_eigval"]) and not normalized["fused_options"]:
        raise ValueError("safe_gate and allow_neg_eigval require fused_options")
    return normalized


def make_cu_seqlens(actual_t: int, num_seqs: int, profile: str) -> list[int]:
    """Create deterministic replay offsets for a variable-length shape."""

    if profile == "balanced":
        return [i * actual_t // num_seqs for i in range(num_seqs)] + [actual_t]
    if profile == "empty_tail":
        return [0] + [actual_t] * num_seqs
    if profile == "ragged":
        # Give the first sequence roughly half the tokens, then distribute the
        # remainder. Duplicate offsets naturally represent empty sequences.
        first = (actual_t + 1) // 2
        if num_seqs == 1:
            return [0, actual_t]
        tail = [first + i * (actual_t - first) // (num_seqs - 1) for i in range(1, num_seqs)]
        return [0, first, *tail]
    raise ValueError(f"Unknown sequence profile: {profile}")


def _build_kda_case(shape: dict[str, Any], mode: str, device: str, seed: int) -> GraphBenchmarkCase:
    import torch
    import torch.nn.functional as F

    from fla.ops.kda import chunk_kda

    shape = normalize_kda_shape(shape)
    if mode not in ("fwd", "fwdbwd"):
        raise ValueError(f"Unsupported KDA graph benchmark mode: {mode}")

    B, T, H, K = (shape[name] for name in ("B", "T", "H", "D"))
    N, actual_t, HV, V = (shape[name] for name in ("N", "actual_T", "HV", "DV"))
    dtype = getattr(torch, shape["dtype"])
    requires_grad = mode == "fwdbwd"
    fused_options = shape["fused_options"]
    state_v_first = fused_options

    def make_inputs(input_seed: int) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator(device).manual_seed(input_seed)
        q = torch.randn(B, T, H, K, dtype=dtype, device=device, generator=generator)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device, generator=generator)
        if not fused_options:
            q = F.normalize(q.float(), dim=-1).to(dtype)
            k = F.normalize(k, dim=-1)
        k = k.to(dtype)
        v = torch.randn(B, T, HV, V, dtype=dtype, device=device, generator=generator)
        if fused_options:
            g = torch.randn(B, T, HV, K, dtype=dtype, device=device, generator=generator)
            beta = torch.randn(B, T, HV, dtype=dtype, device=device, generator=generator)
        else:
            g = F.logsigmoid(torch.randn(B, T, HV, K, dtype=torch.float32, device=device, generator=generator))
            beta = torch.rand(B, T, HV, dtype=dtype, device=device, generator=generator)
        state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        h0 = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator) * 0.01
        A_log = torch.randn(HV, dtype=torch.float32, device=device, generator=generator)
        dt_bias = torch.randn(HV * K, dtype=torch.float32, device=device, generator=generator)
        return tuple(tensor.requires_grad_(requires_grad) for tensor in (q, k, v, g, beta, h0, A_log, dt_bias))

    capture_inputs = make_inputs(seed)
    graph_inputs = make_inputs(seed + 1)
    eager_inputs = tuple(
        tensor[:, :actual_t].detach().clone().requires_grad_(requires_grad) if index < 5
        else tensor.detach().clone().requires_grad_(requires_grad)
        for index, tensor in enumerate(graph_inputs)
    )
    capture_offsets = make_cu_seqlens(T, N, "balanced")
    replay_offsets = make_cu_seqlens(actual_t, N, shape["sequence_profile"])
    capture_cu = torch.tensor(capture_offsets, dtype=torch.long, device=device)
    graph_cu = torch.tensor(replay_offsets, dtype=torch.long, device=device)
    eager_cu = graph_cu.clone()

    generator = torch.Generator(device).manual_seed(seed + 2)
    do = torch.randn(B, T, HV, V, dtype=dtype, device=device, generator=generator)
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    dht = torch.randn(*state_shape, dtype=torch.float32, device=device, generator=generator)

    options = {
        "use_qk_l2norm_in_kernel": fused_options,
        "use_gate_in_kernel": fused_options,
        "use_beta_sigmoid_in_kernel": fused_options,
        "allow_neg_eigval": shape["allow_neg_eigval"],
        "safe_gate": shape["safe_gate"],
        "lower_bound": -5.0 if shape["safe_gate"] else None,
        "disable_recompute": shape["disable_recompute"],
        "state_v_first": state_v_first,
        "chunk_size": shape["chunk_size"],
    }

    def call(inputs: tuple[torch.Tensor, ...], cu_seqlens: torch.Tensor, use_graph: bool):
        q, k, v, g, beta, h0, A_log, dt_bias = inputs
        kwargs = dict(
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            A_log=A_log if fused_options else None,
            dt_bias=dt_bias if fused_options else None,
            **options,
        )
        if use_graph:
            kwargs.update(use_graph=True, max_num_seqs=N)
        return chunk_kda(q, k, v, g, beta, **kwargs)

    def eager_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return call((q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=False)

    def graph_step(q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens):
        return call((q, k, v, g, beta, h0, A_log, dt_bias), cu_seqlens, use_graph=True)

    active_indices = list(range(6))
    if fused_options:
        active_indices.extend((6, 7))
    input_names = ("q", "k", "v", "g", "beta", "h0", "A_log", "dt_bias")

    def run_once(step: Callable, args: tuple[Any, ...], collect: bool) -> dict[str, torch.Tensor]:
        inputs = args[:-1]
        for tensor in inputs:
            if tensor.grad is not None:
                tensor.grad.zero_()
        o, ht = step(*args)
        if mode == "fwdbwd":
            ((o * do[:, :o.shape[1]]).sum() + (ht * dht).sum()).backward()
        if not collect:
            return {}
        result = {
            "o": o[:, :actual_t].detach(),
            "ht": ht.detach(),
        }
        if mode == "fwdbwd":
            for index in active_indices:
                grad = inputs[index].grad
                if grad is None:
                    raise AssertionError(f"Expected a gradient for {input_names[index]}")
                if index < 5:
                    grad = grad[:, :actual_t]
                result[f"d{input_names[index]}"] = grad.detach()
        return result

    def assert_close(actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> None:
        if actual.keys() != reference.keys():
            raise AssertionError(f"Result keys differ: graph={actual.keys()}, eager={reference.keys()}")
        for name in actual:
            if not torch.isfinite(reference[name]).all():
                count = int((~torch.isfinite(reference[name])).sum())
                raise AssertionError(f"Eager reference {name} contains {count} non-finite value(s)")
            if not torch.isfinite(actual[name]).all():
                count = int((~torch.isfinite(actual[name])).sum())
                raise AssertionError(f"Graph result {name} contains {count} non-finite value(s)")
            torch.testing.assert_close(
                actual[name],
                reference[name],
                rtol=3e-3,
                atol=3e-3,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    return GraphBenchmarkCase(
        eager_step=eager_step,
        graph_step=graph_step,
        capture_args=capture_inputs + (capture_cu,),
        eager_args=eager_inputs + (eager_cu,),
        graph_args=graph_inputs + (graph_cu,),
        run_once=run_once,
        assert_close=assert_close,
        metadata=shape,
    )


KDA_GRAPH_SHAPES = {
    "baseline_fp16": {
        "B": 1, "T": 128, "H": 2, "D": 64, "N": 2,
        "dtype": "float16", "chunk_size": 64,
    },
    "fused_bf16": {
        "B": 1, "T": 128, "H": 2, "D": 64, "N": 2,
        "dtype": "bfloat16", "chunk_size": 64, "fused_options": True,
        "safe_gate": True, "allow_neg_eigval": True,
    },
    "chunk32_bf16": {
        "B": 1, "T": 128, "H": 2, "D": 64, "N": 2,
        "dtype": "bfloat16", "chunk_size": 32, "fused_options": True,
    },
    "longer_sequence_fp16": {
        "B": 1, "T": 512, "H": 2, "D": 64, "N": 2,
        "dtype": "float16", "chunk_size": 64,
    },
}

GRAPH = GraphOpConfig(
    name="chunk_kda",
    build_case=_build_kda_case,
    default_shapes=KDA_GRAPH_SHAPES,
    supported_modes=("fwd", "fwdbwd"),
    category="gate_beta",
)


DTYPE = torch.bfloat16


def _print_rank0(*args, **kwargs):
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        dist.barrier()
    else:
        print(*args, **kwargs)


def _measure_cp_tp(fn, step=20, warm_up=10, grad_to_none=None):
    """Benchmark function with CUDA events."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for i in range(warm_up):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None

    # Benchmark
    torch.cuda.synchronize()
    start_event.record()
    for i in range(step):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    return elapsed_time / step


def _profile_kernels(fn, rank, steps=5, warmup=2):
    """Profile individual kernels using PyTorch profiler."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=steps),
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(steps + 1):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            prof.step()

    # Get kernel stats
    kernel_stats = []
    for event in prof.key_averages():
        if event.cuda_time_total > 0:  # Only CUDA kernels
            kernel_stats.append({
                'name': event.key,
                'cuda_time_ms': event.cuda_time_total / 1000,  # Convert to ms
                'calls': event.count,
            })

    # Sort by time
    kernel_stats.sort(key=lambda x: x['cuda_time_ms'], reverse=True)
    return kernel_stats


def _format_kernel_table(kernel_stats, top_n=20):
    """Format kernel stats as a table."""
    if not kernel_stats:
        return "No kernel data available"

    total_time = sum(k['cuda_time_ms'] for k in kernel_stats)

    lines = []
    lines.append(f"{'Rank':<4} {'Kernel Name':<60} {'Time (ms)':<12} {'Calls':<8} {'%':<6}")
    lines.append("-" * 100)

    for i, stat in enumerate(kernel_stats[:top_n]):
        pct = stat['cuda_time_ms'] / total_time * 100 if total_time > 0 else 0
        name = stat['name'][:58] if len(stat['name']) > 58 else stat['name']
        lines.append(f"{i+1:<4} {name:<60} {stat['cuda_time_ms']:<12.3f} {stat['calls']:<8} {pct:<6.1f}")

    lines.append("-" * 100)
    lines.append(f"{'Total':<65} {total_time:<12.3f}")

    return "\n".join(lines)


def _benchmark_cp_tp(args):
    from fla.ops.cp import build_cp_context
    from fla.ops.kda import chunk_kda

    results = []
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Determine CP and TP sizes based on config
    if args.config == "cp8":
        cp_size = args.cp_size if args.cp_size else 8
        tp_size = args.tp_size if args.tp_size else 1
    elif args.config == "cp2tp":
        cp_size = args.cp_size if args.cp_size else 4  # 4-way CP
        tp_size = args.tp_size if args.tp_size else 2  # 2-way TP
    else:
        raise ValueError(f"Unknown config: {args.config}")

    assert cp_size * tp_size == world_size, \
        f"CP size ({cp_size}) * TP size ({tp_size}) must equal world size ({world_size})"

    # Calculate local rank within CP group
    cp_rank = rank // tp_size
    tp_rank = rank % tp_size

    # Create CP and TP groups
    # CP group: ranks that share the same TP rank (communicate along sequence dimension)
    cp_ranks = [i for i in range(tp_rank, world_size, tp_size)]
    # TP group: ranks that share the same CP rank (communicate along head dimension)
    tp_ranks = [i for i in range(cp_rank * tp_size, (cp_rank + 1) * tp_size)]

    cp_group = dist.new_group(cp_ranks)
    tp_group = dist.new_group(tp_ranks)

    # Model dimensions
    B = args.batch
    H = args.heads
    K = args.headdim
    V = args.headdim  # Assuming same dimension for V
    T_total = args.seqlen

    # Validate divisibility for CP and TP
    if T_total % cp_size != 0:
        raise ValueError(
            f"Total sequence length ({T_total}) must be divisible by CP size ({cp_size}). "
            f"Got remainder {T_total % cp_size}."
        )
    if H % tp_size != 0:
        raise ValueError(
            f"Number of heads ({H}) must be divisible by TP size ({tp_size}). "
            f"Got remainder {H % tp_size}."
        )

    T_local = T_total // cp_size
    T_baseline = T_total // cp_size  # For single-GPU baseline test

    # For TP, we split heads
    H_local = H // tp_size

    # Determine test mode
    run_backward = args.backward and not args.forward_only
    test_baseline = args.with_baseline
    profile_kernels_flag = args.profile_kernels

    _print_rank0(f"\n{'='*60}")
    _print_rank0(f"Configuration: {args.config}")
    _print_rank0(f"World Size: {world_size}, CP Size: {cp_size}, TP Size: {tp_size}")
    _print_rank0(f"CP Rank: {cp_rank}, TP Rank: {tp_rank}")
    _print_rank0(f"Batch: {B}, Total Heads: {H}, Local Heads: {H_local}")
    _print_rank0(f"Head Dim: {K}, Total SeqLen: {T_total}, Local SeqLen: {T_local}")
    _print_rank0(f"Mode: {'Forward + Backward' if run_backward else 'Forward Only'}")
    if test_baseline:
        _print_rank0(f"Baseline: Single-GPU {T_baseline} (scale to {T_total})")
    if profile_kernels_flag:
        _print_rank0("Kernel Profiling: Enabled (rank 0 only)")
    _print_rank0(f"{'='*60}\n")

    # No varlen for simplicity - fixed length sequences
    # Create global cu_seqlens for build_cp_context (it will compute local cu_seqlens internally)
    cu_seqlens = torch.tensor([0, T_total], device=device, dtype=torch.int32)

    # Create input tensors with local head dimension for TP
    q = torch.randn(B, T_local, H_local, K, dtype=DTYPE, device=device).requires_grad_(True)
    k = torch.randn(B, T_local, H_local, K, dtype=DTYPE, device=device).requires_grad_(True)
    v = torch.randn(B, T_local, H_local, V, dtype=DTYPE, device=device).requires_grad_(True)
    g = torch.randn(B, T_local, H_local, K, dtype=DTYPE, device=device).requires_grad_(True)
    beta = torch.rand(B, T_local, H_local, dtype=DTYPE, device=device).sigmoid().requires_grad_(True)

    # Create baseline tensors (single GPU, local sequence length)
    if test_baseline:
        q_base = torch.randn(B, T_baseline, H_local, K, dtype=DTYPE, device=device).requires_grad_(True)
        k_base = torch.randn(B, T_baseline, H_local, K, dtype=DTYPE, device=device).requires_grad_(True)
        v_base = torch.randn(B, T_baseline, H_local, V, dtype=DTYPE, device=device).requires_grad_(True)
        g_base = torch.randn(B, T_baseline, H_local, K, dtype=DTYPE, device=device).requires_grad_(True)
        beta_base = torch.rand(B, T_baseline, H_local, dtype=DTYPE, device=device).sigmoid().requires_grad_(True)

    # Output gradient for backward
    if run_backward:
        do = torch.randn(B, T_local, H_local, V, dtype=DTYPE, device=device)
        if test_baseline:
            do_base = torch.randn(B, T_baseline, H_local, V, dtype=DTYPE, device=device)
    else:
        do = None
        do_base = None

    # Create A_log and dt_bias for chunk_kda (local heads)
    A_log = torch.randn(H_local, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H_local * V, device=device, dtype=torch.float32)

    # Build CP context for custom CP implementation
    cp_context = build_cp_context(cu_seqlens, cp_group)
    cp_cu_seqlens = cp_context.cu_seqlens

    # Debug: print cp_context info
    print(f"[Rank {rank}] cp_context: cu_seqlens={cp_context.cu_seqlens.tolist()}, "
          f"is_cp_enabled={cp_context.is_cp_enabled}, is_first_rank={cp_context.is_first_rank}, "
          f"is_last_rank={cp_context.is_last_rank}, pre_num_ranks={cp_context.pre_num_ranks}, "
          f"post_num_ranks={cp_context.post_num_ranks}")

    def kda_with_cp():
        """With CP implementation (using build_cp_context)."""
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
            use_qk_l2norm_in_kernel=True,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cp_cu_seqlens,  # Local cu_seqlens from cp_context
            cp_context=cp_context,
        )

        # TP all-reduce for output (if TP size > 1)
        if tp_size > 1:
            dist.all_reduce(o, group=tp_group)

        dist.barrier()
        if run_backward:
            o.backward(do)
            dist.barrier()
        return o

    def kda_with_all2all_cp():
        """All2All CP implementation: all-gather inputs, compute, scatter outputs."""
        # All-gather inputs across CP group
        if cp_size > 1:
            q_full = all_gather(q.squeeze(0), group=cp_group).unsqueeze(0)
            k_full = all_gather(k.squeeze(0), group=cp_group).unsqueeze(0)
            v_full = all_gather(v.squeeze(0), group=cp_group).unsqueeze(0)
            g_full = all_gather(g.squeeze(0), group=cp_group).unsqueeze(0)
            beta_full = all_gather(beta.squeeze(0), group=cp_group).unsqueeze(0)
        else:
            q_full, k_full, v_full, g_full, beta_full = q, k, v, g, beta

        o_full, _ = chunk_kda(
            q=q_full,
            k=k_full,
            v=v_full,
            g=g_full,
            beta=beta_full,
            cu_seqlens=None,  # Fixed length sequence
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
            use_qk_l2norm_in_kernel=True,
        )

        # TP all-reduce for output (if TP size > 1)
        if tp_size > 1:
            dist.all_reduce(o_full, group=tp_group)

        dist.barrier()
        if run_backward:
            if cp_size > 1:
                do_full = all_gather(do.squeeze(0), group=cp_group).unsqueeze(0)
            else:
                do_full = do
            o_full.backward(do_full)
            dist.barrier()

        # Scatter output (take local portion)
        if cp_size > 1:
            o = o_full.chunk(cp_size, dim=1)[cp_rank]
        else:
            o = o_full
        return o

    def kda_baseline_single_gpu():
        """Baseline: single GPU with local sequence length."""
        o, _ = chunk_kda(
            q=q_base,
            k=k_base,
            v=v_base,
            g=g_base,
            beta=beta_base,
            cu_seqlens=None,  # Fixed length sequence
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5,
            use_qk_l2norm_in_kernel=True,
        )

        if run_backward:
            o.backward(do_base)
        return o

    # Warmup CUDA
    torch.cuda.synchronize()

    # Run kernel profiling (rank 0 only)
    if profile_kernels_flag and rank == 0:
        print(f"\n{'='*60}")
        print("Profiling CP kernels (Rank 0 only)")
        print(f"{'='*60}\n")

        kernel_stats = _profile_kernels(kda_with_cp, rank, steps=5, warmup=2)
        print(_format_kernel_table(kernel_stats, top_n=20))
        print()

    # Run benchmarks
    _print_rank0(f"\n{'='*60}")
    _print_rank0(f"Benchmarking {args.config} configuration")
    _print_rank0(f"Steps: {args.steps}, Warmup: {args.warmup}")
    _print_rank0(f"Mode: {'Forward + Backward' if run_backward else 'Forward Only'}")
    _print_rank0(f"{'='*60}\n")

    # Benchmark CP (with build_cp_context)
    dist.barrier()
    t_cp = _measure_cp_tp(kda_with_cp, step=args.steps, warm_up=args.warmup,
                          grad_to_none=[q, k, v, g, beta] if run_backward else None)
    dist.barrier()

    # Benchmark All2All CP (all-gather approach)
    dist.barrier()
    t_all2all_cp = _measure_cp_tp(kda_with_all2all_cp, step=args.steps, warm_up=args.warmup,
                                  grad_to_none=[q, k, v, g, beta] if run_backward else None)
    dist.barrier()

    # Benchmark baseline (single GPU)
    if test_baseline:
        dist.barrier()
        t_baseline_local = _measure_cp_tp(kda_baseline_single_gpu, step=args.steps, warm_up=args.warmup,
                                          grad_to_none=[q_base, k_base, v_base, g_base, beta_base] if run_backward else None)
        dist.barrier()

    timings = [('cp', t_cp), ('all_to_all', t_all2all_cp)]
    if test_baseline:
        timings.append(('single_gpu', t_baseline_local))
    for provider, elapsed in timings:
        results.append(BenchmarkResult(
            name='kda',
            parameters={'provider': provider, 'B': B, 'T': T_total, 'H': H, 'K': K, 'V': V,
                        'cp_size': cp_size, 'tp_size': tp_size, 'backward': run_backward},
            metrics={'mean_ms': elapsed},
            primary_metric='mean_ms',
        ))
    return results


def _add_cp_tp_arguments(parser):
    parser.add_argument("--config", type=str, choices=["cp8", "cp2tp"], default="cp8",
                        help="Configuration: cp8 (8-way CP) or cp2tp (CP2TP)")
    parser.add_argument("--seqlen", type=int, default=131072, help="Total sequence length (default: 128k)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--heads", type=int, default=32, help="Number of heads (default: 32)")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--cp-size", type=int, default=None, help="Context parallel size (overrides config)")
    parser.add_argument("--tp-size", type=int, default=None, help="Tensor parallel size (overrides config)")
    parser.add_argument("--backward", action="store_true", help="Enable backward pass (default: forward only)")
    parser.add_argument("--forward-only", action="store_true", help="Only run the forward pass")
    parser.add_argument("--with-baseline", action="store_true", help="Also test single-GPU baseline (local seqlen / cp_size)")
    parser.add_argument("--profile-kernels", action="store_true", help="Profile individual kernels (rank 0 only)")
    parser.add_argument("--steps", type=int, default=20, help="Benchmark steps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")


def _run_cp_tp(args):
    with distributed_context():
        rank = dist.get_rank()
        torch.manual_seed(rank + 42)
        torch.cuda.manual_seed(rank + 42)
        random.seed(42)
        return _benchmark_cp_tp(args)


SCENARIOS = ('latency', 'graph', 'cp', 'cp_tp')


def add_arguments(parser, scenario):
    from benchmarks.utils.distributed import add_cp_arguments
    from benchmarks.utils.graph import add_graph_arguments

    if scenario == 'graph':
        add_graph_arguments(parser)
    elif scenario == 'cp':
        add_cp_arguments(parser)
    elif scenario == 'cp_tp':
        _add_cp_tp_arguments(parser)
    else:
        add_operator_arguments(parser=parser, configs=OPERATORS)


def run(args):
    if args.scenario == 'graph':
        from benchmarks.utils.graph import run_graph
        return run_graph(args=args, config=GRAPH)
    if args.scenario == 'cp':
        from benchmarks.utils.distributed import run_distributed
        from benchmarks.utils.ops.distributed import benchmark_ops
        args.kda = True
        return run_distributed(args=args, measure=benchmark_ops)
    if args.scenario == 'cp_tp':
        return _run_cp_tp(args)
    return run_operator(args=args, configs=OPERATORS)
