"""
Benchmark script to compare CP8 and CP2TP configurations for chunk_kda.

Usage:
    # CP8 configuration (8 GPUs, no TP), 128k with backward
    torchrun --nproc_per_node=8 benchmark_kda_cp8_vs_cp2tp.py --config cp8 --seqlen 131072 --backward

    # CP8 configuration, 256k forward only
    torchrun --nproc_per_node=8 benchmark_kda_cp8_vs_cp2tp.py --config cp8 --seqlen 262144 --forward-only

    # CP8 with baseline comparison (test local 32k and scale to 128k)
    torchrun --nproc_per_node=8 benchmark_kda_cp8_vs_cp2tp.py --config cp8 --seqlen 131072 --backward --with-baseline

    # CP8 with detailed kernel profiling (rank 0 only)
    torchrun --nproc_per_node=8 benchmark_kda_cp8_vs_cp2tp.py --config cp8 --seqlen 131072 --forward-only --profile-kernels

    # CP2TP configuration
    torchrun --nproc_per_node=8 benchmark_kda_cp8_vs_cp2tp.py --config cp2tp --seqlen 131072 --backward
"""

import argparse
import os
import random

import torch
import torch.distributed as dist

from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda

# Configuration
DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark chunk_kda with CP8 vs CP2TP")
    parser.add_argument("--config", type=str, choices=["cp8", "cp2tp"], default="cp8",
                        help="Configuration: cp8 (8-way CP) or cp2tp (CP2TP)")
    parser.add_argument("--seqlen", type=int, default=131072, help="Total sequence length (default: 128k)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--heads", type=int, default=32, help="Number of heads (default: 32)")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--cp-size", type=int, default=None, help="Context parallel size (overrides config)")
    parser.add_argument("--tp-size", type=int, default=None, help="Tensor parallel size (overrides config)")
    parser.add_argument("--backward", action="store_true", help="Enable backward pass (default: forward only)")
    parser.add_argument("--forward-only", action="store_true", help="Only run forward pass")
    parser.add_argument("--with-baseline", action="store_true", help="Also test single-GPU baseline (local seqlen / cp_size)")
    parser.add_argument("--profile-kernels", action="store_true", help="Profile individual kernels (rank 0 only)")
    parser.add_argument("--bench", action="store_true", default=True, help="Run benchmark")
    parser.add_argument("--steps", type=int, default=20, help="Benchmark steps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    return parser.parse_args()


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        dist.barrier()
    else:
        print(*args, **kwargs)


def all_gather(x, group=None) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    y = torch.empty(world_size * x.size(0), *x.shape[1:], device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(y, x, group=group)
    return y


def bench(fn, step=20, warm_up=10, grad_to_none=None):
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


def profile_kernels(fn, rank, steps=5, warmup=2):
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


def format_kernel_table(kernel_stats, top_n=20):
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


def run_benchmark(args):
    """Main benchmark function."""
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

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Configuration: {args.config}")
    print_rank0(f"World Size: {world_size}, CP Size: {cp_size}, TP Size: {tp_size}")
    print_rank0(f"CP Rank: {cp_rank}, TP Rank: {tp_rank}")
    print_rank0(f"Batch: {B}, Total Heads: {H}, Local Heads: {H_local}")
    print_rank0(f"Head Dim: {K}, Total SeqLen: {T_total}, Local SeqLen: {T_local}")
    print_rank0(f"Mode: {'Forward + Backward' if run_backward else 'Forward Only'}")
    if test_baseline:
        print_rank0(f"Baseline: Single-GPU {T_baseline} (scale to {T_total})")
    if profile_kernels_flag:
        print_rank0("Kernel Profiling: Enabled (rank 0 only)")
    print_rank0(f"{'='*60}\n")

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

        kernel_stats = profile_kernels(kda_with_cp, rank, steps=5, warmup=2)
        print(format_kernel_table(kernel_stats, top_n=20))
        print()

    # Run benchmarks
    if args.bench:
        print_rank0(f"\n{'='*60}")
        print_rank0(f"Benchmarking {args.config} configuration")
        print_rank0(f"Steps: {args.steps}, Warmup: {args.warmup}")
        print_rank0(f"Mode: {'Forward + Backward' if run_backward else 'Forward Only'}")
        print_rank0(f"{'='*60}\n")

        # Benchmark CP (with build_cp_context)
        dist.barrier()
        t_cp = bench(kda_with_cp, step=args.steps, warm_up=args.warmup,
                     grad_to_none=[q, k, v, g, beta] if run_backward else None)
        dist.barrier()

        # Benchmark All2All CP (all-gather approach)
        dist.barrier()
        t_all2all_cp = bench(kda_with_all2all_cp, step=args.steps, warm_up=args.warmup,
                             grad_to_none=[q, k, v, g, beta] if run_backward else None)
        dist.barrier()

        # Benchmark baseline (single GPU)
        if test_baseline:
            dist.barrier()
            t_baseline_local = bench(kda_baseline_single_gpu, step=args.steps, warm_up=args.warmup,
                                     grad_to_none=[q_base, k_base, v_base, g_base, beta_base] if run_backward else None)
            # Scale to total sequence length (theoretical time on single GPU)
            t_baseline_scaled = t_baseline_local * cp_size
            dist.barrier()

        # Print results
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Results for {args.config} (CP{cp_size}TP{tp_size}):")
            print(f"SeqLen: {T_total}, Heads: {H}, HeadDim: {K}")
            print(f"Mode: {'Forward + Backward' if run_backward else 'Forward Only'}")
            print(f"{'='*60}")
            print(f"CP time:                {t_cp:.3f} ms")
            print(f"All2All CP time:        {t_all2all_cp:.3f} ms")
            print(f"Speedup (vs All2All):   {t_all2all_cp / t_cp:.2f}x")
            if test_baseline:
                print(f"{'='*60}")
                print("Single-GPU Baseline:")
                print(f"  Local {T_baseline}:           {t_baseline_local:.3f} ms")
                print(f"  Scaled to {T_total}:      {t_baseline_scaled:.3f} ms (est.)")
                print(f"Speedup (vs Scaled):    {t_baseline_scaled / t_cp:.2f}x")
            print(f"{'='*60}\n")


def main():
    dist.init_process_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.manual_seed(rank + 42)
    torch.cuda.manual_seed(rank + 42)
    random.seed(42)
    torch.cuda.set_device(local_rank)

    args = get_args()

    # Print environment info
    if rank == 0:
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(local_rank)}")
        print(f"World size: {world_size}\n")

    run_benchmark(args)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
