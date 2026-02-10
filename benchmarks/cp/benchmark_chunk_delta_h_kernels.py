#!/usr/bin/env python
"""
Benchmark script for chunk_delta_h kernels with cp=8 using triton.testing.do_bench.

This script benchmarks all 6 kernels from generate_chunk_delta_h_cache():
1. pre_process_fwd_kernel_stage1 (FWD)
2. pre_process_fwd_bwd_kernel_stage2 (FWD)
3. merge_fwd_bwd_kernel (FWD)
4. pre_process_bwd_kernel_stage1 (BWD)
5. pre_process_fwd_bwd_kernel_stage2 (BWD)
6. merge_fwd_bwd_kernel (BWD)

Usage:
    # Run benchmark with default settings (batch=1, head=32, headdim=128, cp=8, seqlen=32k per rank)
    python benchmark_chunk_delta_h_kernels.py

    # Run with custom settings
    python benchmark_chunk_delta_h_kernels.py --batch 1 --heads 32 --headdim 128 --seqlen 32768
"""

import argparse

import torch
import triton
import triton.testing

from fla.ops.cp.chunk_delta_h import (
    merge_fwd_bwd_kernel,
    pre_process_bwd_kernel_merged,
    pre_process_bwd_kernel_stage1,
    pre_process_fwd_bwd_kernel_stage2,
    pre_process_fwd_kernel_merged,
    pre_process_fwd_kernel_stage1,
)
from fla.utils import device

DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark chunk_delta_h kernels using triton.testing.do_bench"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--heads", type=int, default=32, help="Number of heads (default: 32)"
    )
    parser.add_argument(
        "--headdim", type=int, default=128, help="Head dimension (default: 128)"
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=32768,
        help="Sequence length per CP rank (default: 32768 = 32k)",
    )
    return parser.parse_args()


def create_tensors(B, T, H, K, V, device, dtype):
    """Create input tensors for benchmarking."""
    # Tensors for forward kernels
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    gk = torch.randn(B, T, H, K, device=device, dtype=torch.float32)

    # cu_seqlens for 1 chunk
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

    # hm tensor (zero-initialized)
    hm = torch.zeros(H, K, V + K, device=device, dtype=torch.float32)

    # Tensors for backward kernels
    q = k.clone()
    do = torch.randn(B, T, H, V, device=device, dtype=dtype)
    dv = torch.zeros_like(v)
    dhm = torch.zeros_like(hm)

    # h_out and dht for merge kernels
    h_out = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    dht = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    # ag_hm and ag_dhm for merge kernels (simulating cp=8)
    num_ranks = 8  # Simulating cp=8
    stride = H * K * (K + V)
    ag_hm = torch.zeros(num_ranks * stride, device=device, dtype=torch.float32)
    ag_dhm = torch.zeros(num_ranks * stride, device=device, dtype=torch.float32)

    return {
        "k": k,
        "v": v,
        "w": w,
        "gk": gk,
        "hm": hm,
        "cu_seqlens": cu_seqlens,
        "q": q,
        "do": do,
        "dv": dv,
        "dhm": dhm,
        "h_out": h_out,
        "dht": dht,
        "ag_hm": ag_hm,
        "ag_dhm": ag_dhm,
    }


def benchmark_all_kernels(B, T, H, K, V):
    """Benchmark all 6 kernels from generate_chunk_delta_h_cache()."""
    print(f"\n{'=' * 80}")
    print("Benchmarking chunk_delta_h kernels (using triton.testing.do_bench)")
    print(f"{'=' * 80}")
    print("Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Heads: {H}")
    print(f"  Head dim (K): {K}")
    print(f"  Head dim (V): {V}")
    print(f"  SeqLen per rank: {T}")
    print(f"  Dtype: {DTYPE}")
    print(f"  Device: {device}")
    print(f"{'=' * 80}\n")

    # Create tensors
    tensors = create_tensors(B, T, H, K, V, device, DTYPE)
    BT = 64
    BK = triton.next_power_of_2(K)

    results = {}
    quantiles = [0.5, 0.2, 0.8]  # median, 20th percentile, 80th percentile

    # ========================================
    # 1. pre_process_fwd_kernel_stage1 (FWD)
    # ========================================
    print("[1/6] Benchmarking pre_process_fwd_kernel_stage1 (FWD)...")

    def grid_stage1(meta):
        return (triton.cdiv(V, meta["BV"]), H)

    def kernel_fwd_stage1():
        pre_process_fwd_kernel_stage1[grid_stage1](
            k=tensors["k"],
            v=tensors["v"],
            w=tensors["w"],
            g=None,
            gk=tensors["gk"],
            hm=tensors["hm"],
            cu_seqlens=tensors["cu_seqlens"],
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            USE_G=False,
            USE_GK=True,
            USE_EXP2=True,
            IS_VARLEN=True,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_fwd_stage1, quantiles=quantiles)
    results["pre_process_fwd_kernel_stage1"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # ========================================
    # 2. pre_process_fwd_bwd_kernel_stage2 (FWD)
    # ========================================
    print("[2/6] Benchmarking pre_process_fwd_bwd_kernel_stage2 (FWD)...")

    def grid_stage2(meta):
        return (triton.cdiv(K, meta["BK2"]), H)

    def kernel_fwd_stage2():
        pre_process_fwd_bwd_kernel_stage2[grid_stage2](
            k=tensors["k"],
            w=tensors["w"],
            g=None,
            gk=tensors["gk"],
            hm=tensors["hm"],
            cu_seqlens=tensors["cu_seqlens"],
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_G=False,
            USE_GK=True,
            USE_EXP2=True,
            IS_VARLEN=True,
            FORWARD=True,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_fwd_stage2, quantiles=quantiles)
    results["pre_process_fwd_bwd_kernel_stage2 (FWD)"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # ========================================
    # 2b. pre_process_fwd_kernel_merged (FWD)
    # ========================================
    print("[2b/6] Benchmarking pre_process_fwd_kernel_merged (FWD)...")

    BLOCK_SIZE = 32 if K <= 64 else 64
    grid_merged = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)

    def kernel_fwd_merged():
        pre_process_fwd_kernel_merged[grid_merged](
            k=tensors["k"],
            v=tensors["v"],
            w=tensors["w"],
            g=None,
            gk=tensors["gk"],
            hm=tensors["hm"],
            cu_seqlens=tensors["cu_seqlens"],
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_G=False,
            USE_GK=True,
            USE_EXP2=True,
            IS_VARLEN=True,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_fwd_merged, quantiles=quantiles)
    results["pre_process_fwd_kernel_merged (FWD)"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # Speedup comparison
    stage1_time = results["pre_process_fwd_kernel_stage1"]
    stage2_time = results["pre_process_fwd_bwd_kernel_stage2 (FWD)"]
    merged_time = results["pre_process_fwd_kernel_merged (FWD)"]
    original_total = stage1_time + stage2_time
    speedup = original_total / merged_time if merged_time > 0 else 0
    print(f"\n    [Speedup] Merged kernel vs Split kernels: {speedup:.2f}x")
    print(f"              Original (stage1+stage2): {original_total * 1000:.2f} us")
    print(f"              Merged: {merged_time * 1000:.2f} us")

    # ========================================
    # 3. merge_fwd_bwd_kernel (FWD)
    # ========================================
    print("[3/6] Benchmarking merge_fwd_bwd_kernel (FWD)...")

    def grid_merge(meta):
        return (triton.cdiv(V, meta["BV"]), H)

    def kernel_merge_fwd():
        merge_fwd_bwd_kernel[grid_merge](
            h=tensors["h_out"],
            ag_hm=tensors["ag_hm"],
            pre_or_post_num_ranks=1,
            rank=1,
            H=H,
            K=K,
            V=V,
            BK=BK,
            FORWARD=True,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_merge_fwd, quantiles=quantiles)
    results["merge_fwd_bwd_kernel (FWD)"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # ========================================
    # 4. pre_process_bwd_kernel_stage1 (BWD)
    # ========================================
    print("[4/6] Benchmarking pre_process_bwd_kernel_stage1 (BWD)...")

    def kernel_bwd_stage1():
        pre_process_bwd_kernel_stage1[grid_stage1](
            q=tensors["q"],
            k=tensors["k"],
            w=tensors["w"],
            g=None,
            gk=tensors["gk"],
            do=tensors["do"],
            dhm=tensors["dhm"],
            dv=tensors["dv"],
            cu_seqlens=tensors["cu_seqlens"],
            scale=1.0,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            USE_G=False,
            USE_GK=True,
            IS_VARLEN=True,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_bwd_stage1, quantiles=quantiles)
    results["pre_process_bwd_kernel_stage1"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # ========================================
    # 5. pre_process_fwd_bwd_kernel_stage2 (BWD)
    # ========================================
    print("[5/6] Benchmarking pre_process_fwd_bwd_kernel_stage2 (BWD)...")

    def kernel_bwd_stage2():
        pre_process_fwd_bwd_kernel_stage2[grid_stage2](
            k=tensors["k"],
            w=tensors["w"],
            g=None,
            gk=tensors["gk"],
            hm=tensors["dhm"],
            cu_seqlens=tensors["cu_seqlens"],
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_G=False,
            USE_GK=True,
            USE_EXP2=True,
            IS_VARLEN=True,
            FORWARD=False,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_bwd_stage2, quantiles=quantiles)
    results["pre_process_fwd_bwd_kernel_stage2 (BWD)"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # ========================================
    # 5b. pre_process_bwd_kernel_merged (BWD)
    # ========================================
    print("[5b/6] Benchmarking pre_process_bwd_kernel_merged (BWD)...")

    BLOCK_SIZE_BWD = 32 if K <= 64 else 64
    grid_bwd_merged = (triton.cdiv(V + K, BLOCK_SIZE_BWD), H)

    def kernel_bwd_merged():
        pre_process_bwd_kernel_merged[grid_bwd_merged](
            q=tensors["q"],
            k=tensors["k"],
            w=tensors["w"],
            g=None,
            gk=tensors["gk"],
            do=tensors["do"],
            dhm=tensors["dhm"],
            dv=tensors["dv"],
            cu_seqlens=tensors["cu_seqlens"],
            scale=1.0,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_G=False,
            USE_GK=True,
            USE_EXP2=True,
            IS_VARLEN=True,
            BLOCK_SIZE=BLOCK_SIZE_BWD,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_bwd_merged, quantiles=quantiles)
    results["pre_process_bwd_kernel_merged (BWD)"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # Speedup comparison for backward
    stage1_bwd_time = results["pre_process_bwd_kernel_stage1"]
    stage2_bwd_time = results["pre_process_fwd_bwd_kernel_stage2 (BWD)"]
    merged_bwd_time = results["pre_process_bwd_kernel_merged (BWD)"]
    original_bwd_total = stage1_bwd_time + stage2_bwd_time
    bwd_speedup = original_bwd_total / merged_bwd_time if merged_bwd_time > 0 else 0
    print(f"\n    [Speedup] BWD Merged kernel vs Split kernels: {bwd_speedup:.2f}x")
    print(f"              Original (stage1+stage2): {original_bwd_total * 1000:.2f} us")
    print(f"              Merged: {merged_bwd_time * 1000:.2f} us")

    # ========================================
    # 6. merge_fwd_bwd_kernel (BWD)
    # ========================================
    print("[6/6] Benchmarking merge_fwd_bwd_kernel (BWD)...")

    def kernel_merge_bwd():
        merge_fwd_bwd_kernel[grid_merge](
            h=tensors["dht"],
            ag_hm=tensors["ag_dhm"],
            pre_or_post_num_ranks=1,
            rank=1,
            H=H,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(kernel_merge_bwd, quantiles=quantiles)
    results["merge_fwd_bwd_kernel (BWD)"] = ms
    print(f"      Median: {ms * 1000:.2f} us  (min: {min_ms * 1000:.2f} us, max: {max_ms * 1000:.2f} us)")

    # Print summary
    print(f"\n{'=' * 80}")
    print("Benchmark Summary")
    print(f"{'=' * 80}")
    print(f"{'Kernel Name':<50} {'Time (us)':<15} {'% of Total':<12}")
    print("-" * 80)

    total_time = sum(results.values())
    for name, t in results.items():
        percentage = (t / total_time * 100) if total_time > 0 else 0
        print(f"{name:<50} {t * 1000:<15.2f} {percentage:<12.1f}")

    print("-" * 80)
    print(f"{'Total':<50} {total_time * 1000:<15.2f} {'100.0':<12}")
    print(f"{'=' * 80}\n")

    # Breakdown by direction (using split kernels for fair comparison)
    fwd_time = (
        results["pre_process_fwd_kernel_stage1"]
        + results["pre_process_fwd_bwd_kernel_stage2 (FWD)"]
        + results["merge_fwd_bwd_kernel (FWD)"]
    )
    bwd_time = (
        results["pre_process_bwd_kernel_stage1"]
        + results["pre_process_fwd_bwd_kernel_stage2 (BWD)"]
        + results["merge_fwd_bwd_kernel (BWD)"]
    )

    print(f"Forward pass total:  {fwd_time * 1000:.2f} us ({fwd_time/total_time*100:.1f}%)")
    print(f"Backward pass total: {bwd_time * 1000:.2f} us ({bwd_time/total_time*100:.1f}%)")
    print(f"FWD/BWD ratio: {fwd_time/bwd_time:.2f}")
    print()

    return results


def main():
    args = get_args()

    B = args.batch
    H = args.heads
    K = args.headdim
    V = args.headdim  # Assuming V = K
    T = args.seqlen

    # Print environment info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    # Run benchmark
    benchmark_all_kernels(B, T, H, K, V)


if __name__ == "__main__":
    main()
