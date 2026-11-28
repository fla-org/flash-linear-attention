# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Benchmark chunk_gated_delta_product forward, backward, fwd+bwd latency and peak VRAM."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.utils import device, device_torch_lib


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def _machine_info() -> dict:
    info = {
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'triton_version': triton.__version__,
        'cuda_version': torch.version.cuda or 'N/A',
        'git_branch': _git_output('rev-parse', '--abbrev-ref', 'HEAD'),
        'git_commit': _git_output('rev-parse', '--short', 'HEAD'),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(props.total_memory / (1024 ** 3), 2)
    else:
        info['gpu_name'] = 'N/A'
        info['gpu_memory_gb'] = 0.0
    return info


def _percentile(sorted_vals: list[float], p: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    p25 = _percentile(ordered, 0.25)
    p75 = _percentile(ordered, 0.75)
    return {
        'samples_ms': [round(x, 6) for x in samples],
        'median_ms': round(statistics.median(ordered), 6),
        'min_ms': round(ordered[0], 6),
        'max_ms': round(ordered[-1], 6),
        'p25_ms': round(p25, 6),
        'p75_ms': round(p75, 6),
        'iqr_ms': round(p75 - p25, 6),
    }


def benchmark_case(
    B: int,
    T: int,
    H: int,
    D: int,
    num_householder: int,
    dtype: torch.dtype,
    use_g: bool,
    rounds: int,
    warmup_iters: int,
    warmup_ms: int,
    rep_ms: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, dtype=dtype, device=device), p=2, dim=-1).requires_grad_(True)
    k = F.normalize(
        torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device), p=2, dim=-1,
    ).requires_grad_(True)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device)).requires_grad_(True) if use_g else None
    do = torch.randn(B, T, H, D, dtype=dtype, device=device)

    grad_tensors = [t for t in (q, k, v, g, beta) if t is not None]

    def clear_grads() -> None:
        for t in grad_tensors:
            t.grad = None

    def run_fwd():
        return chunk_gated_delta_product(q, k, v, g, beta, num_householder=num_householder)[0]

    def run_fwdbwd():
        out = chunk_gated_delta_product(q, k, v, g, beta, num_householder=num_householder)[0]
        out.backward(do)

    # Warm up kernels and Triton autotuning before measuring memory or latency.
    for _ in range(warmup_iters):
        clear_grads()
        run_fwdbwd()
    clear_grads()
    device_torch_lib.synchronize()

    # Measure peak allocated VRAM for a single fwd+bwd pass.
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        run_fwdbwd()
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        clear_grads()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        base_bytes = 0
        peak_bytes = 0

    fwd_samples: list[float] = []
    bwd_samples: list[float] = []
    fwdbwd_samples: list[float] = []

    for _ in range(rounds):
        fwd_ms = float(
            triton.testing.do_bench(
                run_fwd,
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=[0.5, 0.2, 0.8],
            )[0]
        )
        fwd_samples.append(fwd_ms)

        clear_grads()
        out_saved = run_fwd()
        bwd_ms = float(
            triton.testing.do_bench(
                lambda out=out_saved: out.backward(do, retain_graph=True),
                grad_to_none=grad_tensors,
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=[0.5, 0.2, 0.8],
            )[0]
        )
        out_saved = None
        clear_grads()
        bwd_samples.append(bwd_ms)

        fwdbwd_ms = float(
            triton.testing.do_bench(
                run_fwdbwd,
                grad_to_none=grad_tensors,
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=[0.5, 0.2, 0.8],
            )[0]
        )
        clear_grads()
        fwdbwd_samples.append(fwdbwd_ms)

    return {
        'B': B,
        'T': T,
        'H': H,
        'D': D,
        'num_householder': num_householder,
        'use_g': use_g,
        'dtype': str(dtype).replace('torch.', ''),
        'fwd': _summarize(fwd_samples),
        'bwd': _summarize(bwd_samples),
        'fwdbwd': _summarize(fwdbwd_samples),
        'base_vram_mib': round(base_bytes / (1024 ** 2), 2),
        'peak_vram_mib': round(peak_bytes / (1024 ** 2), 2),
        'peak_vram_delta_mib': round((peak_bytes - base_bytes) / (1024 ** 2), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-B', '--batch-size', type=int, default=2)
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[2048, 4096, 8192, 16384])
    parser.add_argument('-H', '--num-heads', type=int, default=16)
    parser.add_argument('--head-dims', type=int, nargs='+', default=[64])
    parser.add_argument('--num-householders', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--no-g', action='store_true', help='Disable forget gate g')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--warmup-iters', type=int, default=5)
    parser.add_argument('--warmup-ms', type=int, default=25)
    parser.add_argument('--rep-ms', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    use_g = not args.no_g
    machine = _machine_info()

    print(
        f"GPU: {machine['gpu_name']} ({machine['gpu_memory_gb']} GB) | "
        f"PyTorch {machine['pytorch_version']} | Triton {machine['triton_version']} | "
        f"CUDA {machine['cuda_version']} | {machine['git_branch']}[{machine['git_commit']}]",
        flush=True,
    )
    print(
        f"{'B':>2} {'T':>6} {'H':>3} {'D':>4} {'n':>2} | "
        f"{'fwd(ms)':>10} {'bwd(ms)':>10} {'fwdbwd(ms)':>11} {'fwdbwd IQR':>11} {'peak(MiB)':>10}",
        flush=True,
    )

    results = []
    for D in args.head_dims:
        for T in args.seq_lens:
            for n in args.num_householders:
                row = benchmark_case(
                    B=args.batch_size,
                    T=T,
                    H=args.num_heads,
                    D=D,
                    num_householder=n,
                    dtype=dtype,
                    use_g=use_g,
                    rounds=args.rounds,
                    warmup_iters=args.warmup_iters,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                    seed=args.seed,
                )
                results.append(row)
                print(
                    f"{row['B']:>2} {row['T']:>6} {row['H']:>3} {row['D']:>4} {row['num_householder']:>2} | "
                    f"{row['fwd']['median_ms']:>10.3f} {row['bwd']['median_ms']:>10.3f} "
                    f"{row['fwdbwd']['median_ms']:>11.3f} {row['fwdbwd']['iqr_ms']:>11.3f} "
                    f"{row['peak_vram_mib']:>10.1f}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

    report = {
        'machine_info': machine,
        'config': {
            'B': args.batch_size,
            'seq_lens': args.seq_lens,
            'H': args.num_heads,
            'head_dims': args.head_dims,
            'num_householders': args.num_householders,
            'dtype': args.dtype,
            'use_g': use_g,
            'rounds': args.rounds,
            'warmup_iters': args.warmup_iters,
            'warmup_ms': args.warmup_ms,
            'rep_ms': args.rep_ms,
            'seed': args.seed,
        },
        'results': results,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
