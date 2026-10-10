---
name: fla-nvidia-performance
description: Profile and optimize FLA kernels on NVIDIA GPUs, with same-hardware benchmarks and targeted Nsight Compute analysis.
---

# NVIDIA kernel performance

Use this skill for NVIDIA kernel optimization and backend tuning. Follow [CONTRIBUTING](../../../CONTRIBUTING.md#benchmarking) for correctness and performance evidence, and [fla-optimization-loop](../fla-optimization-loop/SKILL.md) for sustained iteration.

## Hardware baseline

Use datacenter NVIDIA GPUs with sm_90 or newer for PR performance conclusions; H100 and H20 are accepted, and sm_100/sm_103 are preferred. Results from A100, older GPUs, or consumer cards are supplementary. Record the exact GPU and software versions with each comparison.

## Workflow

1. Identify the changed kernel, its callers, and the production workloads it affects. Record a baseline commit and confirm which backend runs.
2. Pass the relevant correctness tests before timing a candidate. Preserve the reference and tolerances; include outputs, states, and supported gradients.
3. Compare baseline and candidate on the same GPU with identical workloads and dispatch settings. Include dense and supported variable-length cases. Use an end-to-end benchmark for a training or generation throughput claim.
4. Use Nsight Compute when a bottleneck or regression needs hardware metrics. Match the optimization to the measured limit, then rerun correctness and timing.
5. Report latency or throughput, relevant memory changes, and any regressions. Keep raw profiler artifacts outside git, for example under `profile/<run_name>/`.

## Benchmark commands

Run from the repository root. Resolve the baseline to a commit SHA before comparing:

```bash
FLA_BENCH_BASE=$(git rev-parse origin/main)
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_kda --base "$FLA_BENCH_BASE"

python benchmarks/benchmark_training_throughput.py --name kda --batch_size 2 --seq_len 8192
python benchmarks/benchmark_training_throughput.py --name kda --batch_size 2 --seq_len 8192 --varlen
```

The verification command runs the current tests; retain the validated baseline and tests yourself. A selected subset is useful during iteration but does not replace the full relevant gate before reporting a gain.

## Nsight Compute

Collect a representative changed kernel with the installed `ncu`. Check which sections and metrics that version supports.

The benchmark runner launches a child process, so use `--target-processes all`. Replace the kernel filter with the changed kernel's name. `--no-base` profiles the current implementation; collect the baseline separately with the same command and workload.

```bash
mkdir -p profile/kda
ncu --target-processes all --set full \
  -k 'regex:chunk_kda' -c 1 -o profile/kda/full \
  python -m benchmarks.ops.run --op chunk_kda --modes fwd --no-base

ncu --target-processes all --set source --section SourceCounters \
  -k 'regex:chunk_kda' -c 1 -o profile/kda/source \
  python -m benchmarks.ops.run --op chunk_kda --modes fwd --no-base
```

Use duration, memory throughput, occupancy, stalls, and hot instructions to explain the bottleneck. Profiling overhead makes these runs unsuitable as the latency comparison; use the synchronized benchmark for that measurement.
