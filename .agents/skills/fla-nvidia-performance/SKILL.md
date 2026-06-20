---
name: fla-nvidia-performance
description: >
  Guidelines for NVIDIA GPU kernel / Triton / Gluon / TileLang / CUDA backend performance work in the FLA repo.
  Covers profiling workflow, hardware baselines, and MR-ready performance evidence requirements.
  Uses an installed ncu-report-skill when a task needs detailed Nsight Compute collection and diagnosis.
---

# FLA NVIDIA Performance Skill

Use this skill when working on Triton, Gluon, TileLang, CUDA, or other NVIDIA GPU kernel optimizations,
backend tuning, or any change that could affect throughput or latency in `fla/ops/` or related modules.

## Hardware baseline

- **Minimum effective baseline**: datacenter NVIDIA GPUs with sm_90 or newer; H100 / H20 are accepted.
- **Preferred targets**: datacenter NVIDIA GPUs with sm_100 or sm_103.
- **Reference only**: A100 (sm_80), pre-sm_80 GPUs, and all consumer cards (including sm_86, sm_89, and sm_120).
  Do not use these numbers as the only MR performance conclusion.

## Detailed NCU workflow

This repo intentionally does **not** vendor `ncu-report-skill`, add it as a submodule, or auto-clone it during agent work.

If a user-level `ncu-report-skill` is available, use it for:

- Nsight Compute collection details (`full`, `source`, PM sampling, source counters);
- sm_100 / sm_103 metric-name caveats;
- report parsing helpers and diagnosis playbooks;
- the final profiling report structure.

If it is not available, use the minimal NCU commands in this skill and state in the MR notes
that the external helper skill was unavailable.
Do not create untracked external clones inside this repo unless the user explicitly asks.

## Day-to-day development

- You are **not** required to run NCU for every incremental change.
- Quick sanity checks with `benchmark_training_throughput.py` or `benchmark_generation.py`
  are enough to catch large regressions during development.
- Prefer dense workloads for quick iteration; varlen workloads are checked before MR.

## Before opening an MR (performance evidence)

An agent-authored MR that touches kernel code must include **complete performance evidence**:

1. **Before / after benchmark**
   - Run the same benchmark script with the same workload on the same hardware.
   - Report throughput (tokens/s or iters/s) and, if relevant, peak memory.

2. **NCU profile**
   - Run `ncu` with both `--set full` and `--set source` for a representative changed kernel
     when Nsight Compute is available.
   - Capture the `.ncu-rep` locally; do **not** commit it to the repo.
   - In the MR description, paste a short summary of key metrics
     (e.g., memory throughput %, SOL, occupancy, top hot instructions).

3. **Workload coverage**
   - Include at least one **dense** workload.
   - Include at least one **variable-length** workload if the op supports it.

4. **Conclusion and risk**
   - State whether the change is an improvement, neutral, or a known trade-off.
   - Flag any backend or shape that regressed and explain why.

## Profile artifact layout

Store local profile artifacts under:

```text
profile/<run_name>/
```

For example:

```text
profile/kda_chunk_bwd_20250603/
  ├── REPORT.md
  ├── reports/
  │   ├── full_<tag>.ncu-rep
  │   └── source_<tag>.ncu-rep
  └── analysis/
```

Keep `.ncu-rep`, `.nsys-rep`, and raw logs out of git.

## Quick commands reference

```bash
# Op microbenchmark
python -m benchmarks.ops.run --op chunk_kda --modes fwd

# Model training benchmark
python benchmarks/benchmark_training_throughput.py \
  --name kda --batch_size 2 --seq_len 8192

# Varlen training benchmark (if supported by the model/op path)
python benchmarks/benchmark_training_throughput.py \
  --name kda --batch_size 2 --seq_len 8192 --varlen

# NCU full profile
ncu --set full --section PmSampling --section PmSampling_WarpStates \
  -k "regex:<kernel_regex>" -c 1 \
  -o profile/<run_name>/reports/full_<tag> \
  python -m benchmarks.ops.run --op chunk_kda --modes fwd

# NCU source profile (for instruction-level analysis)
ncu --set source --section SourceCounters \
  -k "regex:<kernel_regex>" -c 1 \
  -o profile/<run_name>/reports/source_<tag> \
  python -m benchmarks.ops.run --op chunk_kda --modes fwd
```
