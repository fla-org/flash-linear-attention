---
name: fla-ascend-performance
description: Profile and optimize FLA Triton-Ascend kernels using NPU traces, with guidance for UB capacity, memory movement, launch limits, and numerical correctness.
---

# Ascend kernel performance

Use this skill for operator performance work under `triton_ascend` backend directories. Follow [fla-optimization-loop](../fla-optimization-loop/SKILL.md) for sustained optimization and [CONTRIBUTING](../../../CONTRIBUTING.md#benchmarking) for validation and reporting.

Keep collection, analysis, and benchmarks in the active Python/NPU environment. If it is not configured, activate the project's Ascend environment first.

## 1. Establish the baseline

Locate the public entry, dispatch route, and Ascend kernel. Record the supported inputs and production workloads, baseline commit, and synchronized latency. Preserve the public API, validated algorithm, reference, and tolerances. Confirm the intended NPU kernel runs; a Torch fallback must not hide an unsupported path or kernel bug.

## 2. Collect a trace

Use the existing collector instead of duplicating profiler setup. Run from the repository root; the workload file initializes its inputs and defines a repeatable `workload()` callable.

```bash
python .agents/skills/fla-ascend-performance/scripts/profile_npu.py \
  --name my_op --out-dir profile/my_op-npu \
  --metrics PipeUtilization --analyze \
  --kernel-filter my_kernel --exec-file path/to/workload.py
```

One run collects one `aic_metrics` set. Start with `PipeUtilization`; collect `MemoryUB` separately when UB bandwidth is relevant. The default schedule is one warmup step and one active step. Warm the workload enough to exclude compilation from the measurement.

The collector prints the exact trace directory. To revisit it, pass that directory to `scripts/analyze_profile.py`; see [collection and metrics](references/reference.md#collection-and-analysis) for the library interface and output layout.

## 3. Diagnose bottlenecks

Use `op_statistic.csv` to find the dominant operation, then `kernel_details.csv` to inspect its duration and pipe activity. Confirm the dispatch route before tuning a small or missing target kernel. Distinguish compilation failure, UB overflow, grid limits, numerical errors, and measured performance limits; they need different fixes.

Use the [diagnosis table](references/reference.md#diagnosis) to choose the next experiment. Change one relevant factor, then measure whether the expected duration and pipe metrics moved.

## 4. Optimize within the kernel contract

- Estimate the peak live UB footprint and use `fla.utils.ascend_ub_manager` for device capacity and tiling. Model forward and backward separately; fuse only when the live set fits.
- Match the launch to the work: Cube kernels use Cube cores, Vector kernels use Vector cores. Handle the grid limit through the existing helpers or a 1D task loop.
- Preserve numerical precision, optional inputs, and fixed/variable-length behavior. Verify tail handling and pointer arithmetic before increasing tiles or fusing stages.
- Omit `num_warps` and `num_stages` from Ascend launches and autotune settings; these are unsupported on this backend.
- Check [compiler and memory traps](references/TRAPS.md) when changing DMA paths, pointer arithmetic, or reused `tl.dot` operands. Use [contiguous gate loading](references/g-contiguous-loading.md) when gate loads along time are strided.

[Kernel cases](references/cases.md) explain existing convolution, recurrence, and solve implementations. Their tile choices are workload-specific examples.

## 5. Validate and measure

Run kernel comparisons and the affected operator/layer tests, including supported gradients, variable lengths, optional inputs, and boundary shapes. Keep NaN poisoning enabled and confirm that dispatch exercises the changed path. Run the full relevant correctness gate before reporting a speedup; a `--gate-k` subset is an iteration check.

Compare synchronized forward and forward/backward latency on the same NPU, then repeat the relevant profiler collection. Report the timing change, the metrics that explain it, any compiler limitation encountered, and untested cases. If the expected metrics do not change, revisit the diagnosis before adding another optimization.
