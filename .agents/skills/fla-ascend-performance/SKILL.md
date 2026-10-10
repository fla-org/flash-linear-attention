---
name: fla-ascend-performance
description: >
  Profile and optimize FLA Triton-Ascend kernels. Use for NPU performance work,
  UB or grid limits, and Ascend compiler failures in triton_ascend.py or triton_ascend/.
---

# Ascend kernel performance

Use this workflow for Ascend implementations in `triton_ascend.py` and `triton_ascend/`. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) for repository policy and [fla-optimization-loop](../fla-optimization-loop/SKILL.md) for the frozen correctness gate and iteration stop criteria.

## 1. Establish the workload

Trace the public entry point through dispatch to the Ascend implementation. Record the supported layouts, dtypes, head mappings, fixed/variable lengths, states, and gradients. Confirm that the target NPU kernel runs; a fallback can hide both a missing implementation and its performance cost.

Freeze the reference, test cases, tolerances, and benchmark shapes before tuning. Measure a synchronized baseline with warmup and repeated timings. Keep the public API and supported calls intact.

Use the active Python/NPU environment for profiling, analysis, and benchmarks. If it is not configured, activate the project's Ascend environment first and retain that environment throughout the comparison.

## 2. Collect a profile

Use the bundled collector instead of copying profiler setup into each workload. Run from the repository root and keep traces under the ignored `profile/` directory:

```bash
python .agents/skills/fla-ascend-performance/scripts/profile_npu.py \
  --name my_op \
  --out-dir profile/my_op-npu \
  --metrics PipeUtilization \
  --analyze \
  --kernel-filter my_kernel_substr \
  --exec-file path/to/workload_only.py
```

The workload file defines `workload()` with the operator call and backward pass when relevant. Collect one `aic_metrics` per run: start with `PipeUtilization`, then use `MemoryUB` if UB bandwidth needs investigation. See [profiling options and outputs](references/reference.md#profiling).

## 3. Identify the limiting resource

Use `op_statistic.csv` to find the operator's share of total time, then sort `kernel_details.csv` by duration. Check the dominant kernel's Cube, Vector, scalar, MTE, and UB metrics against the [diagnosis table](references/reference.md#diagnose-the-bottleneck).

Classify compile failures, UB overflow, grid limits, and numerical errors separately from performance bottlenecks. Fix those failures before interpreting latency. Choose one measurable hypothesis for the next round.

## 4. Make a targeted change

- **UB capacity:** estimate peak live tiles and use `fla.utils.ascend_ub_manager` for the tile budget. Model forward and backward separately; split stages when fusion cannot fit.
- **Memory traffic:** check contiguous loads, reuse, and intermediate writebacks. For token-axis gate gathers, use the [gate layout guide](references/g-contiguous-loading.md).
- **Grid overhead:** choose host chunking or a one-dimensional core grid. Match the core count to the limiting pipe and preserve variable-length offsets.
- **Compute or scalar work:** adjust tiles, layout, specialization, or fusion according to the measured bottleneck.

Ascend launch tuning does not use `num_warps` or `num_stages`; omit them from launches, wrappers, and autotune configurations. Preserve the established precision and numerical algorithm. Shape-specific tile sizes and memory multipliers from earlier experiments are starting points, not general rules.

Before changing address math, DMA paths, or reused `tl.dot` inputs, read the relevant [correctness hazards](references/reference.md#correctness-and-compiler-hazards). Detailed tuning choices are in [the reference](references/reference.md#choose-a-tuning-change).

## 5. Validate and decide

1. Compare the changed kernel with its reference, including outputs, states, and supported gradients.
2. Cover small and large sequences, tile boundaries, head sharing, optional gates/states, and fixed/variable lengths. Confirm dispatch reaches Ascend.
3. Run the frozen full correctness gate with NaN poisoning before accepting a candidate. A filtered test run or `--no-gate` benchmark does not establish correctness.
4. Repeat the synchronized benchmark and profile with the same workload and metrics. If the expected metric does not improve, revisit the diagnosis before making another change.

Report before/after latency, the target kernel's duration share, relevant pipe or UB metrics, and any compiler workaround. Use [fla-pr-readiness](../fla-pr-readiness/SKILL.md) to package the evidence.

## References by task

| Task                              | Reference                                                          |
| --------------------------------- | ------------------------------------------------------------------ |
| Collect and interpret a profile   | [Profiling and tuning reference](references/reference.md)          |
| Fix compiler or addressing errors | [Compiler traps](references/TRAPS.md)                              |
| Change token-axis gate loads      | [Contiguous gate loading](references/g-contiguous-loading.md)      |
| Compare with a previous kernel    | [Kernel case notes](references/cases.md)                           |
