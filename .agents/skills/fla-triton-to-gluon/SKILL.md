---
name: fla-triton-to-gluon
description: Design and optimize FLA Gluon kernels using explicit layouts, memory movement, and scheduling, including ports from Triton and improvements to existing Gluon implementations.
---

# Triton to Gluon

Use this skill when explicit control over data ownership, communication, or scheduling can address a measured kernel bottleneck. Preserve the validated numerical algorithm and precision. A literal translation can establish parity; the optimization task also needs a reasoned execution design and evidence that it addresses the bottleneck. An already efficient kernel may have no useful Gluon improvement.

Follow [fla-optimization-loop](../fla-optimization-loop/SKILL.md) for the frozen baseline, correctness gate, and experiment log, and [fla-nvidia-performance](../fla-nvidia-performance/SKILL.md) for NVIDIA measurements. Keep the portable fallback and use [fla-dispatch-backends](../fla-dispatch-backends/SKILL.md) when changing routing. Contribution and numerical-change policies remain in [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Establish the target and read the relevant sources

Record the actual execution worker's GPU architecture, Triton version/source revision, supported shapes and dtypes, and selected backend. Check installed APIs before using an upstream example; Gluon is experimental and online `main` can differ from the installed release.

Use the [source map](references/sources-and-evidence.md#select-sources-by-the-decision) to read the relevant official tutorial implementation and API documentation before designing that part of the kernel. Record the source and the idea being adapted in the experiment log. Reuse source checks within an unchanged environment; do not read every tutorial for every kernel.

Translate the profile into a specific hypothesis, such as keeping repeated reductions within a warp, removing an expensive layout conversion, reducing address-register pressure, or overlapping an operand transfer with independent computation. Identify the generated-code or hardware signal expected to change.

## Design the execution

Read [layout and dataflow design](references/layout-and-dataflow.md) when choosing or changing tensor ownership. Trace important tensors from their producers to their final consumers, including loads, reductions/scans, MMA, loop-carried state, and stores. Choose register and shared-memory layouts together, and account for conversion costs and live resources across the whole path.

Read [pipelines and scheduling](references/pipelines-and-scheduling.md) before introducing async transfers, architecture-specific MMA, persistent execution, or warp specialization. Draw the relevant producer/consumer timeline and budget its buffers and registers before implementing it. For an existing async kernel, audit that timeline before changing its layouts or lifetimes.

Compare plausible designs for the measured bottleneck. A small reduction kernel may benefit from ownership changes alone; a matrix-heavy kernel may need coordinated load, MMA, and epilogue pipelines. Explain why a relevant alternative is selected or ruled out. Do not require TMA, warp specialization, TMEM, or any fixed number of candidates when the workload does not justify them.

Keep a compact design record with the chosen ownership, costly communication, live-memory budget, intended overlap, and expected measurement. Preserve mathematical dependencies while reconsidering tile boundaries, loop organization, register reuse, and work partitioning inherited from the Triton implementation.

## Implement and measure incrementally

1. If useful for debugging, establish a synchronous Gluon parity checkpoint with explicit layouts and complete masks. Compare outputs, states, and supported gradients before adding asynchronous execution.
2. Implement one design change at a time. Keep layout parameters, launch warp counts, and autotune pruning consistent through explicit constexpr choices. Gate hardware-specific instructions and descriptor eligibility on the actual target and inputs.
3. Prune configurations using peak live registers, shared memory, and TMEM where applicable. Include conversion scratch space, barriers, epilogue buffers, and allocation granularity. Bound static unrolling and autotune breadth to keep compilation practical; reuse the worker and compiler cache.
4. Run the frozen correctness gate and benchmark passing candidates. Inspect the [generated-code evidence](references/sources-and-evidence.md#verify-the-execution-design) relevant to the hypothesis, then keep or reject the change based on correctness, resource use, and repeatable performance.

A supported shape that cannot fit a resident design needs a valid streaming path or fallback. Preserve tail masking, address widths, initialization, and exact-zero semantics throughout the port. Synchronization and memory-proxy obligations are detailed in the pipeline reference.

## Validate and report

Run from the repository root on the target GPU, with `FLA_BENCH_BASE` set to the baseline SHA recorded before optimization. Reuse that SHA rather than resolving a moving branch again:

```bash
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_kda --base "$FLA_BENCH_BASE"
```

Replace the operation with the one being optimized. The command executes the current tests; it does not freeze them. Use `--gate-k` only for quick iteration, then run the full relevant gate. Set backend flags before launching a fresh process and confirm the intended implementation executed. Measure production entry points as well as individual kernels.

Report which explicit control changed execution, the source informing it, the relevant generated-code/profile evidence, and per-workload results including regressions. Include supported variable-length and boundary cases. A translation or a losing candidate alone does not establish a performance limit; apply the optimization loop's no-win criteria and state unresolved validation separately.
