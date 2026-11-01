---
name: fla-triton-to-gluon
description: Design and optimize FLA Gluon kernels using explicit layouts, memory movement, and scheduling, including ports from Triton and improvements to existing Gluon implementations.
---

# Triton to Gluon

Use this skill when explicit control over data ownership, communication, or scheduling can address a measured kernel bottleneck. Preserve the validated numerical algorithm and precision. A literal translation can establish parity; an optimization also needs a reasoned execution design and evidence that it addresses the bottleneck.

Follow [fla-optimization-loop](../fla-optimization-loop/SKILL.md) for the frozen baseline, correctness gate, and experiment log, and [fla-nvidia-performance](../fla-nvidia-performance/SKILL.md) for NVIDIA measurements. Keep the portable fallback and use [fla-dispatch-backends](../fla-dispatch-backends/SKILL.md) when changing routing. Contribution and numerical-change policies remain in [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Diagnose and read

Record the actual execution worker's GPU architecture, Triton version/source revision, supported shapes and dtypes, and selected backend. Translate the profile into a specific hypothesis: which communication, memory movement, or dependency limits performance, and which explicit control could change it?

Before designing that part of the kernel, read its relevant official tutorial implementation and API documentation. Use the table to select sections of the execution-design reference; unrelated sections can stay unloaded.

| Design decision                                | Guidance                                                                    | Official tutorial                                                                                                                                                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tensor ownership and communication             | [Ownership](references/execution-design.md#ownership-and-communication)     | [Tensor Layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)                                                                                                                       |
| Register loads, shared staging, or descriptors | [Storage and movement](references/execution-design.md#storage-and-movement) | [Async Copy](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html), [TMA](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html)                                          |
| Tensor-core operands and accumulators          | [MMA](references/execution-design.md#mma-operands-and-accumulators)         | [WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html), [tcgen05](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html)                                            |
| Overlap, buffer ownership, and synchronization | [Pipelines](references/execution-design.md#pipelines-and-synchronization)   | [Async Copy](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html), [TMA](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html)                                          |
| Persistent execution or concurrent warp roles  | [Scheduling](references/execution-design.md#scheduling)                     | [Persistent Kernels](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html), [Warp Specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html) |

Gluon is experimental: the online [language API](https://triton-lang.org/main/gluon/api/language.html) and [tutorial index](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html) describe upstream `main`. Prefer sources matching the installed release/revision and check signatures, defaults, and hardware requirements locally. Record the source and the idea being adapted; reuse those checks within an unchanged environment. If a feature is unavailable, select a supported design without silently upgrading the worker.

Consult the current index for further capabilities, such as TMA gather/scatter, TMEM copy, or multi-CTA execution, when the bottleneck warrants them. Read the code and experiments explaining a choice, then separate the transferable idea from the tutorial's shapes, precision, and device.

## Design and implement

Trace important tensors from producers to final consumers. Decide ownership first, then storage and movement, MMA constraints where relevant, and the pipeline/schedule connecting them. Revisit earlier choices when a consumer or resource limit makes them unsuitable. For existing async kernels, audit buffer lifetimes and synchronization before changing layouts.

Keep a compact design record in the optimization log: chosen ownership, costly communication, peak live resources, intended overlap, and the generated-code or hardware signal expected to change. Compare plausible alternatives for the measured bottleneck. Explain relevant exclusions; no fixed number of candidates or mandatory use of TMA, warp specialization, or TMEM is needed.

Implement one design change at a time. A synchronous parity checkpoint is useful when it isolates migration errors; compare outputs, states, and supported gradients before layering on asynchrony. Preserve mathematical dependencies while reconsidering inherited tile boundaries, loop organization, register reuse, and work partitioning.

Keep layout parameters, launch warp counts, and autotune pruning consistent through explicit constexpr choices. Prune from estimated and compiled resource use, and bound static unrolling and autotune breadth to keep compilation practical. Reuse the worker and compiler cache. Preserve masks, address widths, initialization, and exact-zero semantics, and retain a streaming path or fallback when a resident design cannot fit a supported input.

## Verify and report

Run the frozen correctness gate before timing candidates. Inspect a representative compiled specialization's available IR, PTX/SASS, resource metadata, and relevant profiler metrics to test the design hypothesis. Check available `asm` keys and metadata fields in the installed backend. The reference identifies useful evidence alongside each design choice. Instruction presence alone does not establish overlap, and high occupancy alone does not establish speed; explain unavailable evidence without claiming an unobserved mechanism.

Use the recorded baseline SHA for final comparisons. From the repository root on the target GPU, with `FLA_BENCH_BASE` set to that SHA:

```bash
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_kda --base "$FLA_BENCH_BASE"
```

Replace the operation with the one being optimized. This command runs the current tests; it does not freeze them. `--gate-k` is only for quick iteration. Run the full relevant gate, set backend flags before launching a fresh process, and confirm the intended implementation executed. Include supported variable-length and boundary cases, and measure production entry points as well as individual kernels.

Report the explicit control used, its source, execution evidence, and per-workload results including regressions. Use focused ablations when several mechanisms changed and attribution matters. A tutorial's speedup or one losing candidate does not establish an FLA performance limit; apply the optimization loop's no-win criteria and state unresolved validation separately.
