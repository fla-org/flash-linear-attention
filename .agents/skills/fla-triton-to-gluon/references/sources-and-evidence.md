# Sources and execution evidence

Use this reference to select source material before a design change and to check whether the implementation produced the intended execution. Keep the sources, hypotheses, and results in the optimization log defined by [fla-optimization-loop](../../fla-optimization-loop/SKILL.md).

## Select sources by the decision

Start with the actual worker's installed Triton package and the matching release/tag or source revision when available. The online [tutorial index](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html) and [language API](https://triton-lang.org/main/gluon/api/language.html) describe upstream `main`. Check signatures, defaults, architecture requirements, and synchronization semantics locally before adapting them. If a feature is unavailable, document the limitation and select a supported design; do not silently upgrade the target environment.

| Decision                         | Read the official implementation                                                                                       | Extract for this workload                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Ownership and communication      | [Tensor Layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)                            | Mapping, conversions, vector access, reduction/scan communication        |
| Shared staging and prefetch      | [Async Copy](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html)                             | Shared layout, commit groups, buffering, prologue and drain              |
| Descriptor-based movement        | [TMA](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html)                                           | Addressing cost, eligibility, barriers, proxy fences, store completion   |
| Hopper tensor-core execution     | [WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html)                                       | Operand placement, instruction shape, accumulator dependencies           |
| Blackwell tensor-core execution  | [tcgen05](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html)                                   | TMEM layout/lifetime, operand placement, completion                      |
| Scheduling across output tiles   | [Persistent Kernels](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html)                    | Tile ordering, L2 reuse, epilogue overlap, peak live storage             |
| Concurrent warp roles            | [Warp Specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html)           | Partition work, warp/register budgets, cross-partition handoffs          |

Read the code and the experiment explaining the choice, not just the API signature. Note what transfers to FLA and what depends on the tutorial's shapes, precision, and device. Preserve FLA's masks, address widths, supported inputs, and numerical contract when adapting examples.

Consult the current index for capabilities beyond these common paths, such as TMA gather/scatter, TMEM copy, or multi-CTA execution, when an identified bottleneck warrants them. Verify the installed feature and target support first. This map is a starting point, not a requirement to use every hardware feature.

## Verify the execution design

Keep a representative compiled specialization and inspect its available IR, PTX/SASS, and resource metadata. Depending on the installed backend, these can be exposed through the compiled kernel's `asm` and metadata fields; check available keys rather than assuming every artifact exists. Use the NVIDIA performance skill for targeted profiler collection.

Tie the evidence to the hypothesis:

| Intended change                 | Evidence to inspect                                                                                       | Cost that could erase the gain                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| More efficient global accesses  | Load/store width, transaction count, useful bytes per transaction, address instructions                   | More registers or weaker parallelism                                |
| Less layout/reduction exchange  | Conversion lowering, shuffle/shared instructions, synchronization, shared scratch use                     | Worse global coalescing or longer-lived values                      |
| Better shared-memory layout     | Shared instruction shape and target bank-conflict measurements                                            | Padding footprint or incompatibility with another consumer          |
| More effective async overlap    | Issue/wait placement plus relevant dependency stalls and throughput                                       | Buffer growth, producer overhead, drain cost, occupancy loss        |
| Better MMA/epilogue combination | Actual MMA instructions, utilization, accumulator movement, epilogue work                                 | Conversions or stores dominating the complete tile                  |
| Better persistent/warp schedule | Register/SMEM/TMEM allocation where exposed, spills, occupancy, L2 reuse, load balance and barrier stalls | Extra resident warps, lost locality, synchronization                |

An instruction's presence does not demonstrate useful overlap, and high occupancy alone does not demonstrate speed. Reconcile the generated code with the measured bottleneck. If the intended lowering did not occur, resolve that before interpreting the timing as a test of the design. State unavailable evidence and use the strongest supported measurement without claiming an unobserved mechanism.

Compare a passing candidate against the frozen baseline with identical hardware, workloads, flags, and timing method. Use focused ablations when several mechanisms changed together and attribution matters. Inspect regressions at fallback boundaries and in short or tail-heavy workloads, where setup and drain costs can dominate. A tutorial's speedup on its own device is not evidence for an FLA workload.

For synchronization changes, supplement the numerical gate with the target's supported race/synchronization tooling when available and relevant. Exercise buffer reuse and partial pipelines explicitly; a passing numerical sample alone does not prove ordering. Record unsupported tools or unrun checks accurately.
