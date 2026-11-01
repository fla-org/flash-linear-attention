# Pipelines and scheduling

Use this reference for explicit asynchronous execution, MMA, persistent scheduling, and warp specialization. Select the relevant official implementations from the [source map](sources-and-evidence.md#select-sources-by-the-decision) and verify their APIs on the execution worker.

## Choose the data path

Compare ordinary register loads/stores, `cp.async` staging, and TMA for the actual producer/consumer path. Count address calculation, masking, intermediate storage, synchronization, and the final consumer access. `cp.async` can overlap global-to-shared transfers with independent work; staging adds a shared-memory access when the consumer needs registers. TMA can reduce per-element addressing pressure but constrains descriptors and shared layouts. Avoid staging data without a reuse, communication, or overlap benefit.

On NVIDIA, `cp.async` requires Ampere or newer, TMA requires Hopper or newer, WGMMA targets Hopper, and tcgen05/TMEM targets Blackwell. Check the exact instruction, dtype, layout, and architecture combination rather than applying a single minimum compute capability to every feature. Generic Gluon layouts do not imply support for operations in its `nvidia` namespace on other backends.

Derive descriptor eligibility from real base alignment, strides, dtype, and tile shape; use the repository's TMA opt-in and fallback rules in [CONTRIBUTING.md](../../../../CONTRIBUTING.md#triton-kernels). Include descriptor setup in entry-point measurements. For irregular inputs that do not fit this path, retain pointer accesses or investigate a supported gather mechanism using the current tutorial index.

## Construct the timeline

Draw the prologue, steady state, and drain around the dependent operations. Identify which future load overlaps which current computation, and which MMA or store remains outstanding. An asynchronous instruction followed immediately by its wait provides no overlap with intervening work. Do not assume a copied Triton `num_stages` setting constructs the intended Gluon pipeline; inspect the explicit schedule and installed compiler behavior.

Use measured or estimated operation latency and the interval between consumers to choose an initial prefetch distance, then bound it by resources and tune it. More stages may reduce stalls while consuming the shared memory or registers needed for occupancy. Include producer issue cost and the critical dependency chain; a latency ratio alone does not establish the optimal depth.

For each ring buffer, record its producer, consumer, ready event, last read, reuse event, index, and barrier phase. Ready means data can be consumed; reusable means every consumer has finished. Operand loads and MMA consumption may need separate progress tracking. Include output stores and accumulator lifetimes in this accounting.

Exercise zero work when supported, a single iteration, fewer iterations than buffers, partial tiles, and multiple ring wraps. Producers and consumers must agree on which iterations exist and which barriers they signal, including predicated tails. Retain masks on exposed outputs.

## Choose MMA and scheduling

Read [WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html) or [tcgen05](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html) for the target. Choose legal instruction and tile shapes, operand locations, and accumulator layouts under the existing precision contract. Include conversion and epilogue costs when comparing choices. A derived operand already in registers can favor a different WGMMA path from an operand arriving through TMA.

Hopper WGMMA requires B in shared memory and accumulates in registers. Thread its returned accumulator through the wait's dependency argument and consume the wait's returned value. Blackwell tcgen05 accumulates in TMEM and signals completion through mbarriers. Check TMEM allocation granularity, capacity, register-load layout, and participating warp constraints; count those resources separately from SMEM and registers. Initialize accumulators explicitly, including `use_acc=False` where supported.

Pipeline loads and MMA only while their operands and accumulators remain valid. Separate a load's completion from the MMA's last read of that buffer. Consider independent operand pipelines when their transfer costs differ, and compare the extra barriers and storage against the possible overlap.

For persistent execution, decide tile assignment and ordering from parallelism, L2 reuse, and load balance. Persistence alone can lose locality. Consider overlap between a tile's epilogue and the next tile's work, but account for the resulting overlap in buffer lifetimes. Output subtiles or reuse of expired operand storage can reduce the peak footprint when synchronization permits it. Read [Persistent Kernels](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html) before adapting this schedule.

For warp specialization, choose partitions such as load, MMA, elementwise/reduction, and store from the workload's independent work. Budget warps and registers for each partition, including the default partition, total launch allocation, and architecture allocation granularity. Use `maxnreg` and worker register budgets only with an explicit occupancy/spill check. Define ready/reusable barriers between partitions and verify compatible iteration counts. Compare against a pipeline without specialization; extra synchronization and resident warps must earn their cost. See [Warp Specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html).

## Preserve synchronization and memory ordering

`cp.async` commit groups form an ordered queue for the issuing threads. A `wait_group(N)` limits outstanding groups on that queue; it is not a named-buffer wait or a wait for other async instruction families. If unrelated prefetches are interleaved, map every buffer to its group or use supported per-buffer mbarrier tracking. Match arrival counts to participants. Check whether the installed `mbarrier_arrive` increments the expected count; a preinitialized participant count requires the non-incrementing form.

Wait before consuming a load and protect buffer reuse until all readers finish, including shared loads in other threads. Even same-lane staging needs an ordering argument for the actual emitted accesses. Track each mbarrier's phase through reuse; do not run more than one phase ahead or share a completion barrier between TMA and tcgen05 without reinitializing it.

Generic shared-memory accesses and async operations use different memory proxies. Apply the required proxy fence when handing shared storage between them; an ordinary mbarrier arrival/wait does not substitute for that fence. Completion of a TMA load establishes the ordering needed to read its destination. Check handoffs across warp-specialized partitions as well as within one loop.

TMA store waits may distinguish completion of the shared-memory read from completion of the global write. Check the installed API before releasing the source buffer or publishing the destination to another consumer. Require global completion before any dependent operation reads the stored range, and retain the repository's descriptor-store restrictions.

Masked async copies can leave shared-memory elements uninitialized. Prevent invalid elements from entering reductions: use defined zero-fill where supported, or valid-element loads with explicit removal of every invalid contribution. Multiplying by zero does not neutralize a NaN. Do not rely on separately compiled reductions cancelling bitwise; preserve and test mathematically exact-zero cases such as a single-source softmax gradient.
