# Gluon execution design

Choose ownership, storage, and scheduling together for the complete producer/consumer path. Read the relevant official implementation from the [skill's source table](../SKILL.md#diagnose-and-read) and check installed APIs before adapting it. The sections below follow the design dependencies; revisit them as measurements expose a different constraint.

## Ownership and communication

For important tensors, record shape, dtype, physical strides, register/lane/warp ownership, storage, consumers, and lifetime. Include indices, masks, accumulators, and loop-carried state when they materially affect register pressure. Mark each communication edge: global transaction, register reuse, warp shuffle, cross-warp shared exchange, or MMA operand transfer.

Derive `BlockedLayout` from those consumers:

| Parameter          | Decision                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| `size_per_thread`  | Which neighboring values should a thread own for vector access, local reuse, or accumulation?     |
| `threads_per_warp` | Which dimensions need lanes for coalescing, and which operations will require shuffles?           |
| `warps_per_cta`    | Should warps cover independent rows/heads/matrices or cooperate across a reduction?               |
| `order`            | In which dimension should register and thread tiles advance given physical strides and consumers? |

Match the lane product to the target's warp size and the warp product to the participating partition's warp count. Derive index and broadcast tensors with compatible `SliceLayout` parents. Match pointer and value ownership at loads/stores. A logical reshape or transpose is not proof of free physical redistribution; use `DistributedLinearLayout` when blocked/slice layouts cannot express the needed mapping clearly, and work out representative register/lane/warp mappings.

For reductions, scans, and gathers, compare thread-local, warp-local, and cross-warp work while retaining acceptable global accesses. Optimize repeated consumers and the final store together. For recurrent kernels, identify who retains state across iterations; layout changes do not remove mathematical recurrence dependencies.

Classify important `gl.convert_layout` operations by actual movement: equivalent ownership or register reordering, intra-warp exchange, or cross-warp exchange. Use `assert_trivial=True` for intended free conversions. Otherwise inspect the lowering and count shared scratch space and barriers. Consider assigning the producer directly to its consumer's layout, retaining a useful layout across several operations, or loading staged shared memory into the consumer layout. Converting just for one reduction can cost more than it saves.

Two FLA examples make the trade-offs concrete:

- [Sinkhorn](../../../../fla/ops/utils/sinkhorn/gluon.py): for `[BT, BN, BN]`, `warps_per_cta=[4, 1, 1]` places warps along independent matrices so both matrix reduction axes stay within a warp. Lanes split across matrix axes and, for small matrices, across multiple matrices. Check whether repeated row/column reductions avoid shared exchanges without sacrificing global coalescing.
- [Causal convolution](../../../../fla/modules/causal_conv1d/backends/gluon.py): a register window for `BC` adjacent outputs with filter width `W` holds `BC + W - 1` inputs per channel. Compare saved loads with the larger live register set as `BC` grows. Measure transactions and spills; source-level reuse alone does not prove a speedup.

Treat these as reasoning examples, not portable block-size prescriptions. Estimate values per thread including replication when the layout tile exceeds the tensor, dtype packing, fp32 accumulators, addresses, and temporaries. Confirm compiled register use, actual load/store widths, and shuffle/shared instructions. Wider access or less communication can still lose through register pressure or poor coalescing.

## Storage and movement

Compare ordinary register loads/stores, `cp.async` staging, and TMA over the entire path: addressing, masks, intermediate storage, synchronization, and final consumer access. `cp.async` can overlap global-to-shared transfers with independent work but adds a shared load for register consumers. TMA can reduce per-element addressing pressure while constraining descriptors and shared layouts. Staging should earn its cost through reuse, communication, or overlap.

On NVIDIA, `cp.async` requires Ampere or newer, TMA requires Hopper or newer, WGMMA targets Hopper, and tcgen05/TMEM targets Blackwell. Verify the exact instruction, dtype, layout, and architecture combination; one minimum compute capability does not gate every feature. Operations in Gluon's `nvidia` namespace require NVIDIA hardware.

Derive descriptor eligibility from base alignment, strides, dtype, and tile shape. Follow the repository's TMA opt-in and fallback rules in [CONTRIBUTING.md](../../../../CONTRIBUTING.md#triton-kernels), and include descriptor setup in entry-point measurements. Preserve pointer accesses for unsupported inputs or investigate a supported gather mechanism when justified.

Choose shared-memory and distributed layouts together. Derive representative warp byte addresses for every reader and writer, including dtype width and instruction vectorization. Consider `SwizzledSharedLayout` parameters (`vec`, `per_phase`, `max_phase`, `order`) or `PaddedSharedLayout` where both producer and consumer support them. Padding consumes capacity; layouts legal for ordinary shared loads may not be legal for TMA/MMA. Use `NVMMASharedLayout` where the selected NVIDIA operation requires it.

Count peak simultaneously live buffers, conversion scratch space, barriers, and epilogue storage. Check compiled allocation against the estimate and target limits; lifetimes and allocation granularity affect occupancy and feasible pipeline depth. [AttnRes](../../../../fla/ops/attnres/backends/gluon.py) illustrates resident versus streaming backward paths. Measure address instructions, shared accesses/bank conflicts, and occupancy to determine whether the chosen path pays off.

## MMA operands and accumulators

Choose legal instruction/tile shapes and operand locations under the existing precision contract. Include surrounding reductions, conversions, and output stores in the comparison. A derived operand already in registers can favor a different WGMMA path from an operand arriving through TMA; the fastest isolated MMA need not produce the fastest complete tile.

Hopper WGMMA requires B in shared memory and accumulates in registers. Thread the returned accumulator through the wait's dependency argument and consume the wait's returned value. Blackwell tcgen05 accumulates in TMEM and signals completion through mbarriers. Check TMEM capacity, allocation granularity, register-load layout, and participating warp constraints separately from SMEM and registers. Initialize accumulators explicitly, including `use_acc=False` where supported.

Distinguish operand-load completion from the MMA's last read before reusing a buffer. Independent operand pipelines may help when transfer costs differ, but add synchronization and storage. Inspect actual MMA instructions/utilization, accumulator movement, and epilogue work to identify the limiting stage.

## Pipelines and synchronization

Draw the prologue, steady state, and drain. Identify which future load overlaps which current computation and which MMA/store remains outstanding. An async instruction immediately followed by its wait provides no overlap with intervening work. Do not assume a copied Triton `num_stages` setting constructs the intended Gluon pipeline; check the explicit schedule and installed compiler behavior.

Use operation latency and the interval between consumers to choose an initial prefetch distance, then bound it by resources and tune it. Account for producer issue cost and the critical dependency chain. Additional stages can hide stalls while reducing occupancy; a latency ratio alone does not determine the optimal depth.

For each ring buffer, record producer, consumer, ready event, last read, reuse event, index, and barrier phase. Ready means data can be consumed; reusable means all consumers have finished. Include output stores and accumulator lifetimes. Apply these ordering rules to the timeline:

- `cp.async` commit groups form an ordered queue for the issuing threads. `wait_group(N)` limits outstanding groups on that queue; it does not name a buffer or wait for other async instruction families. Map interleaved prefetches to their groups or use supported per-buffer mbarrier tracking. Match arrival counts to participants, and check whether the installed `mbarrier_arrive` increments the expected count; a preinitialized participant count requires the non-incrementing form.
- Protect reuse until all readers finish, including shared loads in other threads. Even same-lane staging needs an ordering argument for emitted accesses. Track mbarrier phases through reuse; do not run more than one phase ahead or share a completion barrier between TMA and tcgen05 without reinitializing it.
- Generic shared-memory accesses and async operations use different memory proxies. Apply the required proxy fence at their handoff; ordinary mbarrier arrival/wait does not replace it. Completion of a TMA load establishes ordering for reads of its destination. Check these handoffs across warp partitions as well as within one loop.
- TMA store waits may distinguish completion of the shared-memory read from the global write. Check the installed API before releasing the source or publishing the destination. Require global completion before a dependent read and retain the repository's descriptor-store restrictions.

Exercise zero work when supported, one iteration, fewer iterations than buffers, partial tiles, and multiple ring wraps. Producers and consumers must agree on iterations and signals, including predicated tails. Masked async copies can leave shared elements uninitialized: use defined zero-fill or valid-element loads with explicit removal of invalid contributions. Multiplying by zero does not neutralize a NaN. Keep output masks, and preserve exact-zero cases such as a single-source softmax gradient rather than relying on separate reductions cancelling bitwise.

Verify issue/wait placement, dependency stalls, and throughput against the intended overlap. Use supported race/synchronization tooling where available and relevant; a numerical sample alone does not prove ordering. Report unsupported tools or unrun checks accurately.

## Scheduling

For persistent execution, choose tile assignment and ordering from parallelism, L2 reuse, and load balance. Persistence alone can lose locality. Overlapping a tile's epilogue with the next tile extends buffer lifetimes; output subtiles or reuse of expired operand storage can reduce the footprint when synchronization permits it. Include short and tail-heavy workloads where setup/drain costs matter.

For warp specialization, derive partitions such as load, MMA, elementwise/reduction, and store from independent work. Budget warps and registers for each partition, including the default partition, total launch allocation, and architecture allocation granularity. Check occupancy and spills when setting `maxnreg` or worker register budgets. Define ready/reusable barriers and compatible iteration counts using the pipeline rules above.

Compare against a pipeline without specialization. Extra resident warps, synchronization, and shared storage must earn their cost. Inspect L2 reuse, load balance, barrier stalls, and resource use alongside latency; high occupancy or the presence of specialized partitions is not itself a performance result.
