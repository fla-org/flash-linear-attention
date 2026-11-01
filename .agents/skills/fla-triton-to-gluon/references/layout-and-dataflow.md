# Layout and dataflow design

Use this reference to choose tensor ownership and memory layouts. Read the official [Tensor Layouts tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) and the relevant layout classes in the [language API](https://triton-lang.org/main/gluon/api/language.html), then check the installed definitions. The objective is to reduce the cost of the complete producer/consumer path.

## Trace ownership and lifetime

For important tensors, record logical shape, dtype, physical strides, register/lane/warp ownership, storage location, consumers, and lifetime. Include index tensors, masks, accumulators, and loop-carried state when they materially affect register pressure. Distinguish data that must survive an iteration from temporary values that can be released or recomputed.

Mark communication at each edge: global transaction, register reuse, warp shuffle, cross-warp shared-memory exchange, or transfer to an MMA operand. A layout that makes a load efficient can make the following reduction or store expensive. Optimize repeated consumers and the final write together.

For recurrent kernels, identify the owner of state across iterations before distributing work along the sequence dimension. Only schedule work concurrently when the existing mathematical dependencies permit it; layout changes do not remove recurrence dependencies.

## Derive distributed layouts

Use `BlockedLayout` parameters as design variables with a reason for each choice:

| Parameter          | Decision to make                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `size_per_thread`  | Which neighboring values should a thread own for vector access, local reuse, or local accumulation?       |
| `threads_per_warp` | Which dimensions need lanes for coalescing, and which operations would then require warp shuffles?        |
| `warps_per_cta`    | Should warps cover independent rows/heads/matrices, or cooperate at the cost of cross-warp communication? |
| `order`            | In which dimension should register and thread tiles advance, given physical strides and consumers?        |

Match the lane product to the target's warp size and the warp product to the participating partition's warp count. Derive index and broadcast tensors with compatible `SliceLayout` parents. Match pointer and value ownership at loads/stores rather than assuming a logical transpose changes physical ownership for free.

Estimate values held per thread, including replication when the layout tile exceeds the logical tensor. Account separately for dtype packing, fp32 accumulators, address arithmetic, and temporaries; a logical element count is not a compiled register count. Wider per-thread access may improve vectorization but lengthen live ranges or reduce parallelism. Confirm the actual instruction width.

For reductions, scans, and gathers, compare thread-local, warp-local, and cross-warp work while retaining acceptable global accesses. Changing the original ownership can eliminate repeated communication. Converting solely for one reduction can cost more than it saves, so include both conversions and any synchronization in the comparison.

Use `DistributedLinearLayout` when blocked/slice layouts cannot express the needed mapping or register-local reshaping clearly. Work out representative register/lane/warp mappings and verify them against the installed implementation. Greater expressiveness is useful only when the resulting communication or access pattern improves the target workload.

## Place and cost conversions

Classify each important `gl.convert_layout` by its actual movement: equivalent ownership or register reordering, intra-warp exchange, or cross-warp exchange. Use `assert_trivial=True` for an intended free conversion. For other conversions, inspect the lowering and include shared scratch space and barriers in the resource budget.

Try assigning a producer directly to its consumer's layout, retaining a useful layout across multiple operations, or loading staged shared memory into the consumer layout. Compare the cost of staging against the conversion it replaces. Move conversions outside repeated loops only when dependencies and lifetimes permit it.

MMA operand and accumulator layouts impose additional constraints. Choose them with the surrounding reductions, elementwise operations, and epilogue in mind. A fast MMA tile can still lose overall through accumulator conversion or costly output stores; use the [MMA and scheduling guidance](pipelines-and-scheduling.md#choose-mma-and-scheduling) for that path.

## Design shared-memory accesses

Shared-memory behavior depends on both the physical shared layout and the accessing threads' distributed layout. Derive representative byte addresses for a warp's reads and writes, accounting for dtype width and instruction vectorization, then check bank conflicts on the target. An identity swizzle is appropriate for some accesses and poor for others.

Consider `SwizzledSharedLayout` parameters (`vec`, `per_phase`, `max_phase`, `order`) or `PaddedSharedLayout` where supported by the producer and consumer. Check each reader and writer, including transpose-like accesses. Padding consumes capacity; a swizzle or padding scheme legal for ordinary shared loads may not be legal for TMA or MMA. Use the required `NVMMASharedLayout` for NVIDIA operations that demand it.

Count the peak simultaneously live allocations, including pipeline stages, layout-conversion scratch space, and epilogue storage. Check compiled usage against the estimate: allocation lifetimes and reuse can change occupancy and the feasible pipeline depth. See [Async Copy](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html) for the interaction between register layouts, shared layouts, and buffering.

## Apply the method to FLA

- [Sinkhorn](../../../../fla/ops/utils/sinkhorn/gluon.py): for its `[BT, BN, BN]` tile, `warps_per_cta=[4, 1, 1]` places warps along independent matrices, keeping both matrix reduction axes within a warp. Lanes split across the matrix axes and, for small matrices, across multiple matrices. The hypothesis is fewer shared exchanges across repeated row/column reductions; verify that lowering while checking global coalescing and per-thread storage.
- [Causal convolution](../../../../fla/modules/causal_conv1d/backends/gluon.py): a register window for `BC` adjacent outputs with filter width `W` holds `BC + W - 1` input values per channel, instead of loading `W` values independently for each output. Evaluate saved loads against the larger live register set as `BC` grows; inspect spills and measured transactions before accepting the change.
- [AttnRes](../../../../fla/ops/attnres/backends/gluon.py): inspect how token/feature ownership interacts with paired reductions, staged values, and resident versus streaming backward paths.

These are examples of design reasoning, not portable block-size or performance prescriptions. Recheck their contracts, current implementation, and target workload before adapting them.
