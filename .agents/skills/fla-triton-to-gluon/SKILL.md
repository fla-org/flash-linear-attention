---
name: fla-triton-to-gluon
description: Port an FLA Triton kernel to Gluon when explicit layouts, asynchronous transfers, or scheduling can address a measured bottleneck.
---

# Triton to Gluon

Use Gluon when profiling points to register pressure, layout conversion, or load/compute overlap that needs explicit control. A kernel already saturating its useful bandwidth or compute limit may not benefit. Keep the portable default path when adding hardware-specific implementations.

Follow [fla-optimization-loop](../fla-optimization-loop/SKILL.md) for the baseline and correctness gate, and [fla-nvidia-performance](../fla-nvidia-performance/SKILL.md) for NVIDIA measurements.

## Check the target environment

Gluon is experimental. Check the installed Triton API before copying [upstream examples](https://triton-lang.org/main/getting-started/tutorials/gluon/); their names and signatures can differ from the installed release.

Gate architecture-specific code appropriately: Ampere supports `cp.async`, Hopper adds TMA/WGMMA, and Blackwell adds TMEM/tcgen05. Code under Gluon's `nvidia` namespace requires NVIDIA hardware. Keep compile-time choices in explicit constexpr arguments so the kernel, launcher, and autotune pruning use the same values.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

## Port incrementally

1. Record the baseline commit and retain the validated tests, reference, and tolerances. Keep the Triton fallback available.
2. Translate loads, stores, and arithmetic with explicit layouts first. Match each tensor's contiguous dimension and preserve masking. Pass forward, state, and supported backward comparisons before adding asynchrony.
3. Add async transfers only where load latency or addressing pressure limits the kernel. Define buffer ownership, completion, and reuse for the prologue, steady state, and drain.
4. Add architecture-specific MMA or scheduling when profiling supports it. Recheck register/shared-memory use and retune after each change.
5. Run the full relevant correctness gate and same-hardware benchmarks, including supported variable-length and boundary cases. Measure production entry points as well as individual kernels.

A literal translation establishes parity; measure it before assuming a performance gain. Preserve the validated numerical algorithm and precision while changing execution layout or scheduling.

## Layout and resource constraints

- Pointer tensors and values need compatible layouts. A cross-warp `gl.convert_layout` may use shared memory; use `assert_trivial=True` when a conversion is intended to be free.
- Broadcasts and oversized layout tiles can duplicate values and increase register use. Choose layouts from actual tensor strides rather than copying a tutorial's block sizes.
- Account for all live shared-memory buffers and the target device's allocation limit. Prune infeasible autotune configurations; provide a streaming path when a resident design cannot fit supported shapes.
- Large static unrolls multiplied by many autotune configurations can dominate compilation. Prune from the actual shape and live-memory budget, and reuse the worker and Triton cache during iteration.
- Persistent schedules can trade launch overhead for worse L2 locality. Warp specialization also changes the total register budget; measure occupancy and stalls after changing either.

## Async correctness

### Transfers and buffer reuse

`cp.async` groups are ordered as a queue. `wait_group(N)` bounds all outstanding groups, so it cannot identify a particular buffer after unrelated prefetches are interleaved. For per-buffer scheduling, use separate mbarriers and match their arrival counts to the participating threads. In the installed API, check whether `mbarrier_arrive` increments the count; a preinitialized thread count requires the non-incrementing form.

Wait for a load before consuming its buffer, and protect reuse until all readers finish. Even same-lane shared-memory staging needs protection against overwriting data still being read. Track each mbarrier's phase as its buffer is reused; do not run more than one phase ahead or share a completion barrier between TMA and tcgen05 without reinitializing it.

TMA descriptors must satisfy the target's alignment and stride requirements. TMA stores may distinguish completion of the shared-memory read from completion of the global write. Check the installed wait API and require global completion before another operation reads the stored range.

### MMA and memory ordering

Hopper WGMMA requires its B operand in shared memory and uses register accumulators; consume the values returned by its wait operation so compiler dependencies remain explicit. Blackwell tcgen05 uses TMEM accumulators and mbarrier completion; respect the participating warpgroup and TMEM layout requirements. Initialize accumulators explicitly, including `use_acc=False` where supported.

Generic shared-memory accesses and async operations use different memory proxies. Apply the required proxy fence when handing a buffer to an async consumer; an mbarrier alone does not replace that fence. A completed TMA load establishes the ordering needed to read its destination. Check ordering across warp-specialized partitions as well as within a single producer/consumer loop.

### Tails and exact-zero cases

Masked async copies can leave shared-memory elements uninitialized. Prevent invalid rows from entering reductions: zero-fill where supported, or use valid-row loads and explicitly remove every invalid contribution. Multiplying by zero does not neutralize a NaN. Keep output stores masked.

Do not rely on separately compiled reductions cancelling bitwise. For a mathematically exact-zero special case, such as a single-source softmax gradient, preserve the exact-zero result explicitly and test it against the reference.

## Verification commands

Run from the repository root on the target GPU:

```bash
FLA_BENCH_BASE=$(git rev-parse origin/main)
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_kda --base "$FLA_BENCH_BASE"
```

Replace the operation with the one being ported. The command executes the current tests; it does not freeze them. Use `--gate-k` only for quick iteration, then run the full gate. Set backend environment flags before launching a fresh process so import-time choices cannot contaminate the comparison.
