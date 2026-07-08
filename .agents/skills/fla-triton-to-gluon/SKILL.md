---
name: fla-triton-to-gluon
description: >
  Workflow for porting an existing Triton kernel in `fla/ops/**` to Gluon (`triton.experimental.gluon`)
  to gain explicit control over tensor layouts, shared memory, async data movement (cp.async / TMA),
  MMA (WGMMA / tcgen05), and scheduling (persistent kernels, warp specialization).
  Covers when a port is worth it, an incremental porting sequence that keeps numerical parity at every
  step, a Triton-to-Gluon API mapping, compile-time / autotune / smem-budget management for heavily
  unrolled kernels, and a pitfall checklist (proxy fences, mbarrier semantics, layout costs,
  bitwise-cancellation traps, NaN-poisoned OOB handling).
  Use when a Triton kernel is register-bound, when `num_stages` pipelining underperforms,
  or when Hopper/Blackwell features (TMA, TMEM, tcgen05) are needed.
---

# FLA Triton → Gluon Porting Skill

Gluon shares Triton's compiler stack, JIT, and SPMD tile model; host-side launch code is unchanged.
The difference: layouts, shared memory, asynchrony, and synchronization are all explicit.
**Port incrementally**: first a literal translation that passes the op's frozen pytest, then upgrade
layer by layer driven by profiling, keeping numerical parity after every step.

Related skills:

- **`fla-optimization-loop`** — the iteration discipline around this port (frozen test contract, recording, when to stop).
- **`fla-nvidia-performance`** — profiling workflow, hardware baselines, MR-ready perf evidence.
- **`fla-correctness-coverage`** — test coverage matrix for the op being ported.

## When a port is worth it

Worth it:

- Register spills cap the block size (Triton gives you no lever; Gluon's TMA path moves addressing out of registers).
- `num_stages` pipelining fails to overlap load and compute the way you want.
- The kernel needs TMA features Triton does not expose well (im2col, gather/scatter, multicast).
- Blackwell-specific paths: TMEM accumulators, `tcgen05_mma`, 2-CTA MMA, CLC dynamic scheduling.
- Load and compute are imbalanced enough to justify warp specialization.

Not worth it:

- The kernel already saturates bandwidth or tensor-core throughput.
- The bottleneck is algorithmic, not scheduling.
- The op must stay portable across vendors — Gluon's `nvidia` modules are NVIDIA-only (AMD is a separate submodule).

## Environment and versions

- Gluon lives under `triton.experimental` and **its API moves between Triton versions**.
  The official tutorials (https://triton-lang.org/main/getting-started/tutorials/gluon/) track the `main` branch;
  verify names against the installed Triton with `dir()` before copying tutorial code.
  Known examples: in Triton 3.5.1 there is no `gluon.aggregate`, and the TMA/cp.async load is
  `async_copy_global_to_shared` (named `async_load` on `main`).
- Since Triton 3.6, a kernel may not read plain module-level Python globals (`NameError: ... instantiated
  as constexpr`). Pass such values as constexpr kernel arguments instead — that also keeps host and
  kernel in sync and makes them visible to autotune key/prune functions.
- Hardware gating: `cp.async` needs Ampere+; TMA, WGMMA, `gl.warp_specialize`, CGA clusters need Hopper+;
  TMEM, `tcgen05_*`, CLC, and TMA gather/scatter need Blackwell.
  Follow the hardware baseline rules in `fla-nvidia-performance`.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import ampere, hopper, blackwell
from triton.tools.tensor_descriptor import TensorDescriptor  # TMA, host side
```

## Triton → Gluon mapping

| Triton                    | Gluon                                                                       | Notes                                                                |
| ------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `@triton.jit`             | `@gluon.jit`                                                                | `triton.autotune`, `triton.cdiv`, `do_bench` are reused as-is        |
| `tl.load` / `tl.store`    | `gl.load` / `gl.store`                                                      | every tensor (including pointer tensors) needs an explicit layout    |
| `tl.arange`               | `gl.arange(..., layout=gl.SliceLayout(dim, parent))`                        | 2D offsets = SliceLayout + `expand_dims` + broadcast (free)          |
| `tl.dot`                  | Hopper: `hopper.warpgroup_mma`; Blackwell: `blackwell.tcgen05_mma`          | async instructions; explicit wait/commit required                    |
| `num_stages=N`            | manual multi-buffering: smem gets a leading `[num_buffers, ...]` dim        | prologue / steady-state / epilogue skeleton below                    |
| (compiler-managed smem)   | `gl.allocate_shared_memory(dtype, shape, layout)`                           | `SwizzledSharedLayout` / `NVMMASharedLayout` to avoid bank conflicts |
| (compiler-managed layout) | `gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)` | see layout guidance below                                            |
| `tl.trans(b)`             | `b_smem.permute((1, 0))`                                                    | forwarded to the MMA hardware, zero-copy                             |
| `tl.static_range`         | `gl.static_range`                                                           | used for prologue peeling                                            |

## Porting sequence

### 1. Freeze the baseline

Keep the Triton kernel and the op's `tests/ops/test_<op>.py` untouched (they are the frozen contract per
`fla-optimization-loop`). The Gluon kernel is added alongside and must pass the same parity tests
(forward **and** backward, via `fla.utils.assert_close`) before any optimization.

### 2. Literal translation (`gl.load`/`gl.store`, correctness first)

Add layouts, no async anything. **A literal port is a parity scaffold, not a deliverable**: it drops
Triton's automatic vectorization and `num_stages` pipelining without adding any manual control, so it
usually ties or loses to the Triton kernel. Measured on a bandwidth-bound op (attnres, GB200): the
literal port was ≈ Triton; the wins (fwd 1.3–1.45×, reaching 66–72% of HBM peak) all came from the
restructuring steps below.

While translating, also restructure what Triton forced you to express dynamically: small runtime
dimensions (e.g. a source/tensor count) become constexprs, so runtime pointer-select chains
(`tl.where(o == i, ptrs_i, p)`) turn into static indexing over an unrolled `gl.static_range` — and
gathers over many tensors become contiguous per-tensor block loads that async copy can handle.

Layout starting points:

- 1D: `size_per_thread=[1]` — each warp issues exactly one 128-byte coalesced access; measured faster
  than larger per-thread tiles in the tutorials.
- 2D row-major: `size_per_thread=[1, N]`, `order=[1, 0]`. **The layout's contiguous dim must match the
  tensor's contiguous dim** — a mismatch costs an order of magnitude of bandwidth (6.3 → 0.8 TB/s in the tutorial).
- Input and output with opposite contiguity: derive a layout from each tensor's strides, pay one
  `gl.convert_layout` in the middle, use square-ish blocks (e.g. 128×128).
- Broadcast waste: a tensor smaller than the layout's block shape still burns the full register budget
  (redundant copies per thread/warp).

### 3. Async data movement (the first big jump)

**cp.async** (Ampere+, small diff): `ampere.async_copy.async_copy_global_to_shared` →
`commit_group()` → `wait_group(N)`.

**TMA** (Hopper+, frees registers so blocks can grow): host side
`TensorDescriptor.from_tensor(t, block_shape)`; smem must use `NVMMASharedLayout`; strides 16-byte aligned;
loads tracked by an mbarrier (`expect(bar, nbytes)` → `wait(bar, phase)`), stores by
`tma.store_wait(pendings=N)`. Out-of-bounds masking is automatic.

Pipeline skeleton (same shape for both mechanisms):

```
smem = allocate([num_buffers, BM, BN]); one mbarrier per buffer
prologue:      issue num_buffers - 1 loads (gl.static_range)
steady state:  issue load i + num_buffers - 1; wait load i; compute; release buffer i
               buffer index = i % num_buffers; mbarrier phase = i // num_buffers & 1
epilogue:      drain with decreasing wait counts
```

Pick the pipeline depth from the load/compute latency ratio; going deeper past bandwidth saturation buys nothing.

cp.async specifics learned the hard way:

- **Same-lane staging**: when the cp.async pointer tensor and the smem readback use the same blocked
  layout, every thread reads back exactly the bytes it copied — smem is pure staging for asynchrony,
  with no cross-thread exchange. Buffer reuse still gets a `gl.thread_barrier()` before the refill
  (WAR safety); it costs ~a barrier, not a pipeline stall.
- **Commit groups are one global FIFO counter**: `wait_group(N)` counts *every* group issued later, so
  it cannot express "wait for slot l only" once you interleave prefetches for the next loop iteration
  with consumption of the current one — the wait would also cover the new issues and serialize you
  again. For per-slot pipelining across iterations, switch to `ampere.mbarrier`: one barrier per slot,
  `mbarrier.init(bar, count=num_warps * 32)`, and after each thread's issues
  `async_copy.mbarrier_arrive(bar, increment_count=False)` (the *noinc* form consumes the
  pre-initialized count; the default self-increments and never completes with a thread-count init).
  Consumers `mbarrier.wait(bar, phase=t & 1)` — one fill per iteration flips parity.

### 4. MMA (if the kernel has a dot)

- **WGMMA (Hopper)**: B must be in smem; accumulator in registers with
  `NVMMADistributedLayout(version=[3, 0])`; M ≥ 64 (one warpgroup minimum); results must flow through the
  return value of `warpgroup_mma_wait(deps=...)` or ordering is not guaranteed.
- **tcgen05 (Blackwell)**: accumulator must live in TMEM (`allocate_tensor_memory` + `TensorMemoryLayout`);
  TMEM loads/stores need a full warpgroup (each warp sees only 32 of 128 rows); completion via
  `tcgen05_commit` + mbarrier; `tcgen05_copy` moves smem→TMEM without a register round-trip, and
  same-pipe tcgen05 instructions are implicitly ordered (a copy followed by an MMA needs no wait).
- Both: `use_acc=False` is the cheapest way to zero-initialize the accumulator.

### 5. Scheduling layer (profile first, never by default)

Persistent kernels (grid = `min(num_sms, num_tiles)` + a tile scheduler — add grouped/swizzled tile order
or L2 hit rate drops) → `gl.warp_specialize` (load/MMA/epilogue partitions; a TMA-issue-only partition
needs 1 warp and 24 registers; **set `maxnreg` explicitly**) → multi-CTA / CLC (Blackwell).
**Re-autotune after every layer**: in the tutorials, the pre-pipelining best config lost >100 TFLOPS
after pipelining was added.

## Compile time, autotune, and the smem budget

Three interacting constraints that only show up at scale:

- **Unroll × configs = compile explosion.** `gl.static_range(K)` fully unrolls; a body unrolled ~30×
  across two passes, multiplied by ~9 autotune configs, can take tens of minutes per shape. Use
  `fla_cache_autotune(..., prune_configs_by={'early_config_prune': fn})`; the prune fn receives all
  kernel args (`{**named_args, **kwargs}`), so it can cap the sweep to 1–2 configs when the unroll
  factor is large.
- **Shared memory is a hard cap** (228KB/SM on Hopper/Blackwell; budget ~192KB to leave headroom).
  When a "keep everything resident" design can exceed it for some shapes, put both designs in one
  kernel behind a `gluon.constexpr_function` switch (e.g. `RESIDENT = (L+1)*BT*BD*ES <= budget`):
  resident path for the common case, streaming double-buffer fallback for the rest. Prune configs
  whose *minimal* footprint still exceeds the budget — Triton's autotuner does not reliably skip
  smem-overflow configs on its own.
- **Big smem buys traffic but kills occupancy.** A resident design at 192KB runs 1 CTA/SM (12.5%
  warp occupancy at 8 warps); at that point latency hiding must come from within the CTA — that is
  exactly what the per-slot mbarrier pipelining above provides. Read the trade-off from NCU:
  `dram__throughput...pct_of_peak` low + `stalled_long_scoreboard` high + `warps_active` low means
  serialized loads, not insufficient bandwidth.

## Pitfall checklist (check here first when things break)

1. **Proxy fences**: registers use the generic proxy; TMA/WGMMA/tcgen05 use the async proxy; the two are
   unordered. Any plain smem access adjacent to a TMA/MMA op on the same buffer needs
   `fence_async_shared()` — this holds across `warp_specialize` partitions and is *not* waived by
   mbarrier arrive/wait ordering. Sole exception: after `mbarrier.wait` on a **TMA read** barrier,
   reading that smem needs no fence.
2. **mbarrier phase**: phase = `i // num_buffers & 1`. A barrier only tracks the current and previous
   phase; running more than one phase ahead desynchronizes permanently. Never reuse one mbarrier for
   both TMA and tcgen05 completion (undefined behavior) — allocate separately or reinitialize.
3. **`tma.store_wait` waits only for the smem read by default**, not the global write. If the stored
   range is read afterwards (e.g. cross-CTA signaling), pass `read_only=False`.
4. Wrong values without a crash: usually a layout broadcast / conversion misunderstanding.
   Debug with `gl.static_print` on layouts and `convert_layout(..., assert_trivial=True)` to prove a
   conversion is actually free.
5. Illegal-instruction / driver errors: check TMA alignment first (descriptor strides 16-byte; gather
   `y_offset` 16-byte).
6. Slower after "optimizing": register budget blown (warp-specialized total ≈
   `maxnreg × (num_warps + 4) × 32`), a cross-warp `convert_layout` silently routing through smem,
   or a persistent schedule tanking L2 hit rate (`lts__t_sector_hit_rate` in NCU — see
   `fla-nvidia-performance` for the profiling workflow).
7. **Never rely on two separately-compiled reductions cancelling bitwise.** If a gradient is
   mathematically zero only because `sum(a*b)` at two program points must agree to the last bit
   (e.g. softmax bwd over a single source: `ds = p*(dp - delta)` with `dp == delta`), Gluon may compile
   the two reductions differently and leave O(eps) residue that explodes against an exactly-zero
   reference. Branch on the degenerate constexpr case and emit exact zeros.
8. **OOB rows under NaN-poisoned tests**: masked cp.async leaves smem uninitialized, and NaN garbage
   in dead rows leaks through cross-row reductions (`0 * NaN = NaN`). Clamp indices to a valid row
   instead of masking the loads, then zero the one tensor (e.g. the incoming gradient) whose zeroing
   provably kills every masked contribution downstream; keep masks only on stores.

## Verification and benchmarking

Run on a GPU worker per `fla-nvidia-performance` hardware baselines (sm_90+; prefer sm_100/sm_103):

```bash
python -m pytest tests/ops/test_<op>.py -q                 # frozen parity gate, fwd + bwd
python benchmarks/ops/run.py --op <op> --base main         # before/after vs the Triton baseline
```

Record every iteration per the `fla-optimization-loop` protocol; dense workloads for quick iteration,
varlen checked before the MR.

Iteration-speed hygiene (Gluon compiles are expensive):

- Keep one warm worker per optimization loop and a persistent `TRITON_CACHE_DIR` — a fresh
  machine per run recompiles every kernel × config from scratch and dominates wall-clock.
- Order each round for fast signal: cheapest bench first, full frozen pytest after; split
  slow-compiling parameterizations (huge unroll factors) into their own pytest invocation.
- If the backend is selected via a cached dispatch env var (`FLA_<OP>_<BACKEND>`), benchmark each
  backend in its own process with the env var set at launch.

## References

- Tutorial series (read in order; last six are advanced topics):
  https://triton-lang.org/main/getting-started/tutorials/gluon/
  (intro → layouts → async-copy → tma → wgmma → tcgen05 → persistence → warp-specialization →
  tma-gather-scatter, tcgen05-copy, tcgen05-mma-scaled, cluster-launch-control, conv-im2col, multicta)
- Tutorial sources live in the Triton repo under `python/tutorials/gluon/`; complete kernels under
  `python/examples/gluon/` (e.g. `02-convolution.py`, a pipelined warp-specialized convolution).
- Ground truth for the installed version:
  `python -c "from triton.experimental.gluon.language.nvidia import hopper; print(dir(hopper.tma))"`
