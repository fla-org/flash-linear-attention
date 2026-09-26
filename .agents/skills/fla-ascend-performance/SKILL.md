---
name: fla-ascend-performance
description: >
  Guidelines for Ascend NPU kernel / Triton-Ascend backend performance work in the FLA repo.
  Covers profiling with torch_npu, PipeUtilization/MemoryUB CSV analysis, Cube/Vector/MTE/UB
  bottleneck diagnosis, and kernel optimization (UB tiling, grid splits, fusion/split, varlen,
  G_T_CONTIG gate loading, constexpr DMA-path split / TAIL_MODE, extract_slice, MTE OOB,
  int32 address overflow, tl.cast vs constexpr .to, make_block_ptr int32 offsets,
  correctness gates, tl.dot left-operand clobber). NPU kernels must not use num_warps/num_stages.
  Per-kernel catalog: references/cases.md (incl. causal_conv1d core-grid).
  Use when working on NPU profiling, kernel_details/op_statistic, aic_metrics,
  fla triton_ascend backends (ops or modules), g transpose stride-1, UB overflow,
  dual-path DCE, grid limits, int64 pointer math, tl.dot reuse, or Ascend performance.
---

# FLA Ascend NPU: Profiling → Bottlenecks → Optimization

Use this skill for Ascend operator performance work on all files under any `triton_ascend` directory (`**/triton_ascend/**`).

Multi-round iteration discipline (frozen tests, task contract, when to stop): **`fla-optimization-loop`**. MR packaging: **`fla-mr-readiness`**.

Collection **must** use this skill's **generic scripts** — do not copy `torch_npu.profiler` boilerplate per op.

Environment: **Use the Python/NPU environment already active in the current terminal** (including any activated conda/venv). Run collection, analysis, and benchmarks in the same shell; do not spawn a new shell or switch environments mid-workflow. If the terminal has no NPU stack loaded yet, activate the project's Ascend environment first, then continue in that same session. Metrics, failure modes, code index: [reference.md](references/reference.md). Past kernel notes: [cases.md](references/cases.md).

Make the target backend semantically correct before optimizing; never hide missing capability or kernel bugs behind a Torch fallback. What generalizes: UB modeling, grid splits, layout/precision, and verification. Values like `BC=16`, K slabs of 64, or specific `mem_mult` are starting points only — do not copy them as rules.

**Hard constraint (NPU launch params):** Ascend Triton kernels **do not support** `num_warps` or `num_stages`. During optimization these kwargs must **never** appear in `@triton.jit` launches, `triton.autotune` configs — do not copy them from CUDA Triton. Tune via tiles, grid, layout, fusion/split, and UB budget only.

## Progress checklist

```
- [ ] 1. Freeze semantics, workload, and baseline latency
- [ ] 2. Collect with generic scripts (first pass: PipeUtilization)
- [ ] 3. Parse CSVs and classify the bottleneck
- [ ] 4. Triton-Ascend optimize for that bottleneck (MemoryUB if needed)
- [ ] 5. Correctness gate + synchronized benchmark
- [ ] 6. Re-profile to confirm metrics, then decide whether to continue
```

## 1. Freeze semantics and baseline

1. Locate the public entry, `@dispatch`, default impl, and closest Ascend impl; list layout, dtype, fixed/varlen, head mapping, fwd/bwd, and optional args.
2. Keep Torch reference implementations only in tests/benchmarks as the oracle.
3. Pick shape/dtype/fwd±bwd; freeze tests, tolerances, and shapes during optimization — do not change tests to manufacture speedups.
4. Baseline with synchronized timing (warmup + `torch.npu.synchronize()` + repeats); confirm the target NPU kernel runs, not a Torch fallback.
5. Do not change the public API to fit the kernel; register backends under `IS_NPU` with lazy imports; verifiers must state real support ranges.

## 2. Generic collection (required)

Scripts live under `.agents/skills/fla-ascend-performance/scripts/` (run from that directory or set `PYTHONPATH`).

| Script | Role |
|--------|------|
| `scripts/profile_npu.py` | Trace any `workload()` |
| `scripts/analyze_profile.py` | Parse `op_statistic` / `kernel_details` |

```bash
SKILL_DIR=.agents/skills/fla-ascend-performance
cd "$SKILL_DIR"

python scripts/profile_npu.py \
  --name my_op --out-dir npu_prof \
  --metrics PipeUtilization --analyze \
  --kernel-filter my_kernel_substr \
  --exec-file path/to/workload_only.py
```

`workload_only.py` only defines `workload()` — no profiler boilerplate:

```python
def workload():
    y = op(...)
    y.backward(grad)
```

Library usage (when not using `--exec-file`):

```python
from profile_npu import profile_callable

def workload():
    y = op(...)
    y.backward(grad)

trace_dir = profile_callable(
    workload,
    name="my_op",
    out_dir="npu_prof",
    aic_metrics="PipeUtilization",  # or MemoryUB / L2Cache / ...
)
```

Default schedule: `wait=0, warmup=1, active=1, repeat=1`. One `aic_metrics` per run; start with `PipeUtilization`, collect `MemoryUB` separately for UB bandwidth.

## 3. Diagnose bottlenecks

```bash
cd .agents/skills/fla-ascend-performance
python scripts/analyze_profile.py path/to/*_profiling_* --kernel-filter <substr>
```

1. **`op_statistic`**: who owns Total Time; is the target kernel the real hotspot?
2. **`kernel_details`** (by Duration): read pipe / UB columns.

| Signal | Bottleneck | Prefer |
|--------|------------|--------|
| High `aiv_vec_ratio`, Cube≈0 | Vector-bound | Larger row tile, less scalar, fuse load/store |
| High `aic_mac_ratio` / `cube_utilization` | Cube-bound | Better matmul tiles/alignment, less non-Cube prelude |
| High `mte2/mte3_ratio`, low compute | Memory-move-bound | More reuse, fewer writebacks; check strides — **gate `g` stride-HV gather** often 10×+ slower ([g-contiguous-loading.md](references/g-contiguous-loading.md)) |
| High `scalar_ratio` | Scalar-bound | Vectorize, kill branches, heuristics |
| High UB bw under MemoryUB, low vec/mac | UB bandwidth saturated | Larger tiles / more fusion |
| Low target Ratio, many tiny ops | Unfused / fallback | Fix dispatch and fusion first |
| Two kernels share `o` + high MTE | Intermediate writeback | Fuse producer/consumer if UB fits; else keep split |
| Frequent host grid chunking | Launch / grid-product overhead | Prefer 1D core-grid (`num_aicore` Cube / `num_vectorcore` Vector) + flat `task_id` |
| Low `aiv_vec_ratio` (~0.75) while MemoryUB is *not* saturated; larger tiles UB-overflow | Dual DMA paths live in UB | Runtime `block_ptr` vs masked load: host-split with `tl.constexpr` so each launch DCE's the other ([cases.md § causal_conv1d](references/cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split)) |

Colloquial “CUDA utilization” → read **Cube/MAC** (`aic_mac_ratio`). Host UB model complements the profiler — see [reference.md](references/reference.md).

Prioritize fixes by **Duration share** in `kernel_details` / `op_statistic` (largest hotspot first). Low pipe ratios on a dominant kernel usually mean room remains on that pipe.

## 4. Optimize (Triton-Ascend)

Change only levers that match the bottleneck; one hypothesis per round. Before tuning, classify the issue: **compile failure / UB overflow / grid limit / numeric error / real performance bottleneck** — do not treat all five the same way.

### UB and tiles

- UB is usually the primary constraint, not theoretical FLOPs. Enumerate peak live tiles (fp32 accum, transpose copies, masks, temp dots).
- `peak ≈ memory_multiplier * tiled_elements * dtype_size`; comment where the multiplier comes from.
- Use `fla.utils.ascend_ub_manager` (`compute_row_tile_block_size`, etc.); do not hard-code capacity; keep ~0.75–0.85 safety margin.
- Prefer power-of-two tiles; matrix ops prefer 16-alignment; model fwd/bwd separately (bwd usually smaller tiles).
- If a fused kernel cannot fit a reliable UB budget, split stages + scratch/recompute — do not keep an inevitably overflowing live set for “fusion”.
- Persistently unused safe budget → consider non-PoT tiles / calibrate `mem_mult`; near 100% and still slow → look at pipe/bandwidth.

### Layout, grid, numerics

- Innermost block-pointer dim should be contiguous; `tl.make_block_ptr` + `boundary_check`; `@input_guard` for layout — do not emulate arbitrary strides in-kernel.
- **Gate `g` along T (critical on Ascend)**: if `g` is `[B, T, HV]`, host `g.transpose(1, 2).contiguous()` and load via `G_T_CONTIG` + stride-1 `g_ptr` (see [g-contiguous-loading.md](references/g-contiguous-loading.md)). Stride-`HV` gathers in bwd hot loops can be **10×–35× slower** than contiguous loads; HV==1 needs no transpose. Match fwd pointer math; keep `T_seq` before varlen overwrites `T`.
- Distinct shapes (e.g. `HV==1`, layout flags) get separate paths — no expensive hot-loop branches.
- Grid product cap `ASCEND_MAX_GRID_DIM=65535`: host-split with `iter_axis_launch_chunks`, pass `*_OFFSET`; after varlen slicing, zero the matching offset — never slice and also add a global offset. UB and grid are independent constraints.
- **1D core-grid** (prefer when there are many independent tiles and multi-axis grids need host chunking): flatten work into `task_num` and schedule with `for task_id in tl.range(core_id, task_num, num_core)` (or `range(pid, total_tasks, num_programs)`). Decode `task_id` → tile indices inside the kernel. One launch, no `ASCEND_MAX_GRID_DIM` host loop, better load balance when `task_num` is irregular. Keep `do_not_specialize` on `T` / `task_num` / `num_core` / dynamic extents.
- **Match core count to the bound pipe**: Cube-bound → `grid=(num_aicore,)` via `get_device_properties()["num_aicore"]`. Vector-bound (conv, layernorm, rotary) → `get_multiprocessor_count` (`num_vectorcore` on NPU; A2 is 48 vector vs 24 Cube). Launching a Vector kernel on `num_aicore` leaves half the vector cores idle.
- In a core-grid task loop, **rebind local pointers each iteration** (`q_ptr = q + …`); do not accumulate with in-place `ptr +=` across tasks — Ascend Triton can mis-compile that pattern.
- **int64 before multiply on runtime indices**: program IDs and grid-derived values (`i_t`, `i_b`, `NT = cdiv(T, BT)`) are runtime int32 or narrower. `do_not_specialize` on `T` makes `NT` runtime, but `i_t * stride` also wraps when `T` is specialized (packed conv: `i_t * BT` then `offset * D`). `(NT - 1) * DH_CS` wraps past 2³¹ before a trailing `.to(tl.int64)`. Example: DH_CS=`HV*K*V`, K=V=128, HV=64, BT=64 → overflow at NT>2048 (T>131K). Packed `offset * D`: T>2³¹/D (D=4096 → T>524K). Cast the index first with `tl.cast` (not `.to` on specialized ints): `tl.cast(i_t, tl.int64) * BT`, `tl.cast(i_b, tl.int64) * T`, `tl.cast(B, tl.int64) * T`, `tl.cast(NT - 1, tl.int64) * DH_CS`. Kernel args `B`/`T` are constexpr — `B.to(tl.int64)` is `AttributeError("'constexpr' object has no attribute 'to'")`; `i_t`/`i_b` can fold to constexpr when NT=1. `tl.load(...).to(tl.int64)` on `cu_seqlens` is fine. Never `(i_b * T).to(tl.int64)` or `((NT - 1) * DH_CS).to(tl.int64)`.
- **`make_block_ptr` offsets stay int32**: Triton rejects int64 `offsets/block_shape`. Flattened pointer math (`bos * D`, `t0 * D`, `i_b * stride`) uses int64; pass `i_t * BT` (int32) as the block row offset. Do not feed `t0` into `make_block_ptr`. Case: [causal_conv1d](references/cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split).
- **Varlen `cu_seqlens` → int64 for pointer math**: host dtype is often `torch.long`, but tests also pass `int32`; load as `tl.int64` either way. Loading `.to(tl.int32)` then `(bos * HV + i_hv) * V` overflows well before `bos` hits 2³¹ (HV=32, V=4096 → safe `bos` ≈ 16K). Pattern: `bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64); T_cur = (eos - bos).to(tl.int32)`. Non-varlen: `bos = tl.cast(i_b, tl.int64) * T` (CUDA/repo often writes `(i_b * T).to(tl.int64)`, which still wraps if `i_b * T` exceeds 2³¹). Alternative when `T_cur` only needs int32: load `bos` as int32 but cast **the index** before the large stride — `tl.cast(bos, tl.int64) * HV + i_h` then `* K`. `(bos * HV + i_h).to(tl.int64) * K` only fixes `* K`/`* V` (HV is small); `bos * HV` itself can still wrap.
- Reductions / recurrence / grads use fp32 accum, cast on store; sensitive solves: `input_precision='ieee'` / `allow_tf32=False`; mask before exp on gated paths; keep a consistent `exp`/`exp2` base.
- **Ascend `tl.dot` clobbers the left operand**: on NPU, `tl.dot(lhs, rhs, …)` may overwrite `lhs` in UB (CUDA Triton does not). Any later read of that tile (second lhs, rhs, store) sees corrupted data unless you reload from GM or copy with `tile + 0.0` **before** the first lhs dot. Full per-kernel catalog: [cases.md § tl.dot lhs clobber](references/cases.md#tldot-lhs-clobber--repo-wide-case-catalog). Symptom: silent numeric drift vs Torch oracle with no compile error.
- **Audit checklist for new/changed kernels**: (1) `rg 'tl\.dot\(' fla/ops/**/triton_ascend/**` — only 8 op files use `tl.dot`; (2) for each lhs tile, flag lhs→lhs, lhs→rhs/store, or post-dot copy; (3) prefer GM reload for one reuse between stages, `+ 0.0` for tight multi-dot sequences; (4) re-run `tests/ops/test_gdn_kernels.py` + op-specific kernel tests.
- **Upstream**: lhs clobber is a Triton-Ascend backend limitation (UB capacity / in-place matmul), not intentional API. Durable fix belongs in the compiler (preserve lhs or emit a diagnostic on post-dot read). Track via the Triton-Ascend / Ascend backend issue tracker.
- For separable gate differences, compute `exp2(gs)[:, None] / exp2(gc)[None, :]` instead of `exp2(gs[:, None] - gc[None, :])` to replace a matrix of exponentials with two vectors. Verify numerics on the target compiler; multiplying by `exp2(-gc)` can produce materially different Ascend results.
- **Constexpr-split mutually exclusive DMA paths** (critical on Ascend): a runtime `if is_tail_chunk` that chooses `make_block_ptr` vs masked `tl.load` keeps **both** paths live in UB. Peak UB ≈ sum of both; Vector cannot saturate even when MemoryUB bandwidth is free; larger tiles then fail compile. Host-split the last tile into a second launch with `tl.constexpr TAIL_MODE` (`0` = never tail / block_ptr only, `1` = always masked, `2` = runtime for varlen / `NT==1`) so each compile DCE's the unused path. Case: [causal_conv1d](references/cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split).
- **MTE DMA past packed allocation**: `make_block_ptr` whose block end overshoots packed `B*T` rows faults MTE (`DDR address out of range`). Use masked load/store on the last chunk, or the constexpr split above so bulk never overshoots. Halo windows (`BT+W-1`) overshoot even sooner — count the halo in the tail predicate.
- **Do not OR a constexpr optional-pointer flag with a runtime check**: `if USE_INITIAL_STATE or i_t*BT < W` still lowers the else and compiles `initial_state + …` when the pointer is `None`. Nest: `if not FLAG: … elif runtime: … else: …`.
- **`tl.extract_slice` / `tl.insert_slice`**: sliding-window taps without extra GM loads (causal conv). Some triton-ascend versions expose them only via `triton.language.extra.cann.extension` — shim onto `tl` if missing. Preloading every tap tile overflows UB; load inside the `static_range` or one `BT+W-1` window + slice.
- **Weight `[D, W]` → host `transpose(0,1).contiguous()` to `[W, D]`** for stride-1 channel `block_ptr` (same idea as G_T_CONTIG). Odd `D` that cannot be tiled with a power-of-two `BD` that divides `D` and `BD>=16` falls back to the legacy multi-axis path.

### Fusion, compile, varlen

- Fuse only stages that share loads, cut traffic, and keep live set under control; split independent grad chains to ease UB.
- **Producer → consumer on the same output** (e.g. inter `o += q@h` then intra `o += A@v` with `ACCUMULATE_OUTPUT`): if both need the same `q` (and live set fits), fuse into one kernel — keep `b_o` / `b_A` in UB, single store. Profiler cue: two kernels own the op and MTE is high from the intermediate `o` writeback. If fused peak UB overflows, keep the split; do not force fusion.
- When fused live set is dominated by fixed tiles (e.g. `BT×BT` + `BT×BV`), fix the Cube-aligned outer tile (`BV`) and **autotune** the K-slab (`BK`) rather than host-hardcoding both.
- Multi-tile contribs to one grad: fp32 partials + deterministic finalize; atomics sparingly. `tl.debug_barrier` only for same-program deps.
- `do_not_specialize=['T']` (and other dynamic launch extents); kill runtime branches with `tl.constexpr` / `triton.heuristics`.
- **No `num_warps` / `num_stages` anywhere**: NPU does not support them. Omit from kernel call sites, autotune config dicts, and wrappers. Do not leave them commented-out “for CUDA parity”; delete them.
- Varlen is first-class: reuse `prepare_chunk_indices` / `prepare_chunk_offsets`. With 1D core-grid, flatten over `total_chunks` and map `global_t → (i_n, i_t)` via `chunk_offsets` (largest `i_n` with `chunk_offsets[i_n] <= global_t`). Tests cover empty tails, non-aligned lengths, multi-length, and fixed/varlen equivalence.

Failure modes and repo paths: [reference.md](references/reference.md). Detailed past cases: [cases.md](references/cases.md).

## 5. Verification loop

Each round, in order:

1. Single kernel vs Torch oracle (fp16/bf16, fwd+bwd).
2. Shape matrix: small/large T, non-aligned tiles, head sharing, gate/state, fixed/varlen.
3. End-to-end tests; confirm dispatch hits `triton_ascend`.
4. Frozen full pytest gate (incl. NaN poisoning); on failure, stop — do not claim speedups.
5. Synchronized benchmark (latency/throughput, fwd and fwd+bwd); re-profile with the same `aic_metrics` and confirm Duration/pipe/UB move as expected.
6. Metrics unchanged → reclassify bottleneck or switch metrics; do not pile unrelated changes.

Prefer: `tests/ops/test_gdn_kernels.py`, `tests/ops/test_solve_tril.py`, `tests/modules/test_conv.py` (causal_conv1d), `tests/utils/test_ascend_ub_manager.py`, `python -m benchmarks.ops.verify --op <op> --base <ref>` (`--gate-k` is a quick signal only).

### Round summary template

After re-profile, report:

- Target kernel Duration (before → after)
- Pipe ratios: Cube/MAC, Vector, scalar, MTE1, MTE2, MTE3
- UB bandwidth (if MemoryUB run collected)
- Any unsupported triton-ascend ops encountered and workarounds used
- Whether another round is warranted (per `fla-optimization-loop` stop criteria)

Generalizable fixes discovered during optimization belong in this skill (`SKILL.md`, `references/reference.md`, or `references/cases.md`) in a separate doc commit — not bundled into a perf PR.

## Review checklist

- [ ] Same algorithm/control flow as CUDA reference (tiling/grid/layout adaptations only)
- [ ] No Torch fallback on production paths; unsupported cases error / verifier rejects
- [ ] No `num_warps` / `num_stages` in Ascend kernel launches, autotune configs, or wrappers
- [ ] Backend registration, lazy import, public signatures correct
- [ ] Peak live tiles estimated; tiles from shared helpers + safety margin
- [ ] Grid ≤ 65535 **or** 1D core-grid (`num_aicore` Cube / `num_vectorcore` Vector); host-split offsets not double-counted with varlen; task-loop pointers rebound each iteration
- [ ] Runtime `block_ptr` vs masked DMA: constexpr-split so bulk DCE's the unused path; tail DMA does not overshoot packed `B*T` (include halo)
- [ ] Optional-pointer constexpr flags are nested, not `or`-ed with runtime checks (None ptr must not compile)
- [ ] Block pointers contiguous innermost; **gate `g` uses G_T_CONTIG** when `[B,T,HV]` (see [g-contiguous-loading.md](references/g-contiguous-loading.md)); tail `boundary_check`
- [ ] fp32 accum consistent with output/exp base; fusion worth the complexity (no gratuitous `ACCUMULATE_OUTPUT` writeback when UB allows)
- [ ] Reused `tl.dot` left-hand tiles: GM reload or `tile + 0.0` **before** first lhs dot (post-dot copy invalid); see [cases.md § tl.dot catalog](references/cases.md#tldot-lhs-clobber--repo-wide-case-catalog)
- [ ] fwd/bwd/varlen/layout branches covered; no unwritten regions under NaN poisoning
- [ ] Runtime indices (`NT`, `i_t`, `i_b`, `B`, program IDs) via `tl.cast(..., tl.int64)` **before** stride / `BT` / `D` multiply — including packed `offset * D`. Not gated on `do_not_specialize`. Do not call `.to(tl.int64)` on specialized kernel args (`constexpr` has no `.to`)
- [ ] Varlen `bos`/`eos` from `cu_seqlens` loaded as `tl.int64`; `T_cur = (eos - bos).to(tl.int32)` only; non-varlen `tl.cast(i_b, tl.int64) * T`
- [ ] `make_block_ptr` offsets/block_shape stay int32 (`i_t * BT`); int64 is only for flattened `ptr + offset * stride`
- [ ] Optional-arg paths exercised (e.g. `use_g` True/False with `g=None` reference) when PR touches gated and ungated paths
- [ ] Did not weaken tests/tolerances/benchmarks for “wins”; synced bench + re-profile on target NPU
- [ ] Round summary includes pipe/UB metrics (template above)

## Anti-patterns

- Copying profiler boilerplate into every `test_*.py`
- Treating async launch time as latency; missing warmup/synchronize
- Expecting Pipe and MemoryUB columns from a single run
- Tuning MTE before confirming the fused NPU kernel is hit
- Loosening tolerances, dropping cases, or editing benchmarks to fake speedups
- Adding or keeping `num_warps` / `num_stages` on Ascend paths (unsupported; not a tuning lever)
- Hiding unsupported triton-ascend ops without documenting workarounds
- Leaving a runtime `is_tail_chunk` (or similar) between `block_ptr` and masked DMA — both stay in UB
- Launching a Vector-bound kernel on `num_aicore` (half the vector cores idle on A2)
- `if CONSTEXPR_FLAG or runtime:` around an optional pointer — else still compiles when the ptr is None
- `B.to(tl.int64)` / `i_t.to(tl.int64)` on specialized or folded constexpr ints (`constexpr` has no `.to`); use `tl.cast`
- Passing int64 `t0` as `make_block_ptr` offsets (`offsets/block_shape` must be int32)

## Related files

- Collect / analyze: `scripts/profile_npu.py`, `scripts/analyze_profile.py`
- Metrics, failure modes, code index: [references/reference.md](references/reference.md)
- Past kernel case notes: [references/cases.md](references/cases.md)
- **Gate `g` stride-1 loading (G_T_CONTIG)**: [g-contiguous-loading.md](references/g-contiguous-loading.md)
- **causal_conv1d 1D core-grid + constexpr DMA split**: [cases.md § causal_conv1d](references/cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split)
- Ascend-specific traps (DMA dual-path UB, None-ptr compile, `constexpr` `.to`, int64 `block_ptr` offsets): [TRAPS.md](references/TRAPS.md)
- Ad-hoc workload output dir: `npu_prof/` (new collection must use the generic scripts)
