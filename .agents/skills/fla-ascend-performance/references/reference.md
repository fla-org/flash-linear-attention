# Ascend Profiling + Optimization Reference

Diagnosis table (profiler signal → bottleneck → fix): see [SKILL.md §3](../SKILL.md#3-diagnose-bottlenecks).

## Choosing AiCMetrics

Each profiling run supports **exactly one** `aic_metrics`. Collect multiple times when you need different evidence.

| metrics | Key CSV columns | Question answered |
|---------|-----------------|-------------------|
| `PipeUtilization` (default first pass) | `aic_mac_ratio`, `aiv_vec_ratio`, `*_mte1/2/3_ratio`, `*_scalar_ratio`, `cube_utilization(%)` | Compute vs move vs scalar dominance |
| `MemoryUB` | `aiv_ub_read/write_bw_*`, `aic_ub_*` | UB bandwidth saturation / R/W imbalance |
| `Memory` / `MemoryAccess` / `MemoryL0` | matching memory columns | Finer memory paths |
| `L2Cache` | L2 / icache related | Cache hit / reuse |
| `ArithmeticUtilization` | arithmetic pipes | Arithmetic unit busy time |
| `ResourceConflictRatio` | conflict related | Whether resource conflicts stall |

Colloquial “CUDA utilization” → Ascend **Cube / MAC** (`aic_mac_ratio` / `cube_utilization(%)`).

## Output layout

```
{name}_profiling_{timestamp}/
  localhost..._ascend_pt/
    ASCEND_PROFILER_OUTPUT/
      kernel_details.csv
      op_statistic.csv
      operator_details.csv
      api_statistic.csv
      step_trace_time.csv
      trace_view.json
```

## Host UB model

```text
peak ≈ memory_multiplier * BT * BD * dtype_size
util = peak / ub_capacity
safe_util = peak / (ub_capacity * safety_margin)
```

| Host signal | Next step |
|-------------|-----------|
| `safe_util` underused | Calibrate `mem_mult`, non-PoT tiles; measure compile-safe UB limit |
| `safe_util` ≈ 100% still slow | Not a capacity issue — revisit pipe/bandwidth (SKILL §3) |

After changing `mem_mult`/tiles, always re-check compile + numeric correctness.

## Common failures and fixes

- **UB overflow**: fewer concurrent fp32 tiles; smaller BK/BV; split kernels; avoid accidental large broadcasts.
- **Grid limit**: host-split axes + offsets, **or** switch to 1D core-grid (`num_aicore` × flat `task_num`); do not only grow tiles.
- **Varlen wrong only on long seqs**: after slicing `chunk_indices`, check for a second global `NT_OFFSET`; on core-grid paths, verify `chunk_offsets` → `(i_n, i_t)` against `cu_seqlens`.
- **Wrong results only in multi-task core-grid loops**: rebind local base pointers each `task_id`; avoid in-place `ptr +=` across iterations.
- **Occasional bf16 NaN**: mask before exp, fp32 accum, exp/exp2 scale, solve precision.
- **`tl.dot` left operand clobbered (Ascend only)**: `tl.dot(lhs, rhs, …)` may mutate `lhs` in UB (CUDA does not). Any later read of that tile — second lhs, rhs, store, or arithmetic — can see corrupted data. Two fixes: **GM reload** between stages (e.g. `wy_fast` u→w on `b_A`) or **`tile + 0.0` before the first lhs dot** when multiple disposable copies are needed in tight sequence. Post-dot `+ 0.0` is invalid. Full per-kernel catalog: [cases.md § tl.dot lhs clobber](cases.md#tldot-lhs-clobber--repo-wide-case-catalog). Symptom: numeric mismatch vs Torch, no compile error. Tests: `test_gdn_kernels.py`, `test_solve_tril.py`.
- **Correct but slower**: launch count (split inter/intra + host grid chunks), tiny tiles, full-size fp32 scratch, extra layout converts, unsynced fake baselines.
- **Local pass, full gate NaN**: tail writeback, boundary masks, invalid exp regions, scratch init before read.
- **Compile-variant explosion**: do not specialize on T; move feature flags to heuristics/constexpr.
- **`num_warps` / `num_stages` on NPU**: unsupported by Ascend Triton — remove from launches/autotune; never use as an optimization knob.
- **int32 chunk-address overflow**: `NT = cdiv(T, BT)` under `do_not_specialize` is int32; `(NT-1)*HV*K*V` wraps before `.to(tl.int64)` on long context (K=V=128, HV=64 → T>131K). Fix: `(NT-1).to(tl.int64)*DH_CS` (same rule for any runtime `i_t`/offset × stride).
- **Ungated path untested**: kernel tests that always pass `g` miss `g=None` regressions (in-place `dv`, scale folds, tiling). Parametrize `use_g` on the existing test + `g=None` reference branch — no new CI file required.
- **Unsupported triton-ascend ops**: work around in-kernel; list every unsupported op in the round summary so follow-ups can track compiler gaps.

## Repo code index

Paths relative to the `flash-linear-attention` repo root. Detailed case notes: [cases.md](cases.md).

### Backend wiring

- `fla/ops/backends/__init__.py` — `BaseBackend` / `dispatch` / verifier
- `fla/ops/common/backends/triton_ascend/__init__.py` — `IS_NPU` + lazy import
- `fla/ops/gated_delta_rule/backends/triton_ascend/__init__.py` — multi-function backend example

### UB / tile / grid

- `fla/utils/ascend_ub_manager.py` — `compute_row_tile_block_size`, `iter_axis_launch_chunks`
- `fla/ops/common/backends/triton_ascend/chunk_scaled_dot_kkt.py` — peak tile, BC, UB-safe BK
- `fla/ops/common/backends/triton_ascend/chunk_delta_h.py` — fwd recurrence, V tiling, bwd `dhu` (see [cases.md](cases.md))
- `fla/ops/common/backends/triton_ascend/chunk_o.py` — fwd fuse + bwd G_T_CONTIG (see [cases.md](cases.md))
- `fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py` — multi-stage bwd; **`tl.dot` lhs clobber** (GM reload + copy) — [cases.md](cases.md)
- `fla/ops/kda/backends/triton_ascend/wy_fast.py` — KDA variant of wy_fast; same clobber patterns
- `fla/ops/kda/backends/triton_ascend/chunk_intra.py` — inter solve fused; multi-copy block merge — [cases.md](cases.md)
- `fla/ops/kda/backends/triton_ascend/chunk_bwd.py` — KDA bwd (`dAv`, wy dw/dqkg); `b_do_c` pattern — [cases.md](cases.md)
- `fla/ops/utils/backends/triton_ascend/solve_tril.py` — blocked triangular solve; 32×32/64×64 merge clobber guards — [cases.md](cases.md)
- `fla/ops/utils/backends/triton_ascend/cumsum.py` — scalar/vector split and leftover UB budget

### Split / numerics / varlen

- `fla/ops/gated_delta_rule/backends/triton_ascend/chunk_fwd.py` — stage split for UB
- `fla/ops/utils/backends/triton_ascend/solve_tril.py` — blocked triangular solve, ieee/RTNE
- `fla/ops/gated_delta_rule/backends/triton_ascend/gate.py` — heuristics, `do_not_specialize=['T']`

### Tests and benchmarks

- `tests/ops/test_gdn_kernels.py` — per-kernel oracle
- `tests/utils/test_ascend_ub_manager.py` — tiling/grid boundaries
- `tests/conftest.py` — NaN memory poisoning
- `benchmarks/ops/verify.py` — correctness-gated benchmark (do not claim wins with `--no-gate`)
- `benchmarks/ops/run.py` / `registry.py` — unified timing entrypoints
- `.github/workflows/ascend-a2-ci.yml` — A2 CI; new ops need their own test entrypoints

## Environment

Use the Python/NPU environment already active in the current terminal (including any activated conda/venv). Run collection, analysis, and benchmarks in the same shell; do not spawn a new shell or switch environments mid-workflow. If the terminal has no NPU stack loaded yet, activate the project's Ascend environment first, then continue in that same session.
