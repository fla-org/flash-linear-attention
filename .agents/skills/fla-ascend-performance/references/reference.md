# Ascend profiling and tuning reference

Follow the [main workflow](../SKILL.md) first. Use the sections below for the current step:

- [Profiling](#profiling): collection options, metrics, and output files.
- [Diagnosis](#diagnose-the-bottleneck): connect profiler signals to a hypothesis.
- [Tuning](#choose-a-tuning-change): UB, layout, grid, and fusion choices.
- [Correctness hazards](#correctness-and-compiler-hazards): address math, DMA, and compiler limitations.
- [Code and tests](#code-and-tests): implementation examples and verification entry points.

## Profiling

The bundled `scripts/profile_npu.py` collects traces; `scripts/analyze_profile.py` reads them. Paths are relative to this skill directory. Use the same active Python/NPU environment for collection, analysis, and benchmarks.

### Workload and collection

The CLI example in [SKILL.md](../SKILL.md#2-collect-a-profile) loads a file defining `workload()`. For library use, add the scripts directory to `PYTHONPATH` from the repository root:

```bash
export PYTHONPATH="$PWD/.agents/skills/fla-ascend-performance/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from profile_npu import profile_callable


def workload():
    y = op(...)
    y.backward(grad)


trace_dir = profile_callable(
    workload,
    name="my_op",
    out_dir="profile/my_op-npu",
    aic_metrics="PipeUtilization",
)
```

The default schedule is `wait=0, warmup=1, active=1, repeat=1`. Each collection supports one `aic_metrics`; collect again when another metric is needed.

| Metric                               | Inspect                                  | Use it to investigate        |
| ------------------------------------ | ---------------------------------------- | ---------------------------- |
| `PipeUtilization`                    | Cube/MAC, Vector, MTE, and scalar ratios | Which execution pipe is busy |
| `MemoryUB`                           | UB read/write bandwidth                  | UB bandwidth saturation      |
| `Memory`, `MemoryAccess`, `MemoryL0` | Memory-specific columns                  | Other memory paths           |
| `L2Cache`                            | L2 and instruction-cache counters        | Cache reuse                  |
| `ArithmeticUtilization`              | Arithmetic pipe counters                 | Arithmetic unit utilization  |
| `ResourceConflictRatio`              | Resource-conflict counters               | Resource stalls              |

### Read the outputs

The collector writes:

```text
{name}_profiling_{timestamp}/
  localhost..._ascend_pt/
    ASCEND_PROFILER_OUTPUT/
      op_statistic.csv
      kernel_details.csv
      operator_details.csv
      api_statistic.csv
      step_trace_time.csv
      trace_view.json
```

Analyze a trace or its parent directory from the repository root:

```bash
python .agents/skills/fla-ascend-performance/scripts/analyze_profile.py \
  profile/my_op-npu --kernel-filter my_kernel_substr
```

Start with total-time share in `op_statistic.csv`, then duration in `kernel_details.csv`. Cube/MAC utilization is reported by `aic_mac_ratio` or `cube_utilization(%)`; Vector utilization uses `aiv_vec_ratio`. Read the MTE and scalar columns alongside them. A low ratio matters most when the kernel also dominates elapsed time.

## Diagnose the bottleneck

Treat the following as hypotheses to verify with a targeted change and a second profile.

| Signal                                                    | Likely limit             | Investigate next                         |
| --------------------------------------------------------- | ------------------------ | ---------------------------------------- |
| High Vector ratio, little Cube activity                   | Vector work              | Row tiles, scalar work, load/store reuse |
| High Cube/MAC ratio                                       | Matrix compute           | Matmul tiles, alignment, prelude cost    |
| High MTE2/MTE3, low compute                               | Memory movement          | Strides, reuse, intermediate writebacks  |
| High scalar ratio                                         | Scalar work              | Vectorization and runtime branches       |
| High UB bandwidth, low Vector/Cube ratio                  | UB bandwidth             | Tile size and fusion                     |
| Many tiny ops, little target-kernel time                  | Dispatch or fusion       | Whether the intended backend ran         |
| Frequent host grid chunks                                 | Launch overhead          | A one-dimensional core grid              |
| Shared output written by two hot kernels                  | Intermediate traffic     | Producer/consumer fusion within UB       |
| Low Vector use, spare UB bandwidth, larger tiles overflow | Both DMA paths live      | Compile-time separation of bulk and tail |

For strided token-axis gate loads, read [contiguous gate loading](g-contiguous-loading.md). For bulk/tail DMA overlap, read [compiler traps](TRAPS.md#runtime-dma-path-if-keeps-both-sides-in-ub). Do not infer UB bandwidth saturation from capacity usage alone.

## Choose a tuning change

### UB budget and tiles

Estimate peak live storage, including fp32 accumulators, transpose copies, masks, and temporary dot products:

```text
peak_bytes ≈ memory_multiplier * tiled_elements * dtype_size
safe_utilization = peak_bytes / (ub_capacity * safety_margin)
```

Use `fla.utils.ascend_ub_manager` instead of hard-coding capacity. A safety margin around 0.75–0.85 is a starting point; explain the live buffers behind each memory multiplier and validate on the target compiler. Forward and backward often need different tile budgets.

Prefer power-of-two tiles and 16-aligned matrix tiles. If the safe budget remains underused, calibrate the multiplier or test non-power-of-two tiles. If capacity is nearly full but the kernel remains slow, inspect pipe and bandwidth metrics before increasing tiles. Split stages or recompute when the fused live set cannot fit reliably.

### Layout and grid

Keep the innermost loaded dimension contiguous and use the existing input-layout guards. Ascend backend files retain the repository's block-pointer exemption; boundary checks still need to cover allocation tails.

For gates shaped `[B, T, HV]`, token-axis loads may benefit from a contiguous `[B, HV, T]` copy and `G_T_CONTIG`. `HV == 1` needs no transpose. Preserve each sequence's original `T_seq` in variable-length paths and keep forward/backward pointer formulas consistent. See the [gate guide](g-contiguous-loading.md) for implementation details.

The Ascend grid-product cap is `ASCEND_MAX_GRID_DIM=65535`. Choose between:

- **Host chunking:** use `iter_axis_launch_chunks` and pass the matching offsets. After slicing variable-length indices, do not add the same global offset again.
- **Core grid:** flatten independent tiles into tasks and iterate `task_id` by core stride. Use `num_aicore` for Cube work and `get_multiprocessor_count` (`num_vectorcore` on NPU) for Vector work. On A2, these are 24 Cube and 48 Vector cores.

In a core-grid loop, rebuild local pointers from the original bases on every task; accumulating `ptr += offset` across tasks can miscompile. Keep dynamic extents such as `T`, `task_num`, and `num_core` unspecialized. Reuse `prepare_chunk_indices` and `prepare_chunk_offsets` to map global chunks to sequences.

### Fusion and specialization

Fuse stages when they share loads and avoid an intermediate writeback without exceeding UB. For example, inter- and intra-chunk output contributions can share `q` and one final store. If fixed tiles dominate the live set, hold the outer Cube-aligned tile fixed and autotune the reduction slab. Split independent gradient chains when their combined live set is too large.

Preserve fp32 reduction and gradient accumulation where the validated implementation uses it. Combine partial gradients deterministically; use atomics sparingly. `tl.debug_barrier` only synchronizes dependencies within a program.

Use `tl.constexpr` or `triton.heuristics` for feature flags and `do_not_specialize=['T']` for sequence length. Omit `num_warps` and `num_stages` from Ascend launches and autotune configurations.

## Correctness and compiler hazards

### Address math and variable lengths

Cast runtime indices to `tl.int64` **before** multiplying by dimensions or strides. Casting the product afterwards cannot repair an overflow. Prefer `tl.cast(index, tl.int64)` because specialized arguments and folded program IDs may be `constexpr` values without `.to()`.

Load `cu_seqlens` entries as int64 for address math even when the host tensor is int32. Keep sequence lengths separate from absolute offsets. For example, `(NT - 1) * HV * K * V` can overflow at modest chunk counts, and `(bos * HV + head) * V` can overflow before `bos` reaches the int32 limit.

Block-pointer metadata is different: `make_block_ptr` offsets and block shapes must remain int32. Use int64 for the flattened base pointer and a bounded int32 offset inside that block. See [addressing traps](TRAPS.md).

For failures limited to long or mixed-length inputs, also check duplicated offsets after slicing, chunk-to-sequence mapping, and pointers carried across task-loop iterations.

### DMA tails and optional pointers

A runtime branch between block-pointer and masked DMA can keep both paths live in UB. Separate bulk and tail launches with a constexpr mode so the compiler removes the unused path. In the convolution example, `TAIL_MODE` selects bulk, masked, or runtime handling; its exact policy is workload-specific.

A block or halo window extending past packed `B*T` rows can cause an MTE `DDR address out of range` fault. Include the halo in the tail predicate and use masked loads/stores for the tail.

Do not combine an optional-pointer constexpr flag and a runtime condition with `or`: the compiler may still lower arithmetic on a `None` pointer. Give the constexpr flag its own branch. See [compiler traps](TRAPS.md) and the [convolution case](cases.md#causal_conv1d--1d-core-grid--constexpr-dma-split).

### Matmul input reuse and numerical failures

Ascend `tl.dot` may overwrite its left operand in UB. Audit every later use as another dot operand, a stored value, or arithmetic input. Reload from global memory between stages, or create disposable copies with `tile + 0.0` **before** the first dot; a copy afterwards preserves the corrupted value.

Find affected calls with:

```bash
rg 'tl\.dot\(' fla/ops fla/modules \
  --glob '**/triton_ascend/**' --glob '**/triton_ascend.py'
```

Use [the per-kernel cases](cases.md#tldot-lhs-clobber--repo-wide-case-catalog) to select the workaround, then run the operator's reference tests. This is a compiler limitation, not an intended matmul API contract.

For NaNs or drift, check masks before exponentiation, exp/exp2 scaling, accumulation precision, solve precision, and scratch initialization. Gate factorization into vector exponentials can reduce work, but division and multiplication by a negated exponential may differ numerically on Ascend. Validate the exact expression under the existing tolerance.

### Convolution window loads

`extract_slice` and `insert_slice` can reuse a loaded `BT+W-1` window. Some triton-ascend versions expose them through `triton.language.extra.cann.extension`. Loading every tap tile at once can overflow UB; use one window or load taps inside the loop.

A contiguous weight copy from `[D, W]` to `[W, D]` enables stride-one channel loads. Preserve the existing fallback for channel widths that cannot satisfy the chosen tile's alignment and divisibility. Detailed examples belong in the [convolution case notes](cases.md#causal_conv1d--1d-core-grid--constexpr-dma-split).

## Code and tests

Paths below are relative to the repository root. The [case catalog](cases.md) contains operator-specific measurements and compiler workarounds.

| Need                         | Start here                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| Registration and selection   | `fla/backends.py` and the [dispatch skill](../../fla-dispatch-backends/SKILL.md)       |
| Module registration example  | `fla/modules/norm/triton_ascend/l2norm.py`                                             |
| UB budgeting and grid splits | `fla/utils/ascend_ub_manager.py`                                                       |
| Tiled output and gradients   | `fla/ops/common/backends/triton_ascend/chunk_o.py`                                     |
| Recurrence and gate examples | `fla/ops/common/backends/triton_ascend/chunk_delta_h.py` and `chunk_scaled_dot_kkt.py` |
| Multi-stage gradient work    | `fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py` and `chunk_fwd.py`        |
| KDA fusion and matmul reuse  | `fla/ops/kda/backends/triton_ascend/wy_fast.py`, `chunk_intra.py`, and `chunk_bwd.py`  |
| Solves and cumulative sums   | `fla/ops/utils/backends/triton_ascend/solve_tril.py` and `cumsum.py`                   |
| Convolution core grid        | `fla/modules/causal_conv1d/backends/triton_ascend.py`                                  |

Use `tests/ops/test_gdn_kernels.py` and `tests/ops/test_solve_tril.py` for relevant kernel comparisons, `tests/modules/test_conv.py -k 'not cuda'` for NPU convolution, and `tests/utils/test_ascend_ub_manager.py` for tile/grid boundaries. Include ungated calls (`g=None`) when supported, plus empty tails, non-aligned lengths, and fixed/variable-length equivalence. A local pass followed by NaNs under the full gate often indicates unwritten tails or uninitialized scratch.

`tests/conftest.py` provides NaN poisoning. `python -m benchmarks.ops.verify --op <op> --base <ref>` runs the correctness-gated comparison; `--gate-k` is an early signal only, and `--no-gate` cannot justify promotion. Timing entry points live in `benchmarks/ops/run.py` and `registry.py`; `.github/workflows/ascend-a2-ci.yml` defines A2 coverage.
