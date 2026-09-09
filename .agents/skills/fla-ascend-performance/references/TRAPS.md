# Ascend Triton traps

Silent-bug and compile/UB traps specific to Triton-Ascend. General FLA measurement traps stay in `fla-optimization-loop/references/TRAPS.md`. Case notes: [cases.md](cases.md).

Each entry: **Fact / Why / How to apply**.

---

## Runtime DMA-path `if` keeps both sides in UB

**Fact:** A runtime `if is_tail_chunk:` that chooses `make_block_ptr` vs masked `tl.load` does **not** DCE the unused path on Triton-Ascend. Peak UB is the sum of both; Vector stays ~0.75 occupied; larger tiles fail compile even when MemoryUB bandwidth is free.

**Why:** The compiler treats the predicate as data-dependent, so both DMA sequences stay in the live set. Halo windows (`BT+W-1`) make the tail path even larger. The same class of bug: `if CONSTEXPR_FLAG or runtime:` around an optional pointer still compiles `ptr + …` when `ptr is None`.

**How to apply:** Host-split the last tile into a second launch with a `tl.constexpr` mode (never / always / runtime-for-varlen). Nest constexpr optional-pointer flags; do not OR them with runtime checks. See [cases.md § causal_conv1d](cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split).

---

## `constexpr` has no `.to()` — use `tl.cast` for address math

**Fact:** Specialized kernel args (`B`, `T`) and program IDs that fold (e.g. `i_t` when `NT==1`) are `constexpr`. `x.to(tl.int64)` is `AttributeError("'constexpr' object has no attribute 'to'")` at compile. CUDA kernels often write `i_t.to(tl.int64)` because those indices stay runtime there.

**Why:** Ascend specializes more integers than CUDA. `tl.cast(x, tl.int64)` works on constexpr and runtime ints; `.to()` only exists on tensor / load results.

**How to apply:** `t0 = tl.cast(i_t, tl.int64) * BT`, `bos = tl.cast(i_b, tl.int64) * T`, `tl.cast(B, tl.int64) * T`. Keep `tl.load(cu_seqlens + i_n).to(tl.int64)`. Never `(i_b * T).to(tl.int64)`.

---

## Int64 `make_block_ptr` offsets fail compile

**Fact:** `make_block_ptr` rejects int64 `offsets` / `block_shape` (`Block pointers only support 32 bit offsets/block_shape`). Feeding `t0 = tl.cast(i_t, tl.int64) * BT` as the row offset breaks compile after the overflow fix.

**Why:** Block-pointer metadata is int32 by design. Flattened `ptr + offset * stride` is the path that needs int64.

**How to apply:** Keep `t0` (int64) for `x + bos * D + t0 * D`. Pass `i_t * BT` (int32) to `make_block_ptr`. See [cases.md § causal_conv1d](cases.md#causal_conv1dpy--1d-core-grid--constexpr-dma-split).
