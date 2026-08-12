# Ascend optimization case notes

Experience notes from past kernel work. Read the current code before applying — numbers are not immutable hardware constants. Paths are relative to the `flash-linear-attention` repo root.

## `chunk_delta_h.py` — bwd `dhu`

File: `fla/ops/common/backends/triton_ascend/chunk_delta_h.py`

- Host-transpose `g` → `[B,HV,T]` (stride-`HV` gather was ~30–45× slower; Pipe: tiny `aic_mac_ratio`, huge `aiv_mte3`/`aiv_scalar`). See [g-contiguous-loading.md](g-contiguous-loading.md).
- Fold gate+scale into `do` once; host-precompute `g_exp`/`g_ratio=exp2(g_last-g)` when `T%BT==0` (log-space only — **not** `exp2(g_last)/exp2(g)`).
- Avoid in-place `ptr -=` across chunks (Ascend miscompile/NaN). `tl.advance` + vectorized last-lane extract **regressed** — keep remake-`block_ptr` + scalar `bg_last_exp`.
- Dropping `boundary_check` via aligned constexpr flags was **neutral** (cost is address-gen, not masks).
- **In-place `dv`**: callers rebind `dh,dh0,dv = bwd_dhu(...)`; store corrected residual through `p_dv` (no `dv2` buffer/ptr).
- **Tile selection is cost-model, not max-BK**: minimize `cdiv(V,BV)*(3+4*cdiv(K,BK))` under soft UB (`peak <= UB*1.15`, multi-slab dh live) on the **host-precomputed gate** path. Max oneslab `BK=min(256,pow2(K))` with BV capped at 64 loses on D256 to **BK=128/BV=128** (nv 4→2 beats extra K-slab ptrs). Shrinking BV only to grow BK (e.g. BV 64→32) remains a loss. Measured B8 T2048 H32 bf16: D256 ~15.5→11.4ms, D128 ~6.2→3.3ms; still AIC-scalar dominated (`aic_scalar_ratio`~0.66, `aic_mac_ratio`~0.07).
- **Gate-inline UB (T%BT!=0 / varlen)**: in-kernel `exp2(g)` inflates compile UB ~1.7× vs analytical peak — `BK=128/BV=128` fails (`req 262400 > 196608`) on D128 unaligned even though host util≈0.75. Use `gate_inline` → soft cap `UB*0.60` (D128→`BK=128/BV=64`, D256→`BK=256/BV=32`); keep precomp tiles unchanged.
- **`tl.dot` lhs clobber (bwd `dhu`)**: `b_do` reused as lhs across K-slabs; `b_dv` corrected in-place then subtracted. See [§ tl.dot catalog — chunk_delta_h](cases.md#chunk_delta_hpy).

## `chunk_o.py` — fwd fuse + bwd G_T_CONTIG

File: `fla/ops/common/backends/triton_ascend/chunk_o.py`

- Fwd: fuse inter+intra, 1D core-grid, host `g.transpose(1,2).contiguous()`.
- Bwd `G_T_CONTIG`: `chunk_bwd_dv_local_npu`, `chunk_bwd_dqkwg_npu`, `chunk_bwd_kernel_dg_npu` — stride-1 `g_ptr` (`i_b*HV*T+i_h*T` / varlen `bos+i_h*T`), `T_seq` before varlen.
- dv_local kernel ~6.5→0.18ms, dqkwg ~10.8→0.91ms (B2 T2048 HV8). Fix `BV`, autotune `BK`. Details: [g-contiguous-loading.md](g-contiguous-loading.md).
- **`tl.dot` lhs clobber**: fwd `b_q`/`b_q_c`; bwd `b_k0_c`, `b_ds_c`, `b_A_pristine`. See [§ tl.dot catalog](cases.md#chunk_opy).

## `chunk_bwd.py` — varlen `cu_seqlens` int32 overflow

File: `fla/ops/kda/backends/triton_ascend/chunk_bwd.py`

- `chunk_kda_bwd_kernel_dAv_npu` and `chunk_kda_bwd_kernel_wy_v_part_npu` loaded `bos`/`eos` via `.to(tl.int32)` then `(bos * HV + i_hv) * V` — int32 wrap on packed varlen offsets (e.g. HV=32, V=4096 safe `bos` ≈ 16K). Later kernels in the same file already used `tl.int64`.
- Fix: load `bos`/`eos` as int64; `T = (eos - bos).to(tl.int32)`; non-varlen else-branch `(i_b * T).to(tl.int64)`.

## `wy_fast.py` — Ascend `tl.dot` left-operand clobber

Files: `fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py`, `fla/ops/kda/backends/triton_ascend/wy_fast.py`

See also the full catalog in [§ `tl.dot` lhs clobber — repo-wide case catalog](#tldot-lhs-clobber--repo-wide-case-catalog) below.

- **Behavior**: Ascend `tl.dot(lhs, rhs, ...)` can overwrite `lhs` in UB. CUDA Triton leaves `lhs` intact.
- **Symptom / cost**: numeric mismatch vs Torch on Ascend only; no compile error. One extra GM load per stage is cheaper than keeping a duplicate tile live for the whole kernel.

### Strategy A — GM reload (large tile, reused once between stages)

| Kernel | Exposure | Fix |
|--------|----------|-----|
| `recompute_w_u_fwd_kernel_npu` / `recompute_w_u_fwd_kda_kernel_npu` | `u` loop: `b_u = tl.dot(b_A, b_vb, …)` clobbers `b_A`; `w` loop needs pristine `A` | `b_A = tl.load(p_A, …)` before each K-tile `w` dot (also reload each V-tile in `u` loop if needed) |
| `prepare_wy_repr_bwd_kv_npu` | `k` loop clobbers `b_A`; `v` loop reuses it | Reload `b_A` from GM before the `v` loop |

### Strategy B — pristine copy before first lhs dot (same stage, tight reuse)

| Kernel | Exposure | Fix |
|--------|----------|-----|
| `prepare_wy_repr_bwd_kv_npu` | `b_dw` lhs in `b_dA += tl.dot(b_dw, …)`, then rhs in `tl.dot(b_A, b_dw, …)` | `b_dw_c = b_dw + 0.0` **before** first dot; second dot uses `b_dw_c` |
| same | `b_du` lhs then rhs in V loop | `b_du_c = b_du + 0.0` before first dot |
| `prepare_wy_repr_bwd_finalize_npu` | `b_dA` lhs in `tl.dot(b_dA_lhs, b_k, …)`; same tile needed as rhs in `tl.dot(tl.trans(b_kb), b_dA_c, …)` | `b_dA_c = b_dA + 0.0` and `b_dA_lhs = b_dA + 0.0` before any dot |

### Copy-ordering rule (applies to all kernels)

- `tile + 0.0` must happen **before** the first `tl.dot` that uses `tile` as lhs — copying after the first dot captures the clobbered UB value, not the original.
- lhs clobber corrupts the tile for **any** subsequent read (rhs, store, arithmetic), not only a second lhs use.

### Multi-use patterns (each role needs its own pristine copy, all taken before any dot)

| Pattern | Example |
|---------|---------|
| lhs → lhs again | `b_k0` in `chunk_o` dv_local; `b_A_42`/`b_A_43` in `solve_tril` 64×64 |
| lhs → rhs | `b_dw`/`b_du` in `wy_fast` bwd; `b_do` in `chunk_bwd` dAv |
| lhs → store | `b_Ai_33_c` in `chunk_intra` merge; `b_Ai_22_c` in `solve_tril` |
| lhs on copy A, then rhs on copy B from same source | `b_Ai22_c2` lhs in `b_Ai20`, `b_Ai22_c3` rhs in `b_Ai32` |
| lhs across K-slabs (no GM reload) | `b_do` / `b_do_c{,2,3}` in `chunk_delta_h` bwd |
| lhs in expr₁, same tile lhs again in expr₂ | `b_Akk31`/`b_Akk32` in `b_Ai31` then `b_Ai30` → `b_Akk31_c`/`b_Akk32_c` for second expr |

### Safe patterns (look like reuse but are not exposures)

- **GM reload each iteration**: `b_w`/`b_k` K-segments in `chunk_delta_h` fwd; `b_k` in `prepare_wy_repr_bwd_finalize_a2_npu`; `b_A` per tile in `wy_fast` inner loops.
- **Fresh load per loop trip**: `b_dAqk`/`b_dAkk` in `chunk_intra` bwd (each `i_j` reloads from GM; halves separated by `tl.debug_barrier()`).
- **Single lhs use**: `chunk_scaled_dot_kkt` — `b_ks` lhs once per `(s,c)` block.

---

## `tl.dot` lhs clobber — repo-wide case catalog

**Audit scope (2026-08):** only **8 files** under `fla/ops/**/triton_ascend/**` contain `tl.dot` (~143 call sites). All others (layernorm, rotary, gate, fused_recurrent, cumsum, …) are unaffected.

**Verification:** `tests/ops/test_gdn_kernels.py`, `tests/ops/test_solve_tril.py`.

### `chunk_o.py`

File: `fla/ops/common/backends/triton_ascend/chunk_o.py`

| Kernel / site | Pattern | Fix |
|---------------|---------|-----|
| `chunk_fwd_kernel_o_npu` | `b_q` lhs for `b_o` dot, then needed for `b_A` dot | `b_q_c = b_q + 0.0` **before** first dot; `b_o` uses `b_q`, `b_A` uses `b_q_c` |
| `chunk_bwd_kernel_dv_npu` | `b_A` reused across V tiles as lhs | `b_A_pristine = b_A + 0.0`; per V tile: `b_A_i = b_A_pristine + 0.0` |
| `chunk_bwd_kernel_dv_local_npu` | `b_k0` lhs in `b_A00` and `b_A01` | `b_k0_c = b_k0 + 0.0` before dots; second dot uses `b_k0_c` |
| `chunk_bwd_dqkwg_npu` | `b_ds` lhs in `b_dq_r += tl.dot(b_ds, b_k_c)`, then lhs of `tl.dot(tl.trans(b_ds_c), b_q_r)` | `b_ds_c = b_ds + 0.0` **before** first `b_ds` dot |

### `chunk_delta_h.py`

File: `fla/ops/common/backends/triton_ascend/chunk_delta_h.py`

| Kernel / site | Pattern | Fix |
|---------------|---------|-----|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu` | `b_w`/`b_k` across K segments | **Safe** — each segment `tl.load`s a new tile from GM |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu` | `b_do` lhs across K-slabs (`STATE_V_FIRST` and else) | After gate×scale fold: `b_do_c/b_do_c2/b_do_c3 = b_do + 0.0` before any slab dot; slab1=`b_do`, slab2=`b_do_c`, slab3=`b_do_c2`, slab4=`b_do_c3` |
| same | `b_dv` updated in-place then subtracted in dh update | `b_dv_pristine = b_dv + 0.0` after dv correction; each slab: `b_dv_i = b_dv_pristine + 0.0` for the subtract dot |

### `chunk_intra.py`

File: `fla/ops/kda/backends/triton_ascend/chunk_intra.py`

| Kernel / site | Pattern | Fix |
|---------------|---------|-----|
| `chunk_kda_fwd_kernel_inter_solve_fused_npu` (NC≥3) | `b_qg2`/`b_kg2` lhs twice (Aqk20/Aqk21) | Copies before first dot: `b_qg2_c`, `b_kg2_c` |
| same (NC≥4) | `b_qg3`/`b_kg3` lhs three times | `b_qg3_c1/c2`, `b_kg3_c1/c2` before any dot |
| same — inter solve | `b_Ai22` lhs then store; lhs on copy then rhs on another copy | `b_Ai22_c` (store), `b_Ai22_c2` (lhs `b_Ai20`), `b_Ai22_c3` (rhs `b_Ai32`) — all before solve |
| same — inter solve (NC≥4) | `b_Ai33` lhs/store/lhs roles; `b_Akk31`/`b_Akk32` lhs in both `b_Ai31` and `b_Ai30` | `b_Ai33_c/c2/c3` for store/`b_Ai31`/`b_Ai30` lhs; `b_Akk31_c`/`b_Akk32_c` for second expr in `b_Ai30` |
| `chunk_kda_bwd_kernel_intra_npu` | Two kernel halves reuse tile names | **Safe** — `tl.debug_barrier()` + GM reload each loop trip |

### `solve_tril.py`

File: `fla/ops/utils/backends/triton_ascend/solve_tril.py`

| Kernel / site | Pattern | Fix |
|---------------|---------|-----|
| `merge_16x16_to_32x32_inverse_kernel_npu` | `b_Ai_22` lhs in inner dot of `b_Ai_21`, stored at end | `b_Ai_22_c = b_Ai_22 + 0.0` before solve; store `b_Ai_22_c` |
| `merge_16x16_to_64x64_inverse_kernel_npu` | `b_Ai_22` lhs in `b_Ai_21`, then rhs in `b_Ai_32` | Outer `b_Ai_32` uses `b_Ai_22_c` not `b_Ai_22` |
| same | `b_Ai_33_c` rhs in `b_Ai_43`, lhs in `b_Ai_31`, stored | Split: `b_Ai_33_c` (store+rhs), `b_Ai_33_c2` (lhs `b_Ai_31`) |
| same | `b_Ai_44_c` lhs in `b_Ai_42` and `b_Ai_41` | `b_Ai_44_c2` for second lhs |
| same | `b_A_42`/`b_A_43` lhs in `b_Ai_42` inner sum and again in `b_Ai_41` | `b_A_42_c`/`b_A_43_c` for `b_Ai_41` dots |

### `chunk_bwd.py` (KDA)

File: `fla/ops/kda/backends/triton_ascend/chunk_bwd.py`

| Kernel / site | Pattern | Fix |
|---------------|---------|-----|
| `chunk_kda_bwd_kernel_dAv_npu` | `b_do` lhs in `b_dA`, rhs in `b_dv` | `b_do_c = b_do + 0.0` before first dot |
| `chunk_kda_bwd_kernel_wy_k_part_npu` | `b_v_new` in V loop | **Safe** — fresh `tl.load` each `i_v` |
| `chunk_kda_bwd_kernel_wy_dw_part_npu` | `b_dw_c` in inner loop | **Safe** — `tl.load(p_dw_c, …)` each `s_c` trip |
| `chunk_kda_bwd_kernel_wy_dA_dot_npu` / finalize | single dot per launch | **Safe** |

### `chunk_scaled_dot_kkt.py`

File: `fla/ops/common/backends/triton_ascend/chunk_scaled_dot_kkt.py`

- `b_ks` lhs once per `(s,c,k)` accumulation — **no exposure**.

### Audit checklist (for new/changed kernels)

1. `rg 'tl\.dot\(' fla/ops/**/triton_ascend/**` — list all call sites in the kernel.
2. For each lhs tile: track first lhs dot line; flag any later read (lhs, rhs, store, `+`/`-`).
3. Confirm every `+ 0.0` copy is **before** the tile's first lhs dot.
4. Prefer **GM reload** when the tile is large and reused once between stages; prefer **`+ 0.0`** when multiple disposable lhs copies are needed in tight sequence (block merges, K-slabs).
5. Re-run `tests/ops/test_gdn_kernels.py` + any op-specific kernel tests.

### Upstream

lhs clobber is a Triton-Ascend backend limitation (UB capacity / in-place matmul), not intentional API. Durable fix: compiler preserves lhs or emits a diagnostic on post-dot read. Track via Triton-Ascend / Ascend backend issue tracker.
