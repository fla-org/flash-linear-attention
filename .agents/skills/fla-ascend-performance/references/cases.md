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

## `chunk_o.py` — fwd fuse + bwd G_T_CONTIG

File: `fla/ops/common/backends/triton_ascend/chunk_o.py`

- Fwd: fuse inter+intra, 1D core-grid, host `g.transpose(1,2).contiguous()`.
- Bwd `G_T_CONTIG`: `chunk_bwd_dv_local_npu`, `chunk_bwd_dqkwg_npu`, `chunk_bwd_kernel_dg_npu` — stride-1 `g_ptr` (`i_b*HV*T+i_h*T` / varlen `bos+i_h*T`), `T_seq` before varlen.
- dv_local kernel ~6.5→0.18ms, dqkwg ~10.8→0.91ms (B2 T2048 HV8). Fix `BV`, autotune `BK`. Details: [g-contiguous-loading.md](g-contiguous-loading.md).

## `chunk_bwd.py` — varlen `cu_seqlens` int32 overflow

File: `fla/ops/kda/backends/triton_ascend/chunk_bwd.py`

- `chunk_kda_bwd_kernel_dAv_npu` and `chunk_kda_bwd_kernel_wy_v_part_npu` loaded `bos`/`eos` via `.to(tl.int32)` then `(bos * HV + i_hv) * V` — int32 wrap on packed varlen offsets (e.g. HV=32, V=4096 safe `bos` ≈ 16K). Later kernels in the same file already used `tl.int64`.
- Fix: load `bos`/`eos` as int64; `T = (eos - bos).to(tl.int32)`; non-varlen else-branch `(i_b * T).to(tl.int64)`.
