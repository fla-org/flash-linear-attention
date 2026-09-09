# Gate `g` contiguous T-loading (G_T_CONTIG)

Apply when a Triton-Ascend bwd/fwd kernel loads gate `g` along the **time axis** but `g` is laid out as `[B, T, HV]`. Strided loads with inner stride `HV` (e.g. `block_ptr(..., stride=(HV,))`) are often **orders of magnitude slower** on Ascend than stride-1 contiguous loads — even when Cube utilization looks high.

Reference implementation: `fla/ops/common/backends/triton_ascend/chunk_o.py` (`chunk_fwd_kernel_o_npu`, `chunk_bwd_kernel_dv_local_npu`, `chunk_bwd_kernel_dqkwg_npu`, `chunk_bwd_kernel_dg_npu`).

## Symptom

- Kernel Duration in the **milliseconds** range for modest shapes (e.g. B=2, T=2048, HV=8) while Cube ~98%
- PipeUtilization: high **MTE2** (aiv/aic), high **scalar**; fix does not require larger tiles
- A/B: same kernel with `G_T_CONTIG=False` vs `True` shows **10×–35×** wall-clock gap on kernel-only timing

## Root cause

`g[b, t, h]` is stored with stride `HV` along `T`. A block load of `g[t0:t0+BC, h]` becomes a **gather** (stride `HV`), which Ascend MTE handles poorly in hot loops (many repeated small loads in bwd sub-block loops).

## Fix: host transpose + kernel stride-1 pointer

### 1. Host wrapper

```python
g_t_contig = g is not None and HV != 1
g_arg = g.transpose(1, 2).contiguous() if g_t_contig else g
# pass g=g_arg, G_T_CONTIG=g_t_contig to kernel
```

- **HV == 1**: skip transpose (stride along T is already 1).
- Transpose cost is ~10–15 µs — negligible vs multi-ms saved.
- Output tensors (`dg`, etc.) stay in original `[B, T, HV]` layout; only **input g reads** use transposed storage.

### 2. Kernel: `G_T_CONTIG` constexpr

Save packed length **before** varlen overwrites `T`:

```python
T_seq = T
if IS_VARLEN:
    ...
    T = eos - bos   # local sequence length for block_ptr bounds
```

**Pointer** (must match `chunk_fwd_kernel_o_npu` — do not reuse `g += bos * HV + i_h` with transposed storage):

| Mode | `g_ptr` |
|------|---------|
| Fixed batch | `g + i_b * HV * T_seq + i_h * T_seq` |
| Varlen (packed) | `g + bos + i_h * T_seq` |

**Block loads** (stride 1 along T):

```python
p_g = tl.make_block_ptr(g_ptr, (T,), (1,), (i_t * BT,), (BT,), (0,))          # full chunk
p_gr = tl.make_block_ptr(g_ptr, (T,), (1,), (i_tc_r,), (BC,), (0,))         # sub-block
b_g_last = tl.load(g_ptr + last_idx).to(tl.float32)                          # scalar tail
```

**Fallback** when `G_T_CONTIG=False` (legacy `[B,T,HV]` layout):

```python
g += bos * HV + i_h
p_gr = tl.make_block_ptr(g, (T,), (HV,), (i_tc_r,), (BC,), (0,))
b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
```

Only offset `g` once in the non-contiguous branch; all loads in that branch use the offset base.

### 3. Varlen checklist

- `T_seq` = host-passed total packed length (used in `i_h * T_seq` term).
- `T` after varlen setup = `eos - bos` (sequence length for `(T,)` block_ptr shape).
- Token offsets (`i_t * BT`, `i_tc_r`) are **relative to sequence start** (same as k/q/do pointers after `bos` offset).

## Measured wins (chunk_o.py, B=2, T=2048, H=4, HV=8, K=V=64)

| Kernel / entry | Before (stride HV) | After (G_T_CONTIG) |
|----------------|-------------------|---------------------|
| `chunk_bwd_kernel_dv_local_npu` (kernel only) | ~6.5 ms | ~0.18 ms |
| `chunk_bwd_dqkwg_npu` (kernel only) | ~10.8 ms | ~0.91 ms |
| `chunk_bwd_dqkwg_npu` (e2e incl. dg + transpose) | — | ~1.5 ms |

MTE2 (aiv) on dv_local dropped from ~31% to ~12% after fix.

## Correctness gates

- `tests/ops/test_gdn_kernels.py::test_chunk_bwd_dv_local`
- `tests/ops/test_gdn_kernels.py::test_chunk_bwd_dqkwg`
- Compare against Torch ref on T=2048 after optimization (not only small T).

## Anti-patterns (verified on Ascend Triton 3.2)

| Attempt | Result |
|---------|--------|
| Host transpose + kernel `g += bos*HV + i_h` + stride `(HV,)` | Wrong pointer → UB alignment crash or wrong values |
| `b_g[tl.arange(0, BC)]` after one BT load | Unsupported tensor index |
| `tl.reshape(b_dof, [2, BC, BV])` then sub-tile | `reshape() cannot change total number of elements` |
| `tl.join(b_dv0, b_dv1)` + reshape for single dv store | Compiles but **wrong layout** (~15 max diff vs split store) |
| `exp2(g_col) / exp2(g_row)` instead of `exp2(g_col - g_row)` | Risky on Ascend; verify numerics before using |
| Runtime `if r == 0` on task_id-derived index | Correctness bugs on Ascend; use fused constexpr paths instead |

## When to apply elsewhere

Any `triton_ascend` kernel that:

1. Reads `g` at multiple `(t, h)` positions inside nested `BC` / `BT` loops, and
2. Currently uses `make_block_ptr(g + bos*HV + i_h, (T,), (HV,), ...)`.

Also see `chunk_delta_h.py` bwd `dhu` notes in [reference.md](reference.md) (same stride-HV gather issue; additional gate precompute patterns there).

## Optimization round template

1. Baseline kernel-only timing with `G_T_CONTIG=False`.
2. Add host transpose + `G_T_CONTIG` kernel path (keep fallback for HV==1 / tests).
3. Run pytest gates above.
4. Re-profile PipeUtilization; confirm Duration and MTE2 drop.
5. Report **wall-clock** kernel time (one grid launch), not summed per-block Duration if profiler aggregates differently.
