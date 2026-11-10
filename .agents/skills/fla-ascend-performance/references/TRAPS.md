# Ascend compiler and memory traps

Read the relevant section before changing DMA, launch indexing, or reused matrix operands. [Kernel cases](cases.md) show these constraints in existing implementations.

## DMA paths and optional pointers

A runtime branch between an in-bounds bulk load and a masked tail load can keep both paths live in UB. Larger tiles may then overflow even when measured UB bandwidth is low. Where the workload permits, split bulk and tail launches with a constexpr mode so each compilation removes the unused path. Include convolution halos when deciding whether a bulk load fits inside the packed allocation.

An optional pointer needs a compile-time guard before any runtime condition. `if USE_INITIAL_STATE or runtime_condition:` can still compile pointer arithmetic in the other branch when the pointer is `None`; nest the constexpr condition separately.

## Address arithmetic

Cast **scalar** indices to `tl.int64` before multiplying by strides or dimensions. Casting the product cannot repair overflow. On Ascend, specialized arguments and folded program IDs may be constexpr values without a `.to()` method; `tl.cast` works for both runtime and constexpr integers.

```python
token_offset = tl.cast(i_t, tl.int64) * BT
batch_offset = tl.cast(i_b, tl.int64) * T
bos = tl.load(cu_seqlens + i_n).to(tl.int64)
rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
```

Load packed sequence offsets as int64 even when the input `cu_seqlens` uses int32. A small token index can overflow after multiplication by head count and feature size.

Do **not** cast every subscript to int64. A few scalar promotions per program are cheap; widening every `tl.arange` lane to int64 adds vector 64-bit work on hot load/store paths and can hurt performance without improving correctness.

| Role | Policy |
| ---- | ------ |
| Program IDs, batch/sequence indices, loaded `bos`/`eos` | Promote to int64 **before** any multiply by `T`, strides, or head×feature sizes. |
| Global memory block offsets | Form as `int64_base + tl.arange(0, BLOCK)`; promote the base (or the multiply), not the arange. |
| UB tile coordinates (K/V slabs, small fixed tiles) | Keep int32 when the range is bounded by the tile and no large stride product is involved. |
| Local sequence length inside a tile | Narrow to int32 when the value is provably in range; see [packed variable lengths](cases.md#chunk_bwdpy--packed-variable-lengths). |

`tl.make_block_ptr` and `tl.advance` are deprecated on triton-ascend. Follow the [Triton kernel rule](../../../../CONTRIBUTING.md#triton-kernels): address tiles with `offset + tl.arange` and plain `tl.load` / `tl.store`. Global pointer offsets must not rely on int32 overflow in the **base** arithmetic; block lanes may stay narrow until they combine with that base.

## Grid and task loops

The existing Ascend grid helpers enforce a grid-product limit of 65535. Use their host splitting or flatten independent tiles into a 1D loop over available cores. `get_multiprocessor_count()` selects Vector cores on NPU; `use_aicore=True` selects Cube cores.

After slicing variable-length chunk metadata, apply the corresponding global offset only once. In a task loop, derive local pointers from the original base on each iteration; accumulating pointer updates across tasks can miscompile on Ascend.

## Reused `tl.dot` operands

Triton-Ascend can reuse the left operand's UB storage during `tl.dot`. A later read of that tile can therefore be wrong even without a compiler error. Audit all later uses, including another left operand, a right operand, arithmetic, and stores.

Reload the pristine tile from global memory when reuse is separated by stages. For tight reuse, create the needed `tile + 0.0` copies before the first dot that consumes the original as a left operand. Copying afterwards preserves the corrupted value. Account for the extra live copies in the UB budget and validate against the numerical reference.

[The case index](cases.md#tldot-lhs-clobber--repo-wide-case-catalog) identifies the affected kernel patterns. A fresh load on each loop iteration or a tile used as a left operand only once needs no extra copy for this issue.

## Numerical and tail behavior

Preserve the validated accumulation precision and exponential base. Mask invalid values before exponentials and initialize every region consumed later. Factoring a gate difference into a ratio or reciprocal changes rounding and range behavior; check the existing reference and tolerances before using it.

Exercise gated and ungated paths when both are supported. A suite that always supplies a gate cannot catch `None` pointer handling or incorrect ungated residual updates.
