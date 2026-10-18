# Ascend kernel cases

Use these examples to understand existing design choices. Read the current implementation before changing a path; tile settings depend on the workload and compiler. Paths are relative to the repository root.

## Causal convolution: core grid and DMA paths

Implementation: `fla/modules/causal_conv1d/backends/triton_ascend.py`.

The contiguous packed training path uses a 1D Vector-core task loop when channel tiling and state requirements permit. Other layouts, channel dimensions, and cache-state calls retain separate paths.

- Transpose the public convolution weight to make channels contiguous for block loads. Preserve the public weight and output layouts.
- Load one window including the convolution halo and use `extract_slice` for taps where supported. Keeping every tap live at once can exceed UB.
- Preserve the saved pre-activation used by backward for silu/swish; removing it can trigger a second forward computation.
- Separate bulk and tail DMA at compile time. A runtime choice keeps both paths live and can prevent larger tiles even when UB bandwidth is not saturated.

| Path                               | DMA policy                                                     |
| ---------------------------------- | -------------------------------------------------------------- |
| Full chunks in packed input        | Block-pointer loads, with the tail path compiled out.          |
| Final packed chunk                 | Masked loads/stores, including the halo bounds.                |
| Variable lengths or a single chunk | Runtime bounds where a uniform bulk/tail split is unavailable. |

Use `tests/modules/test_conv.py` for forward/backward, optional state, and layout validation. [TRAPS.md](TRAPS.md) covers optional pointers, packed bounds, and address types shared by these paths.

## `chunk_delta_h.py` — bwd `dhu`

Implementation: `fla/ops/common/backends/triton_ascend/chunk_delta_h.py`.

Gate loads along time use the [contiguous gate layout](g-contiguous-loading.md). The aligned fixed-length path can precompute gate terms, while variable-length or unaligned paths compute them inline. Model these paths separately: inline exponentials and temporary tiles can exceed the UB estimate for the precomputed path.

Choose K and V tiles together. Reducing one dimension merely to maximize the other can increase repeated work. Rebuild task-local pointers from base addresses and return the corrected value gradient separately to keep the caller's `dv` unchanged. Backward also reuses dot operands across K slabs; apply the [operand lifetime rule](TRAPS.md#reused-tldot-operands).

## `chunk_o.py` — fusion and gate loading

Implementation: `fla/ops/common/backends/triton_ascend/chunk_o.py`.

Forward combines inter- and intra-chunk contributions when their live tiles fit UB, reducing intermediate output traffic. Backward gate loads use contiguous time storage while gradient outputs retain their public layout. Profile the combined launch and transpose cost, not just one kernel. See [gate loading](g-contiguous-loading.md) for pointer formulas and focused tests.

## `chunk_bwd.py` — packed variable lengths

Implementation: `fla/ops/kda/backends/triton_ascend/chunk_bwd.py`.

Packed offsets multiplied by head and value dimensions can overflow int32 far before the token count reaches its limit. Load sequence boundaries as int64 and promote fixed-batch indices before multiplication. Only local lengths and block-pointer metadata should narrow to int32 when their bounds permit it.

## `tl.dot` lhs clobber — repo-wide case catalog

The common rule is in [TRAPS.md](TRAPS.md#reused-tldot-operands). This index identifies where to inspect operand reuse without prescribing variable names or a fixed number of copies.

| Implementation                                               | Reuse to inspect                                                                               |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `fla/ops/common/backends/triton_ascend/chunk_o.py`           | Query/key tiles shared by multiple dots; attention tiles reused across value blocks.           |
| `fla/ops/common/backends/triton_ascend/chunk_delta_h.py`     | Output-gradient and corrected residual tiles reused across K slabs.                            |
| `fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py` | Inverse tiles reused between stages; gradient tiles used first on the left, then on the right. |
| `fla/ops/kda/backends/triton_ascend/wy_fast.py`              | Inverse tiles reused across the value and key loops.                                           |
| `fla/ops/kda/backends/triton_ascend/chunk_intra.py`          | Triangular block merges that reuse operands for several products and final stores.             |
| `fla/ops/utils/backends/triton_ascend/solve_tril.py`         | Block inverse operands used in multiple products or stored after a dot.                        |
| `fla/ops/kda/backends/triton_ascend/chunk_bwd.py`            | Output gradients used as both left and right operands.                                         |

Audit the changed kernel's first left-operand use and every subsequent read. Copies must precede that first use; reloads must come from pristine storage. Validate with the operator's kernel tests, including `tests/ops/test_gdn_kernels.py` and `tests/ops/test_solve_tril.py` where relevant.
