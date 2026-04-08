# AI Review Guidelines for flash-linear-attention

## Triton Kernel Integer Overflow Prevention

`tl.program_id()` and `tl.load()` from `cu_seqlens` return **int32** values. When these are multiplied by strides (e.g., `T`, `H*K`, `D`), the intermediate product can exceed `INT32_MAX` (2^31) for realistic tensor sizes (e.g., `B=4096, T=576, H=8, K=128`), causing silent wrong results or illegal CUDA memory accesses.

**Rule**: All index arithmetic derived from `tl.program_id()` or `cu_seqlens` loads should be cast to `int64` *before* any multiplication with strides or dimensions. Results should be cast back to `int32` before passing to `tl.make_block_ptr`, which requires 32-bit shape and offset arguments.

### When reviewing Triton kernels, flag:
- Any `i_b * T`, `i_n * T`, `i_b * D`, `i_n * D`, or `i_b * stride_*` without a prior `tl.cast(..., tl.int64)` **only when the product can plausibly exceed INT32_MAX** (consider the actual dimensions involved — not every int32 multiply needs promotion)
- Any `tl.load(cu_seqlens + ...).to(tl.int32)` (should be `.to(tl.int64)`)
- Any compound index expression like `(i_b * S + i_s) * D` where `i_b` or `i_s` comes from `tl.program_id()` without int64 promotion

See [#783](https://github.com/fla-org/flash-linear-attention/pull/783) and [#803](https://github.com/fla-org/flash-linear-attention/pull/803) for prior instances of this bug class.
