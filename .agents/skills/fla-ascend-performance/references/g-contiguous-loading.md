# Contiguous gate loading

Use this pattern when an Ascend kernel repeatedly loads gate values along time from `[B, T, HV]` storage. The time stride is `HV`, so these loads gather separated values. Transposing the input gate can reduce memory-transfer overhead; measure the transpose and kernel together.

Reference implementation: `fla/ops/common/backends/triton_ascend/chunk_o.py`.

## Host layout

```python
g_t_contig = g is not None and HV != 1
g_arg = g.transpose(1, 2).contiguous() if g_t_contig else g
```

Pass the selected input and layout flag to the kernel. `HV == 1` is already contiguous along time. Keep gradient outputs in their original `[B, T, HV]` layout.

## Kernel addressing

Save the total packed length before computing a local sequence length. Promote batch, head, and packed sequence offsets before multiplying them; see [address arithmetic](TRAPS.md#address-arithmetic).

| Input mode              | Contiguous gate base                    |
| ----------------------- | --------------------------------------- |
| Fixed batch             | `g + batch * HV * T_seq + head * T_seq` |
| Packed variable lengths | `g + bos + head * T_seq`                |

Here `batch`, `head`, and `bos` are int64 values, `T_seq` is the host-passed storage length, and the block-pointer bound uses the local sequence length. Block offsets are relative to that sequence's start.

For a contiguous gate base, use stride 1 along time. The original-layout branch instead uses `g + bos * HV + head` with time stride `HV`. Apply the base offset once and keep every load in a branch consistent with that layout. Reusing the original stride or offset after transposing produces wrong values or alignment faults.

## Validation

Compare both layout paths against the same reference, including head sharing, variable lengths, non-aligned tails, and long sequences. The focused tests are:

- `tests/ops/test_gdn_kernels.py::test_chunk_bwd_dv_local`
- `tests/ops/test_gdn_kernels.py::test_chunk_bwd_dqkwg`

Inspect wall-clock duration and MTE activity before and after, including transpose and gradient-finalization work. Sum-of-block profiler durations may differ from one launch's elapsed time.

Preserve the exponential formula while changing layout. Replacing `exp2(a - b)` with a ratio can change overflow and rounding behavior. On Ascend, use the supported slicing/layout operations from the current implementation; a reshape or join that compiles does not prove the intended element ordering.
