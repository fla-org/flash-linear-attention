# Context parallel tests

Run the suite with `FLA_CI_ENV=0 pytest tests/cp -q`. Distributed cases use 2, 4, or 8 devices as indicated by their parameter IDs; cases that need more devices than are available are skipped. KDA also supports Ascend NPU with HCCL. Each test starts fresh processes and uses its own file rendezvous, so tests do not share fixed TCP ports.

| File                          | Purpose                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `test_context.py`             | Sequence partition validation, rank metadata, and backward snapshots.                                                     |
| `test_comm.py`                | Neighbor selection and boundary behavior for forward/backward communication.                                              |
| `test_state.py`               | State compression, hardware precision selection, intra-card affine summaries/merge, and per-head vector-gate addressing.  |
| `test_conv.py`                | Triton/CUDA/mixed convolution outputs and gradients, residuals, layouts, short sequences, and CUDA argument validation.  |
| `test_token_shift.py`         | Distributed token-shift outputs and input gradients, including one- and two-token tails.                                  |
| `test_gdn.py`                 | GDN outputs and all input gradients against non-CP execution; GVA and both state layouts.                                 |
| `test_gdp.py`                 | GDP outputs and all input gradients against recurrence, with two or three Householder updates and unequal K/V dimensions. |
| `test_gdn2.py`, `test_kda.py` | Recurrent-reference comparisons, fused and pre-activated KDA gates, recomputation, both state layouts, and KDA zigzag partitions.          |
| `test_dplr.py`                | DPLR recurrent-reference comparisons, safe gates, TileLang routing, and the RWKV7 wrapper.                                |
| `test_graph.py`               | KDA forward/backward graph replay against eager CP while changing sequence boundaries and zero-length padding.            |

Operator cases keep their input distributions, seeds, references, and tolerances explicit. KDA and GDN-2 retain their existing warning-only gate/update-gradient comparisons against recurrence; those checks are diagnostic rather than strict correctness gates. GPU skips and warnings must be reported separately from passing numerical assertions. KDA, GDN-2, and GDP also compare CP against non-CP execution with strict assertions on all input gradients; KDA covers both partition and state layouts, GDN-2 both state layouts, and GDP two or three Householder updates. Common state-kernel reference comparisons also live in `tests/ops/test_gdn_kernels.py`.

The intra-card state cases cover scalar/vector gates, GVA, nonzero initial state, both state orientations, unequal K/V dimensions, partial chunks, and one through four 64-row state tiles. Cross-rank merge is also checked against an FP64 affine recurrence in both directions and both state layouts, including masked K tiles. Host-only metadata, communication mocks, and precision-selection checks do not establish GPU numerical correctness.

## Debugging numerical failures

Compare CP and non-CP outputs and per-token gradients against the same reference. A rank cut can change chunk boundaries, so compare intermediate chunk states only after matching their global token positions. Forward summaries use the last local sequence; backward summaries use the first.

When investigating backward recomputation, check that the merged initial state is saved and expanded before the forward-state recomputation. Keep the merge grid dependent on the autotuned `BV`; hardcoding its value can leave part of the value dimension unwritten. Use a separate cache directory for diagnostics instead of deleting a Triton cache used by running processes.

For the state-chain equations and layout contract, see [the CP implementation notes](../../fla/ops/cp/README.md).
