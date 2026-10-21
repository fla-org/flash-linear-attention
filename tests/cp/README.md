# CP(Context Parallel)

1. CP2TP refers to Context Parallelism with Tensor Parallelism on attention heads. When CP is mentioned in FLA kernels, it specifically denotes TP (head parallelism), which requires two all-to-all collective operations—one before and one after the attention computation.

2. Ring CP introduces sequential dependencies between ranks. For example, in a CP2 configuration, Rank1 must wait for Rank0 to complete before it can compute; during backpropagation, Rank0 must conversely wait for Rank1.

3. True CP resembles the CP implementation in FLA(KDA and GDN first), enabling genuinely parallel computation across all ranks with minimal communication overhead. However, this approach is more complex and requires sophisticated mathematical optimizations.

## Running tests

Run the suite with `FLA_CI_ENV=0 pytest tests/cp -q`. Distributed cases require the device count shown in their parameter IDs. For example, `pytest tests/cp/test_kda.py -k cp2-sequence-cut` selects KDA sequence-cut cases.

Files identify the operator or shared component; test functions identify the operation, and parameter IDs identify the partition, layout, or compatibility case.

| Files                                                       | Coverage                                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `test_context.py`                                           | Sequence partitions, backward snapshots, precision keywords, and device requirements. |
| `test_state.py`                                             | Backward affine-state summaries and per-head vector gates.                            |
| `test_graph.py`                                             | KDA graph capture and replay against eager execution.                                 |
| `test_conv.py`, `test_token_shift.py`                       | Convolution and token-shift outputs and gradients.                                    |
| `test_kda.py`, `test_gdn.py`, `test_gdn2.py`, `test_gdp.py` | Chunked operators against their existing references.                                  |
| `test_dplr.py`, `test_rwkv7.py`                             | DPLR and its RWKV7 wrapper against the recurrent reference.                           |

The `legacy-precision-true` cases exercise the accepted legacy keyword; all affine-chain tests use TF32x3. Numerical investigation guidance is in [debug.md](debug.md).
