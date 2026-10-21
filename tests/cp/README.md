# Context-parallel tests

Run CP coverage from the repository root:

```bash
FLA_CI_ENV=0 pytest tests/cp -q
FLA_CI_ENV=0 pytest tests/cp/test_kda.py -k cp2-sequence-cut -q
```

Each component keeps its inputs, reference, CP call, and assertions in one test file. DPLR and its RWKV7 wrapper share a parameter matrix. The `run_distributed` fixture owns process startup, device-count checks, and rendezvous cleanup. Workers receive explicit keyword arguments; token sharding and gathering live in `utils.py`.

| File                                  | Coverage                                                                   |
| ------------------------------------- | -------------------------------------------------------------------------- |
| `test_context.py`                     | Sequence partitions, precision keywords, device requirements, and backward snapshots.                          |
| `test_state.py`                       | Per-head vector-gate addressing in backward state summaries.               |
| `test_intracard.py`                   | Intra-card backend selection, index caching, and grouped heads.            |
| `test_graph.py`                       | KDA graph replay with one or multiple ranks and changing sequence cuts.    |
| `test_conv.py`, `test_token_shift.py` | Boundary exchange, outputs, and gradients for convolution and token shift. |
| `test_kda.py`, `test_gdn2.py`         | Gate modes, recomputation, state layouts, and recurrent references.        |
| `test_gdn.py`, `test_gdp.py`          | Grouped heads and multiple Householder updates against their references.   |
| `test_dplr.py`                        | DPLR, its RWKV7 wrapper, and explicit TileLang routing.                    |

Multi-rank tests compare sharded outputs and gradients against full-sequence references. Context metadata, backward state-kernel, intra-card cache, and single-rank graph tests cover separate CP infrastructure; they do not establish cross-rank numerical correctness. Ordinary operator tests retain backend-verifier cases that only check whether a CP argument is supported.

Distributed cases spawn their own local ranks. Run pytest once per host, without `torchrun` or `pytest-xdist`: every case uses the first required devices. Each run has its own rendezvous file, and failures in a worker fail the parent test. Cases requiring 2, 4, or 8 devices skip when fewer devices are available. Affine-chain CP requires NVIDIA Ampere or newer; direct CUDA graph and CUDA convolution coverage retain their platform restrictions.

Test functions name the operation, and parameter IDs identify the scenario: `test_chunk[cp2-sequence-cut]`, for example. Cases cover sequence cuts, aligned boundaries, short sequences, state orientation, grouped heads, gate modes, recomputation, and zigzag layouts. The `legacy-precision-true` suffix exercises the accepted legacy keyword; all affine-chain cases use TF32x3. References and tolerances remain specific to each operator. KDA and GDN-2 retain their existing warning-only gradient diagnostics; those warnings are not strict numerical passes, even with `FLA_CI_ENV=0`.

Single-rank graph and intra-card cache tests retain the NaN allocation guard used by their original operator tests. Distributed graph workers keep captured communicators alive until process exit.

See [the CP implementation](../../fla/ops/cp/README.md) for the algorithm and [debugging notes](debug.md) for numerical investigations.
