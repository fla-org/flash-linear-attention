# Context-parallel tests

Run CP coverage from the repository root:

```bash
FLA_CI_ENV=0 pytest tests/cp -q
FLA_CI_ENV=0 pytest tests/cp/modules -q
FLA_CI_ENV=0 pytest tests/cp/ops/test_kda.py -k CP2 -q
```

The suite covers CP context metadata, module communication, operator outputs and gradients, intra-card caching, and graph replay. Module tests live in `modules/`; operator and state-kernel tests live in `ops/`. Ordinary operator tests retain backend-verifier cases that only check whether a CP argument is supported.

Distributed cases spawn their own local ranks. Run pytest once per host, without `torchrun` or `pytest-xdist`: every case uses the first required devices. Each run has its own rendezvous file, and failures in a worker fail the parent test. Cases requiring 2, 4, or 8 devices skip when fewer devices are available. The KDA suite also supports Ascend HCCL; direct CUDA graph and CUDA convolution coverage retain their platform restrictions.

The parameter IDs identify rank count and sequence layout. They preserve sequence cuts, aligned boundaries, short sequences, state orientation, grouped heads, gate modes, recomputation, and zigzag layouts. References and tolerances remain specific to each operator. KDA and GDN-2 retain their existing warning-only gradient diagnostics; those warnings are not strict numerical passes, even with `FLA_CI_ENV=0`.

Single-rank graph and intra-card cache tests retain the NaN allocation guard used by their original operator tests. Distributed graph workers keep captured communicators alive until process exit.

See [the CP implementation](../../fla/ops/cp/README.md) for the algorithm and [debugging notes](debug.md) for numerical investigations.
