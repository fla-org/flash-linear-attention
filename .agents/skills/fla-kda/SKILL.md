---
name: fla-kda
description: Modify or review KDA gates, chunk and recurrent kernels, backend routes, and their correctness coverage.
---

# KDA

Use this skill for `fla/ops/kda/` and its tests. Public entry points are `chunk_kda` and `fused_recurrent_kda`; inspect the affected entry point's signature before changing its implementation.

## Code map

- `gate.py` defines gate activations and cumulative decay.
- `chunk_fwd.py` and `chunk_bwd.py` coordinate training; `chunk_intra.py`, `chunk_intra_token_parallel.py`, and `wy_fast.py` implement the main stages.
- `backends/` contains alternative implementations and verifiers. Follow [backend dispatch](../fla-dispatch-backends/SKILL.md) when changing routing.
- `tests/ops/test_kda.py` contains the reference comparisons and supported gate-mode cases.

## Gate contracts

With `use_gate_in_kernel=False`, the caller supplies log-space decay. With it enabled, the caller supplies raw gates and optional bias:

- Without a lower bound, activation is `-exp(A_log) * softplus(g + dt_bias)` and `A_log` is required.
- With a lower bound, activation is `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`. Omitting `A_log` uses a multiplier of one.

`safe_gate` selects the intra-chunk implementation independently of gate activation. Safe in-kernel activation requires `-5 <= lower_bound < 0`. Pre-gated safe mode requires bounded log-space gates supplied by the caller; tests cover `[-5, 0]`, but the wrapper does not validate tensor values.

## Numerical invariants

The safe intra path uses 16-token diagonal blocks and midpoint offsets before exponentiation. At a lower bound of `-5`, a block can accumulate `-80` in natural-log units; local offsets limit each exponent's span to about half that range. Preserve these offsets rather than exponentiating the full cumulative gate. Inter-block decay uses paired differences whose exponents remain non-positive for monotonic decay.

The non-safe path uses token-parallel diagonal computation. Both paths share inter-block and triangular-solve work, so changes there require coverage of both modes. Keep precision and tolerance changes within the design process in [AGENTS.md](../../../AGENTS.md#scope-and-direction).

## Validation

Use [correctness coverage](../fla-correctness-coverage/SKILL.md) for the test strategy. Select the KDA cases affected by the change:

- Dense and variable-length sequences; forward, final state, and supported gradients.
- Pre-gated and in-kernel gates, safe and non-safe paths, and omitted optional arguments.
- Beta activation and query/key normalization modes when touched.
- Grouped value heads and differing key/value dimensions when indexing or shapes change.
- Initial states, intermediate states, and context parallelism when their paths are touched.
- Accepted and rejected backend routes when changing verifiers.

For gate or decay changes, include the lower-bound limits, saturated gate inputs, extreme gate scales, long cumulative decay, and chunk or ragged-sequence boundaries. Test through the public entry point so the selected route and effective arguments are exercised.
