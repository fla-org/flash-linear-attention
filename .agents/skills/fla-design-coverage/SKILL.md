---
name: fla-design-coverage
description: Define supported inputs, numerical budgets, routing, and validation before designing an FLA kernel or numerical change.
---

# FLA Design and Coverage

Use this skill to establish the contract before implementing a kernel, backend route, or numerical change. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) for contribution rules, especially [public compatibility](../../../CONTRIBUTING.md#public-api-compatibility) and [design/RFC requirements](../../../CONTRIBUTING.md#submit-pull-requests).

## Define the supported domain

Trace public entries and defaults, backend verifiers, layer/model/CP callers, and existing tests. Record the merge-base SHA and the source of each supported case. Group cases only when they share the same numerical assumptions and execution route; omit combinations no supported caller can reach.

Distinguish routes when kernel stages, chunk geometry or anchors, capability checks, dtype conversions, or input value ranges change. A backend here means an implementation selected by dispatch, including the default implementation, rather than a hardware product. For each group, identify its effective arguments, oracle, output/state and per-gradient tolerances, and test coverage. Assign one outcome:

- The proposed implementation, with passing reference comparisons.
- An existing fallback, with route-parity tests covering outputs, states, dtypes, and supported gradients.
- A documented unsupported-input error raised before kernel execution and asserted by a test.

Preserve previously supported cases under the public compatibility policy. A warning, partially initialized output, or fallback that changes semantics is not a valid outcome. When changing defaults, record the old and new effective values and selected routes.

## Establish numerical safety

For each affected stage in forward and backward, record input-range assumptions, load/operand/accumulation/store dtypes, reduction behavior, overflow/underflow headroom, and error propagation to the end-to-end tolerance. Forward agreement alone does not establish backward safety, including inverse recomputation and state gradients.

Safety flags and bounds apply to the stages and domains that justify them. Reassess them when chunk geometry, anchors, dtype staging, or input scaling changes; a safe exponent does not prove that a later inversion is safe. Document the algebra behind numerical acceptance thresholds, implement each shared predicate once, and test its acceptance boundary and the adjacent rejection or fallback. Enforce additional assumptions, such as input-norm bounds, at the public entry or every affected verifier; report the failed predicate and effective values.

Keep the committed oracle, input distributions, assertions, and tolerances fixed while evaluating the candidate. A correction to that contract belongs in a separate change with before/after error distributions for the unchanged baseline and candidate; precision, tolerance, and algorithm changes follow the linked RFC policy. Keep the corrected gate green before the candidate lands. Every selectable autotune configuration must satisfy the same numerical contract.

## Check routing

- Share semantic argument normalization between the public entry, verifier, and executor. Test that omitted arguments match their documented explicit defaults in route, outputs, and gradients. `input_guard` handles contiguity and device context, not semantic defaults.
- Use capability and availability helpers from `fla.utils`. Distinguish backend availability, policy enablement, and acceptance of an individual call; use [fla-dispatch-backends](../fla-dispatch-backends/SKILL.md) for implementation mechanics.
- For context-parallel routes, every rank must make the same selection. If verifier inputs can differ by rank, combine acceptance with a collective and select the optimized route only if all ranks accept. Record whether forward commits backward to the same backend.
- Test parity between routes claiming the same domain. Record precision settings such as `TRITON_F32_DEFAULT` and matmul flags, keep them identical across baseline, candidate, and tests, and restore global settings after tests.

## Choose evidence

Apply the relevant axes from [fla-correctness-coverage](../fla-correctness-coverage/SKILL.md) to each supported group across these distinct layers:

- **Production:** reproduce effective arguments and numerical input distributions from checked-in layer/model callers. A generic benchmark shape list alone does not establish production coverage.
- **Public boundaries:** cover previously supported inputs, changed defaults, and verifier acceptance/rejection boundaries.
- **Adversarial:** outside the contract, require documented rejection before kernel execution or explicit finite-output and finite-gradient checks. This does not expand the supported domain.

Before performance work, record baseline correctness, latency or throughput, commit, environment, workload, input distribution, and seed. Benchmark production cases first, then boundaries whose routing or expected performance changes. Unaffected boundaries still need correctness coverage. Use [fla-optimization-loop](../fla-optimization-loop/SKILL.md) to freeze the gate and compare candidates.
