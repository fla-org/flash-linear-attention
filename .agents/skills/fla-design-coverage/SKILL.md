---
name: fla-design-coverage
description: >
  Contract-first design and coverage discipline for FLA kernel and numerical
  changes. Use before implementation to define supported cells, numerical
  budgets, dispatch semantics, compatibility, tests, and benchmarks.
---

# FLA Design & Coverage Skill

Use this skill at the start of designing a kernel, backend path, or numerical change. Complete these steps in order before implementing:

1. Enumerate every reachable contract cell and classify its route.
2. Record forward and backward numerical budgets for each leaf stage.
3. Intersect the three coverage layers with the relevant axes from `fla-correctness-coverage`.
4. Audit routing, effective arguments, defaults, and every previously supported cell.
5. Freeze the production oracle, workload, baseline, and benchmark order.

This skill defines the design contract. Use `fla-correctness-coverage` for test axes and `fla-dispatch-backends` for dispatch mechanics.

## Contract invariants

The unit of design is a contract cell: one reachable combination of the dimensions below. Do not generate combinations that no public API or supported FLA layer, model, or CP route can reach, counting fallback outcomes as reachable.

- operator
- backend, meaning a selected implementation route — a registered `BaseBackend` implementation or the original decorated implementation; not a hardware product or the `fla.utils.Backend` enum
- leaf pipeline: the exact execution route after wrappers and `@dispatch` selection, identified by its ordered leaf stages; a leaf stage is one kernel or numerical operation whose load, operand, accumulation, store, range, or reduction behavior needs its own numerical budget
- chunk geometry, including `BT`, `BC`, and, for operators with anchors, the anchor placement
- bound regime: the decay/gate ranges each affected stage is licensed for, recorded per stage, e.g. `gk >= lower_bound` or the `safe_gate` range
- routing capability regime: the platform, resource, or software predicates that can change route selection, expressed as capability conditions rather than product names
- dtype staging: the per-stage sequence of load dtype, explicit conversions, operand dtype, accumulation dtype, and store dtype; public input/output dtypes alone do not identify staging
- input numerical domain: the value-range, sign, normalization, scaling, and correlation assumptions on numerical inputs, excluding shape, dtype, and layout

Before implementation, write a contract table with one row per reachable cell. Derive the rows from four inventories — public entry points and their defaults, registered backend verifiers, layer/model callsites, and existing parameterized tests — and keep a `Source` column so every inventory item maps to at least one row. Each row ends in exactly one disposition:

- **Optimized path**: routing selects the proposed implementation for the cell, and the row names its oracle, output and per-gradient tolerances, and passing test IDs.
- **Existing fallback**: the cell stays on an implementation that predates the proposal, passes the applicable public-contract forward and backward tests, and preserves the public API semantics.
- **Explicit unsupported error**: the public entry rejects the cell before kernel execution with a documented exception type and a deterministic reason, asserted by a test.

There is no implicit fourth state. Uninitialized output, a silent or semantics-changing fallback, a tolerance change made to pass a gate, and a warning that hides a hard regression are correctness failures.

A fallback counts as **Existing fallback** only when it preserves the return structure, output dtypes, state and autograd behavior, and the row's oracle within its tolerances, verified by a route-parity test. Under context parallelism, the routing decision must be identical on every rank: if any verifier input can differ by rank, combine the accept bits with a collective and select the optimized backend only when every rank accepts. If a successful forward call commits autograd to the same backend, record that in the row.

Do not change a committed correctness-gate oracle, input distribution, assertion, or tolerance in the same change that relies on it to pass. A tolerance change goes in a separate commit with before/after error distributions for the unchanged baseline and candidate, and `python -m benchmarks.ops.verify --op <op>` stays green before the candidate lands. Any committed tolerance change also follows the RFC requirement in `AGENTS.md`.

## Local permissions

Apply a safety flag only to cells that satisfy its documented bound regime and input-domain assumptions. Evaluate the flag for each combination of leaf pipeline, chunk geometry, anchor placement, dtype staging, bound regime, and input numerical domain; never promote it to an operator-wide guarantee.

Record each stage's assumptions independently: an exponent budget for one stage does not prove that a later inversion is safe. Add each extra assumption (input-norm bounds, correlation bounds, and the like) as a named predicate in the row, enforced by public-entry validation or by every affected backend verifier, with one acceptance test at the boundary and one rejection or fallback test just outside it.

Resolve user-visible defaults and normalization in one shared helper. The public entry, every backend verifier, and every backend executor call that helper rather than duplicate its rules; a verifier may call it independently for routing, but must not reimplement the formula. Add parity tests asserting that an omitted argument and its documented explicit default select the same route and produce matching outputs and gradients. `input_guard` covers contiguity and device context only; it does not normalize semantic arguments or dtype staging.

## Numerical budgets by stage

A leaf-stage budget records the contract below for one leaf stage. A cell's numerical budget is the ordered set of its leaf-stage budgets plus the end-to-end acceptance rule. Record budgets in both directions: outputs and gradients each pass their committed `assert_close` tolerances, and passing the forward comparison does not cover backward stages such as inverse recomputation, gradient combination, and state backpropagation.

| Field      | Required decision                                                   |
| ---------- | ------------------------------------------------------------------- |
| Source     | Input numerical domain and upstream range assumptions               |
| Load       | Dtype and any conversion applied at ingress                         |
| Operand    | Dtype presented to each numerical operation                         |
| Accumulate | Accumulation dtype and reduction behavior                           |
| Store      | Output dtype and conversion at egress                               |
| Range      | Exponent headroom and overflow/underflow limits                     |
| Error      | Rounding allowance and how it propagates downstream                 |
| Acceptance | Oracle, output and per-gradient tolerances, and test IDs            |

End-to-end error is the propagation of these stage budgets: algebraically equivalent expressions are not numerically equivalent when staging, reduction order, or range changes. Changing the compute or accumulation precision of a validated path, relaxing a committed tolerance, or replacing its validated numerical algorithm requires the RFC process under "Scope and direction" in `AGENTS.md`; ordinary reorderings within the same precision are exempt there.

Every numerical-safety threshold used to accept or reject a cell has a documented algebraic derivation, is implemented once in the operator's backend-neutral module, and is called by every verifier that depends on it; the rejection reason names the failed predicate and the effective values. Performance-routing thresholds are outside this rule but require benchmark evidence. Example pattern — DPLR's `gate_bound_is_safe(lower_bound, chunk_size)`:

```text
abs(lower_bound) * (chunk_size // 2 + 1) * log2(e) <= 124
```

`124` is the fp32 exponent limit (~128 in base-2 units) minus activation-multiply headroom. Tests cover the largest accepted value and the adjacent rejected value.

## Coverage layers

Keep these layers separate:

| Layer | Question | Required evidence and home |
| ----- | -------- | -------------------------- |
| Production-representative hard gate | Do the cells FLA layers and models actually dispatch stay correct and fast? | Pin effective arguments, shapes, dtypes, bound regimes, input distributions, and routes from checked-in layer/model callsites, reproduced in `tests/models/test_modeling_*.py` or an exact op-level equivalent; global `SHAPE_CONFIGS` alone do not establish production use. Run `python -m benchmarks.ops.verify --op <op>` (see `fla-optimization-loop`). |
| Public-contract boundary hard gate | Does the full documented and previously supported domain remain valid? | Parameterized matrices in `tests/ops/test_<op>.py`; backend verifier accept/reject cases; parity tests asserting an omitted argument matches its documented explicit default. |
| Beyond-contract adversarial coverage | Does out-of-contract input fail safely? | Each case requires one of two outcomes: rejection before kernel execution with the documented error, or completion with finite outputs and gradients via explicit `torch.isfinite` assertions. NaN memory poisoning covers only `tests/ops/` and `tests/modules/`; layer and CP tests add their own finite checks. |

No layer substitutes for another: production evidence does not prove compatibility, boundary evidence does not represent production numerics, and adversarial robustness does not expand the public contract.

Build one coverage plan per reachable cell: label each test case with its layer, then vary the applicable axes from `fla-correctness-coverage` within that cell. Do not maintain a separate layer matrix and axis matrix. When two routes both claim the same cell, add a route-parity test that selects each route explicitly (e.g. `FLA_TILELANG=0/1`) and compares outputs and gradients under the cell's tolerances.

## Routing and compatibility

Implementations may be selected by backend, platform capability, architecture capability, or optional-software availability; all routes for the same contract cell preserve the same public numerical semantics — same oracle, tolerances, return structure, and documented behavior. Centralize product, architecture, and platform detection in `fla.utils`; kernel and backend code consumes the narrowest existing capability or availability helper, adding a new product-family flag only when no capability-level predicate expresses the requirement. Keep the dispatch terminology distinct: `BaseBackend.is_available()` checks backend availability, `is_enabled()` checks policy enablement, and a call verifier checks one call; helpers such as `find_spec_cached` and `has_usable_nvcc` supply facts those checks consume, and `get_device_capability` is CUDA-specific and appears only inside platform-guarded paths. Capability-based performance routing is allowed with benchmark evidence and is recorded in the contract table.

Preserve every cell supported at the PR's merge-base — documented by the public API, covered by a committed test, or dispatched by a supported layer, model, or CP route. Record the merge-base SHA in the design note. Each such cell keeps its existing implementation or gains a fallback that satisfies the Existing-fallback definition; replacing support with a new rejection requires the breaking-change approval in `AGENTS.md`.

Autotune configs and caches must not define separate numerical contracts: every config selectable for a cell stays within its budget, even though config keys do not encode the input numerical domain. Record compute-precision environment state (`TRITON_F32_DEFAULT`, matmul precision settings) in the budget context, set the same explicit values for baseline, candidate, and every test of the operator, and restore global state after the test.

If a proposal changes a public argument default or the route a default call selects, add explicit `Default before` and `Default after` entries to every affected contract-table row and obtain the breaking-change approval in `AGENTS.md`. For any interface, docstring, or behavior change, audit every caller, backend, test, and document in one repository-wide pass, and record when a category has no affected sites.

## Production-first benchmarking

Before comparing implementations, record for every production-dispatched cell: its oracle, normalized effective arguments, input distribution and seed, baseline correctness status, baseline latency or throughput, hardware and software environment, and baseline commit SHA.

Benchmark the production cells first, then each public-contract boundary that changes routing, chunk geometry, dtype staging, bound regime, or expected performance of the optimized path; boundaries unaffected by the proposal need correctness coverage but no new performance measurement. A speedup on a path that no checked-in layer or model can dispatch under the recorded environment is supporting evidence only, not evidence that the design improves FLA.
