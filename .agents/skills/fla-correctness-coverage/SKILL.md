---
name: fla-correctness-coverage
description: >
  Guidelines for kernel correctness testing and coverage in fla/ops/** and related
  modules, including common Triton grid/addressing pitfalls. Helps decide what
  tests to add or run before an MR.
---

# FLA Correctness & Coverage Skill

Use this skill when adding or modifying a kernel in `fla/ops/` (e.g., KDA, GDN,
GLA, DeltaNet, NSA, etc.) and you need to verify correctness or close a coverage
gap.

## Workflow

1. **List the current coverage matrix** for the op you are touching.
2. **Compare** against the axes below.
3. **Add tests** for missing combinations that are reachable by user code.
4. **Run** the relevant tests and make sure they pass.

## Public reference docs

When a task needs operator math or protocol details, read only the relevant
reference file:

- `references/cp.md` — context parallelism for linear attention, including KDA/GDN CP formulation.
- `references/delta-rule.md` — Delta Rule operator background.
- `references/generalized-delta-rule.md` — Generalized Delta Rule operator background.
- `references/simple-gla.md` — Simple GLA operator background.

Do not load every reference by default; use these only when the touched code or
test depends on that operator's math or distributed protocol.

## Coverage axes

For each kernel, check coverage across these dimensions:

| Axis | Values to cover |
|------|-----------------|
| **Sequence layout** | dense, variable-length (`varlen`) |
| **Direction** | forward, backward |
| **Gate mode** | safe gate, non-safe gate (if applicable) |
| **Beta mode** | raw beta, post-sigmoid beta (if applicable) |
| **QK normalization** | with L2 norm, without L2 norm |
| **State** | initial state, final state (if the op supports state passing) |
| **GVA** | grouped value attention (GVA) enabled vs disabled |
| **Head dimensions** | `D != Dv` (different qk and v head dims) |
| **Backend verifier** | reference implementation, `torch.autograd.gradcheck`, and backend-specific sanity checks |

## Kernel implementation safety checks

Before adding or changing a Triton kernel, check these implementation details in
addition to numerical tests:

- Treat program IDs and grid-derived values as potentially narrow. On NVIDIA,
  non-first grid dimensions may be narrow; on AMD, Ascend, or other non-NVIDIA
  backends, every grid dimension may be narrow. Cast to `tl.int64` before using
  them in address arithmetic.
- Keep tensor address arithmetic in `tl.int64`, including block bases, strides,
  varlen sequence offsets, head offsets, and element offsets. Do not rely on
  `int16` or `int32` overflow behavior.
- Do not introduce new `tl.make_block_ptr` use. Triton marks it deprecated; use
  `TensorDescriptor` / `tl.make_tensor_descriptor` when descriptor semantics are
  needed, or explicit `tl.load` / `tl.store` pointer arithmetic following an
  existing validated kernel pattern.
- If a change touches grid shape, program-id mapping, varlen offsets, or pointer
  math, run a shape that exercises the changed path on NVIDIA and any supported
  non-NVIDIA backend, or add a precise verifier/skip for unsupported platforms.

## Code style constraints

- Use `fla.utils.device` and `fla.utils.device_platform` in tests instead of
  adding new hard-coded device strings.
- Use `IS_NVIDIA`, `IS_NVIDIA_HOPPER`, `IS_NVIDIA_BLACKWELL`, `IS_AMD`, and
  `IS_INTEL` from `fla.utils` for platform-specific skips or branches.
- Do not add new direct `torch.cuda` platform checks in correctness tests. If no
  existing helper covers the condition, add a small helper in `fla.utils` first.

## Default open-source test paths

Use these paths when looking for existing tests or deciding where to add new ones:

- `tests/ops/test_kda.py` — KDA kernel tests
- `tests/context_parallel/` — context-parallel variants (e.g., `test_cp_kda.py`, `test_cp_gdn.py`)
- `tests/models/test_modeling_kda.py` — end-to-end model tests for KDA

Adapt the path to the specific op you are working on (replace `kda` with `gdn`,
`gla`, `nsa`, `delta`, etc.).

## What NOT to put in this skill

- Internal-only test paths, local machine paths, private model names, and
  private workload identifiers.
- The open-source skill only points to public tests and public operator docs.

## Running tests

```bash
# Single op test
pytest tests/ops/test_kda.py -v

# Context parallel tests for the same op
pytest tests/context_parallel/test_cp_kda.py -v

# Model-level test
pytest tests/models/test_modeling_kda.py -v

# All dependent tests (see fla-mr-readiness skill)
python scripts/find_dependent_tests.py <changed_files>
```
