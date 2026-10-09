# Contributing

Contributions to Flash Linear Attention are welcome. This guide defines the development, validation, and pull request conventions. Start with the core principles, then follow the setup and verification steps relevant to your change.

## Table of Contents

* [Report Bugs](#report-bugs)
* [Ask Questions](#ask-questions)
* [Core Principles](#core-principles)
* [Setup Development Environment](#setup-development-environment)
  * [Prerequisites](#prerequisites)
  * [Setup](#setup)
  * [Lint Check](#lint-check)
  * [Test Locally](#test-locally)
* [Project Structure](#project-structure)
  * [Public API compatibility](#public-api-compatibility)
* [Code Style](#code-style)
  * [Copyright Header](#copyright-header)
  * [Formatting and Linting](#formatting-and-linting)
  * [Imports](#imports)
  * [Naming Conventions](#naming-conventions)
  * [Docstrings and Comments](#docstrings-and-comments)
  * [Prose and Markdown](#prose-and-markdown)
  * [Triton Kernels](#triton-kernels)
  * [PyTorch Operators](#pytorch-operators)
* [Adding a New Operator](#adding-a-new-operator)
* [Adding a New Model](#adding-a-new-model)
* [Testing](#testing)
  * [Running Tests](#running-tests)
  * [Writing Tests](#writing-tests)
  * [NaN Memory Poisoning](#nan-memory-poisoning)
* [Benchmarking](#benchmarking)
* [Submit Pull Requests](#submit-pull-requests)
  * [PR Scope and Size](#pr-scope-and-size)
  * [Commit Message Convention](#commit-message-convention)
  * [PR Description](#pr-description)
  * [Evidence by Change Type](#evidence-by-change-type)
  * [CI Pipeline](#ci-pipeline)
  * [Responding to Review](#responding-to-review)
  * [Review Checklist](#review-checklist)
* [Environment Variables](#environment-variables)
* [License](#license)

## Report Bugs

If you run into any weird behavior while using `fla`, feel free to open a new [issue](https://github.com/fla-org/flash-linear-attention/issues)! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open should include:

- A minimal code snippet that reproduces the bug.
- A clear explanation of what the issue is.

## Ask Questions

Please ask questions in [issues](https://github.com/fla-org/flash-linear-attention/issues) or on [Discord](https://discord.gg/vDaJTmKNcS). Check [FAQs.md](FAQs.md) first for common questions.

## Core Principles

Read these before changing any kernel — they are the bar every PR is held to.

1. **Match the reference numerically.** Every optimized kernel must agree with its reference within the existing `assert_close` tolerance. Refactors, fused paths, and autotune tweaks must preserve outputs **and supported gradients** within those tolerances — verify before vs. after, don't assume. Claim bitwise identity only when it has been checked.
2. **Find the root cause before patching.** Don't land band-aid fixes. If a change appears to help but you can't explain why, keep digging.
3. **Reuse over duplication.** Check `fla/ops/common/` and existing operators before writing new kernels; unify shared code paths instead of copying per-operator variants.
4. **Audit every callsite when touching shared code.** Renaming a symbol, changing a config field, or editing a common kernel/component means updating *all* of its uses in one pass — not one spot at a time. Changes in `fla/ops/` or `fla/modules/` ripple up to `fla/layers/` and `fla/models/`: check those consumers and decide explicitly whether the public interface needs to change. See [Triton Kernels](#triton-kernels) for the kernel-level checklist.
5. **Protect battle-tested paths; keep diffs minimal.** Changes to converged kernels or public APIs can silently break user code or checkpoints. Change only what the fix or feature needs, plus light incidental cleanups — don't revert or rewrite working code just because it could be cleaner (note it as optional in review instead). Flag risky changes, and when in doubt, ask.

## Setup Development Environment

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.7.0
- A supported NVIDIA, AMD, or Intel GPU, or Ascend NPU, to execute accelerator kernels. Documentation and applicable host-side checks do not require an accelerator.

### Setup

1. Fork flash-linear-attention ([fork](https://github.com/fla-org/flash-linear-attention/fork)) on GitHub and clone the repository.

    ```bash
    git clone git@github.com:<your username>/flash-linear-attention.git
    cd flash-linear-attention

    git remote add upstream git@github.com:fla-org/flash-linear-attention.git
    ```

2. Install in development mode with a backend extra (`cuda` / `rocm` / `xpu` / `npu` / `cpu`):

    ```bash
    pip install -e '.[cuda,test]'
    ```

    Follow [INSTALL.md](INSTALL.md) for the chosen backend before the editable install. ROCm, XPU, and CPU use their matching PyTorch package indexes; Ascend NPU requires CANN and its own `torch_npu` / `triton-ascend` packages. Select the corresponding extra, such as `.[rocm,test]` or `.[npu,test]`.

    > [!TIP]
    > For CUDA components that compile native extensions, check the PyTorch/CUDA toolkit compatibility and that `nvcc` is available in your `PATH`.

3. Setup the [`pre-commit`](https://pre-commit.com) hooks:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

### Lint Check

To check the linting, run:

```bash
pre-commit run --all-files
```

### Test Locally

```bash
FLA_CI_ENV=0 pytest tests/
```

## Project Structure

```
fla/
├── layers/          # PyTorch attention layer implementations
├── ops/             # Triton kernel operators (the core of the project)
│   ├── common/      # Shared kernels reused across operators
│   └── <op_name>/   # Each operator in its own directory
│       ├── __init__.py
│       ├── naive.py             # Reference implementation in pure PyTorch
│       ├── chunk.py             # Chunk-based implementation
│       ├── parallel.py          # Parallel Triton kernel implementation
│       ├── fused_recurrent.py   # Fused recurrent implementation
│       └── README.md            # (optional) Mathematical derivations
├── models/          # Full language model definitions (config + modeling)
├── modules/         # Utility modules (norms, feature maps, rotary, etc.)
└── utils/           # Global utilities and decorators

tests/
├── context_parallel/   # Context-parallel tests
├── layers/             # Layer tests
├── models/             # Model tests
├── modules/            # Module tests
├── ops/                # Operator tests
├── utils/              # Tests for fla.utils
└── conftest.py         # Pytest config with NaN memory poisoning
```

### Public API compatibility

Implementation refactors preserve documented operator entry points exported from `fla.ops.<operator>` (for example, `fla.ops.kda.chunk_kda`), public imports from `fla.modules` and `fla.layers`, and documented model/configuration APIs in `fla.models`. This includes signatures, argument defaults, return structures, supported gradients, numerical tolerances, and checkpoint/state-dict compatibility. An intentional incompatible change needs prior agreement, a documented migration path, and compatibility aliases or a deprecation period where practical.

Backend adapters, kernel files, private helpers, and their implementation paths are internal. Moving them does not change the public contract. Keep public re-exports stable, audit callers, and test old imports when reorganizing packages; compatibility shims do not make every internal symbol a permanent public API.

## Code Style

### Copyright Header

Project Python source files should begin with the following header:

```python
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
```

A CI workflow (`check-header.yml`) enforces this for Python files, with exclusions defined in `scripts/check_header.py`.

### Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and [autopep8](https://github.com/hhatto/autopep8) for formatting. Pre-commit hooks run both automatically.

Key rules:

- **Max line length**: 127 characters
- **Target Python version**: 3.10+
- **Import sorting**: `isort`-compatible via Ruff (`fla` as first-party)
- **Type hints**: Use modern syntax (`X | None` instead of `Optional[X]`, `list[str]` instead of `List[str]`)
- Use `TYPE_CHECKING` for imports only needed at type-check time
- **Line width**: use the full 127 characters before reaching for a line break — a statement that fits on one line stays on one line.
- **Calls**: prefer keyword arguments over positional ones. A call that fits within the limit stays on one line; a call that overflows breaks with a hanging indent, **one keyword argument per line** — never several.
- **Parameter order**: keep related parameters adjacent, and pass keyword arguments at call sites in the same order they appear in the signature.

### Imports

Prefer absolute `from ... import ...` imports for project code. Import the symbols you need directly; when you need a module object, import it from its parent package.

```python
from fla.ops.kda.backends.tilelang import KDATileLangBackend
from fla.utils import env
```

Avoid unnecessary aliases and ad hoc abbreviations. Keep established conventions such as `torch.nn.functional as F` and `triton.language as tl`, and use descriptive aliases when they clarify real name collisions. A long package path alone is not a reason to rename the imported object.

For tests that monkeypatch module globals, retain the module object and patch the namespace where the code under test looks up the value. Module access is also appropriate when values may be rebound at runtime. Keep the module's original name unless an alias helps distinguish it from another object in the same scope.

### Naming Conventions

| Entity          | Convention         | Example                                   |
| --------------- | ------------------ | ----------------------------------------- |
| Classes         | PascalCase         | `GatedDeltaNet`, `LinearAttention`        |
| Functions       | snake_case         | `chunk_delta_rule`, `fused_recurrent_gla` |
| Constants       | UPPER_SNAKE_CASE   | `FLA_CI_ENV`, `SUPPORTS_AUTOTUNE_CACHE`   |
| Private helpers | Leading underscore | `_guarded_empty`, `_is_called_from_fla`   |

### Docstrings and Comments

Comments and docstrings are hints for other readers, not a chain of thought. Give the reader what the code cannot say for itself, in as few words as possible — correct, simple, and with no narration of your reasoning.

Write docstrings at a high level: say what the function or test does and the contract it guarantees, and let the code speak for its own mechanics. A docstring that only restates the body is better left unwritten.

Use a two-line hanging format for `Args:` / `Returns:` entries: a `name (type, Optional):` header line, then the description and `Default:` on the next indented line(s).

```python
Args:
    hidden_size (int, Optional):
        The hidden size of the input. Default: 2048.
    use_output_gate (bool, Optional):
        Whether to apply a gated RMSNorm on the attention output. Default: `False`.
```

Capitalize `Optional` (not `optional`), put the default as `Default: <value>` (not "Defaults to ..."), and wrap `True` / `False` / `None` in backticks. See `fla/layers/gla.py::GatedLinearAttention` for the canonical example.

Keep inline comments restrained, especially in Triton kernels: shape annotations (e.g. `# [BL, BD]`) plus at most a one-line "why" for genuinely non-obvious tricks. Avoid multi-line derivations and narration that just restates the next line — math derivations belong in the operator's `README.md`, the PR description, or a single pointer, not inline.

Put explanatory comments on their own line **above** the code they describe, not trailing it — write `# why` on the line above `x = f()`, not `x = f()  # why`. Start the comment text with a lowercase letter (`# guard against overflow`, not `# Guard against overflow`), and wrap a multi-line comment at clause boundaries like other prose. Reserve inline trailing comments for terse shape / type annotations like `# [BL, BD]`.

Comments and docstrings must not go stale: when a change makes one factually wrong — a renamed symbol, a changed default, a removed code path — update or delete it in the same commit. An outdated comment is worse than none. If you can't tell whether a comment is still true, keep it and say so in the PR description; don't delete a "why" comment you merely can't verify. Fixing stale content means fixing the words, not reformatting the surrounding comment or docstring style.

Beyond narration that restates the next line, these comment patterns are banned:

- **Banner blocks** (`##### ... #####`): section boundaries should be visible from the code structure; use a blank line.
- **Commented-out code**: delete it — git has the history. Exception: commented-out *configurations* deliberately kept as documented alternatives (e.g. known-good autotune configs for a future dtype) may stay if a one-line comment says why they are kept.
- **Personal asides** (`# XY: remove this?`): a name in a comment is not an owner. Convert it to a TODO with an anchor, or delete it.
- **Anchorless TODOs**: a TODO must name when it can be acted on — a link to a tracking issue (this repo or upstream), a version bound (`TODO: drop once we require triton>=3.5`), or an externally checkable event. This applies to TODOs in docstrings too; see `fla/ops/utils/op.py::safe_dot` for the pattern.

Never treated as excess comments: the license header required by `scripts/check_header.py`, a one-line attribution with a URL for adapted code, shape/dtype annotations, and `NOTE:` / `WARNING:` prefixes on a genuine "why" comment.

### Prose and Markdown

Don't hard-wrap prose at an arbitrary short column — this covers Markdown files, Python docstrings (including `Args:` / `Returns:` descriptions), and comment paragraphs. Either keep a paragraph on a single line, or break **only at sentence or clause boundaries** (after a `.`, `,`, `;`, or `—`), never mid-clause. In Python files the 127-character limit still applies, so wrap a docstring or comment at a clause boundary before it reaches the limit. Format Markdown tables with aligned columns so the `|` separators line up; table rows are exempt from the line limit.

### Triton Kernels

- Kernel functions use `@triton.jit` with `do_not_specialize=['T']` for the sequence-length argument.
- Use `tl.constexpr` for compile-time constants (block sizes, flags like `USE_INITIAL_STATE`).
- Write block accesses as explicit offset vectors (`offset + tl.arange`) with
  plain `tl.load` / `tl.store`. Masked loads must cover every dimension that
  can overrun at any call site, with `other=` where masked lanes matter; assert
  any divisibility you rely on. Do not use `tl.make_block_ptr` / `tl.advance`:
  deprecated upstream and removed in triton main. (`triton_ascend/` is
  exempt — triton-ascend still requires block pointers.)
- `tl.make_tensor_descriptor` (TMA) is an opt-in optimization for hot-path tiles, not a default substitute for block access. Availability is `IS_TMA_SUPPORTED` in `fla/utils/hardware.py`, which covers Nvidia Hopper and newer plus AMD gfx1250, and is gated behind `FLA_USE_TMA=1` on every backend. The AMD side is an arch allowlist rather than a version floor — gfx942 and gfx950 have no such lowering — so extend `IS_AMD_TMA_ARCH` explicitly when another arch gains it, and don't assume a higher `gfx` number implies support. Note that no CI runner currently has a TMA-capable AMD arch, so that path is only exercised by local runs.
- Descriptors require 16-byte-aligned bases and stride multiples, a stride-1 innermost dim, no transposed blocks, and a registered allocator for device-side descriptors. Nothing verifies the alignment for you and a violation faults at launch rather than failing to compile, so derive a guard on the host from the dtype and the shapes the op is actually tested at, and fall back to the plain-pointer path when it fails — a head dim of `K=60` in fp16, as KDA is tested at, is 120 bytes per row and cannot back a descriptor. Descriptor stores are also asynchronous, so a kernel must never read back a location it just stored through a descriptor.
- Use TMA when a per-kernel benchmark in the PR shows it pays off, behind a `USE_TMA: tl.constexpr` flag that keeps the plain-pointer path intact, as in `fla/ops/utils/solve_tril.py`. A benchmark on one vendor does not carry over to the other: the descriptor path in `solve_tril` was tuned on Hopper and has not been measured on gfx1250.
- Treat program IDs and grid-derived indices as potentially narrow integers.
  Cast them to `tl.int64` before multiplying by sizes, strides, or sequence
  offsets. This is especially important for non-first grid dimensions on NVIDIA
  and for non-NVIDIA backends where every grid dimension may be narrow.
- Keep all tensor address arithmetic in `tl.int64`: block bases, varlen offsets,
  strides, sequence positions, and element offsets must not rely on `int16` or
  `int32` overflow behavior.
- Pass `**autotune_cache_kwargs` to autotune decorators for compatible cache-result handling.
- Kernel naming: `<op>_fwd_kernel` / `<op>_bwd_kernel`, with a descriptive suffix for variants.
- When renaming a symbol or adding/moving a parameter, sweep **every** site in one pass: the tensor, its `b_*` value, the `p_*` block pointer, comments, and — across the forward/backward kernels, host wrappers, and autograd `Function` — every signature, launch, return tuple, and `save_for_backward`/`saved_tensors` list.

### PyTorch Operators

- Use `@input_guard` on custom autograd entry points to ensure tensor contiguity and the input device context.
- Apply `@autocast_custom_fwd` to `Function.forward` and `@autocast_custom_bwd` to `Function.backward` for mixed-precision support; follow an existing operator's decorator order.
- Provide a reference (naive) implementation in `naive.py` for testing.

## Adding a New Operator

When adding a new operator under `fla/ops/<op_name>/`:

1. **Create the directory** with an `__init__.py` that exports the public API.
2. **Write a naive implementation** (`naive.py`) in pure PyTorch. This serves as the ground-truth reference for testing.
3. **Implement the optimized kernel(s)** in `chunk.py`, `parallel.py`, and/or `fused_recurrent.py`.
4. **Reuse shared kernels** from `fla/ops/common/` where possible (e.g., `chunk_fwd_o`, `chunk_gated_delta_rule_fwd_h`).
5. **Add tests** in `tests/ops/test_<op_name>.py` (see [Testing](#testing) below).
6. **(Optional)** Add a `README.md` with mathematical derivations.

## Adding a New Model

Each model lives under `fla/models/<model_name>/` with:

- `configuration_<model_name>.py` — Config class extending `PretrainedConfig`
- `modeling_<model_name>.py` — Model, PreTrainedModel, and ForCausalLM classes
- `__init__.py` — Auto-registration with `transformers`

Register the config and model with the appropriate Transformers auto classes in the model package's `__init__.py`, then import and export them from `fla/models/__init__.py`. See `fla/models/gla/__init__.py` for an example.

## Testing

Every change to `fla/ops/` or `fla/modules/` must add or update the matching test under `tests/`, and a new operator must ship with a naive reference to compare against. Correctness is checked by **numerical comparison** against that reference — forward outputs *and* gradients for training APIs. Forward-only inference APIs need reference output/state checks and an explicit statement that backward is unsupported.

Use `FLA_CI_ENV=0` for strict local validation. `assert_close` compares relative RMS error with an absolute-error shortcut; CI mode and explicitly warning-only checks can turn some tolerance failures into warnings. A green run with warnings is not evidence that the original tolerance passed. Preserve the existing per-test thresholds.

### Running Tests

```bash
# Run all tests
FLA_CI_ENV=0 pytest tests/

# Run a specific test file
FLA_CI_ENV=0 pytest tests/ops/test_delta.py

# Run a specific test
FLA_CI_ENV=0 pytest tests/ops/test_delta.py::test_chunk -v
```

### Writing Tests

Tests compare optimized implementations against differentiable naive or recurrent references. The simplified pattern below assumes a tensor-only return; adapt it to the actual return tuple, gates, and initial/final states. See `test_chunk` in [tests/ops/test_delta.py](tests/ops/test_delta.py) for a complete output, state, and gradient comparison. Reuse the operator's existing tolerances rather than treating the example threshold as a universal budget.

```python
import pytest
import torch

from fla.ops.your_op import chunk_your_op
from fla.ops.your_op.naive import naive_your_op
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float16),
            (2, 1000, 4, 128, torch.float16),
        ]
    ],
)
def test_chunk(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    k = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    v = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    do = torch.rand_like(v)

    tri = chunk_your_op(q=q.clone(), k=k.clone(), v=v.clone())
    (tri * do).sum().backward()
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    q.grad = k.grad = v.grad = None

    ref = naive_your_op(q=q.clone(), k=k.clone(), v=v.clone())
    (ref * do).sum().backward()
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad

    assert_close('o', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.006)
    assert_close('dk', ref_dk, tri_dk, 0.006)
    assert_close('dv', ref_dv, tri_dv, 0.006)
```

Key guidelines:

- **Always use `torch.manual_seed(42)`** for reproducibility.
- **Use `assert_close`** from `fla.utils` with the existing per-test error thresholds.
- **Use `device` and platform helpers from `fla.utils`** for device-agnostic tests. Reuse helpers such as `device_platform`, `IS_NVIDIA`, `IS_AMD`, and `IS_INTEL` instead of adding direct `torch.cuda` platform checks; keep vendor-specific profiling in benchmark scripts.
- **Parametrize** with diverse shapes including non-power-of-2 sequence lengths (e.g., 63, 100, 2000).
- **Skip unsupported platforms** with helpers such as `IS_INTEL` imported from `fla.utils`, e.g. `@pytest.mark.skipif(IS_INTEL, reason="unsupported on Intel")`.
- **Include test IDs** in parametrize for readable output.

**Naming and structure.** Name the file `tests/ops/test_<op>.py`, and name each test after the implementation entry point it exercises — `test_chunk`, `test_fused_recurrent`, `test_parallel` — mirroring the functions in `fla/ops/<op>/`. Distinguish a genuinely different code path with a short suffix (`test_chunk_varlen`, `test_fused_recurrent_state_v_first`). Prefer adding a new shape, dtype, or flag as a `@parametrize` case on an existing test rather than writing a new function; only add a new function when the path or purpose is clearly different — varlen vs. dense, a specific feature flag, or a separate entry point. See `tests/ops/test_gla.py` and `tests/ops/test_gdn.py` for the pattern.

### NaN Memory Poisoning

The fixture in `tests/conftest.py` poisons eligible floating-point and complex allocations made by FLA through `torch.empty`, `torch.empty_like`, and `Tensor.new_empty` during operator and module tests. It excludes allocations made directly by tests, gradient-requiring tensors, and compilation paths. This catches uninitialized reads; kernels must initialize every output element they expose. No manual opt-in is needed for covered tests.

## Benchmarking

Kernel changes and other changes that can affect performance — including autotune or backend tweaks — must include before/after numbers in the PR, measured on the same hardware and workload. This applies regardless of the PR title tag. For a new implementation, name the reference or existing path used as the comparison.

Benchmark only against a **green test gate**. A kernel that runs faster but fails its `tests/ops/test_<op>.py` (forward, backward, and NaN-poisoned init) is not an improvement, so confirm correctness first — see [Testing](#testing).

**Op microbenchmark** — times forward and forward+backward across a shape sweep, and compares against a git ref (it builds a throwaway worktree, so your working tree is untouched):

```bash
FLA_BENCH_BASE=$(git rev-parse origin/main)
python -m benchmarks.ops.run --op chunk_gla --base "$FLA_BENCH_BASE"
python -m benchmarks.ops.run --list
```

Resolve the baseline to a commit SHA: a checked-out branch cannot be reused by the comparison worktree, and slash-containing refs are unsuitable for its temporary directory names. Record that SHA with the results. The default sweep uses BF16 and per-operator shapes; add relevant dtype and dense/varlen coverage explicitly. New ops are registered in `benchmarks/ops/registry.py`.

**Correctness-gated driver** — runs the op's pytest file before benchmarking and stops on a failing test run by default. Keep references and tolerances fixed and verify that the changed path ran: the driver does not freeze files or reject an all-skipped suite. The `fla-optimization-loop` skill defines that iteration discipline:

```bash
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_gla --base "$FLA_BENCH_BASE"
```

**Model-level throughput and generation (CUDA):**

These scripts require additional dependencies: `accelerate` for training throughput and `datasets` for generation. Install them alongside the benchmark extra before running the examples:

```bash
pip install -e '.[cuda,benchmark]' accelerate

python benchmarks/benchmark_training_throughput.py --name kda --batch_size 2 --seq_len 8192
python benchmarks/benchmark_training_throughput.py --name kda --batch_size 2 --seq_len 8192 --varlen
python benchmarks/benchmark_generation.py --path fla-hub/gla-1.3B-100B
```

The generation benchmark loads a pretrained model or checkpoint via `--path`; it does not accept a model config name via `--name`.

For profiling (Nsight Compute, hot-instruction analysis), see the `fla-nvidia-performance` agent skill. Report throughput (tokens/s or iters/s) and, when relevant, peak memory, and flag any shape or backend that regressed and why.

## Submit Pull Requests

Open pull requests against `main`. Search existing issues and PRs before starting. For a new operator, model, public API, or algorithm, discuss the direction in an issue or draft PR first. Precision changes, relaxed tolerances, and replacement numerical algorithms need an RFC issue before a PR; ordinary tiling or scheduling changes within the same precision do not.

- **Keep the scope focused**: one PR should do one thing. If you have multiple unrelated changes, please split them into separate PRs.
- **Use Draft PRs**: open a draft for design feedback or incomplete work. Mark it ready when the implementation, applicable tests, and performance evidence are available. Describe missing validation honestly; do not tick a checklist item for work you intend to do later.
- **Read `AGENTS.md` and `.agents/skills/fla-pr-readiness` first**: they cover the PR checklist, test-plan requirements, and benchmark evidence standards expected of every pull request.
- **No busywork PRs**: don't open standalone PRs for typos or isolated style tweaks; fold them into a related substantive change instead.

### PR Scope and Size

One PR should solve one concrete problem or deliver one concrete capability. State that outcome in a sentence: what input or use case is affected, and what changes for the user. "Fix cached KDA state updates when caching is disabled" is a scope; "various attention improvements" is not. Changes sharing a directory or a broad label such as "cleanup" or "performance" are not necessarily one task.

Keep implementation, regression tests, and documentation for the same outcome together. Split unrelated fixes, speculative helpers, and independent optimizations into separate PRs. For a larger feature, prefer independently reviewable steps that each work and pass their relevant tests; link their dependencies.

The default limit is **500 changed lines**, counted as additions plus deletions in the complete PR diff against its base. A replaced line counts twice. Tests, documentation, renames with edits, and generated files count as GitHub reports them; the limit is not per commit. Do not remove tests, compress code, or hide generated changes to fit the limit.

A PR above 500 lines must tick the large-PR acknowledgement in the template and fill in `### Large PR justification`. Explain the single outcome, why these changes need to be reviewed together, and how to review them. Legitimate reasons include:

- A mechanical migration to one agreed style or API rule across its callers.
- One complete kernel or model implementation with its reference, registration, tests, and documentation.
- One coherent test suite or benchmark reorganization whose shared fixtures or entry points need to change together.

"Many files changed" or "all changes are related to performance" is not a justification. An exception acknowledges the review cost; reviewers can still ask for a split. Small PRs must satisfy the same scope rule.

The size check verifies the acknowledgement and a filled explanation (at least 20 non-whitespace characters); reviewers judge whether the exception is warranted. The check runs again when the PR body or diff changes.

### Commit Message Convention

Use a prefix tag in square brackets to categorize your change. Here are some common examples:

| Tag          | Usage                      | Example                                           |
| ------------ | -------------------------- | ------------------------------------------------- |
| `[Fix]`      | Bug fixes                  | `[Fix] Guard checkpoint weight re-initialization` |
| `[Misc]`     | Miscellaneous              | `[Misc] Upgrade minimum PyTorch requirement`      |
| `[Docs]`     | Documentation              | `[Docs] Update CP README`                         |
| `[CI]`       | CI/CD changes              | `[CI] Fix skip-test check failing on fork PRs`    |
| `[Test]`     | Test additions or fixes    | `[Test] Add varlen backward gradient checks`      |
| `[Perf]`     | Performance optimizations  | `[Perf] Fuse gate multiplication in delta rule`   |
| `[Refactor]` | Code refactoring           | `[Refactor] Unify chunk kernel entry points`      |
| `[Ops]`      | General operator changes   | `[Ops] Refactor common chunk reduction utilities` |
| `[Model]`    | Model architecture changes | `[Model] Add RoPE scaling to GLA config`          |
| `[Layer]`    | Layer-level changes        | `[Layer] Normalize initial state initialization`  |
| `[Attn]`     | Attention-related changes  | `[Attn] Add sliding window attention support`     |
| `[GDN]`      | Gated Delta Net            | `[GDN] Add fused gate kernel`                     |
| `[KDA]`      | Kimi Delta Attention       | `[KDA] Fix illegal memory access in backward`     |
| `[CP]`       | Context Parallel           | `[CP] Enable KCP for DPLR`                        |
| `[Conv]`     | Convolution                | `[Conv] Fix int32 overflow in varlen conv kernel` |
| `[CE]`       | Cross Entropy              | `[CE] Add logit softcapping support`              |

If your change doesn't fit any of the above, `[Misc]`/`[chore]` is the safe default.

Use the same convention for the PR title: `[Tag] <specific outcome>`. CI checks the PR title, accepts any alphanumeric tag beginning with a letter, and requires a space and description after it. Prefer `[Fix] Preserve KDA cache when caching is disabled` over `[Fix] Fix bugs`.

### PR Description

Use the [PR template](.github/pull_request_template.md). The description should let a reviewer identify the problem, outcome, affected users, and evidence before opening the diff. Keep it accurate as the implementation changes.

**Summary.** Start with one or two sentences stating the concrete outcome: what now works, what becomes faster, or what capability is added. For a bug, name the triggering condition and resulting behavior. Link the issue or design discussion. A routine fix usually needs only one short paragraph; a complex change may need a short explanation of the approach and its trade-offs.

Keep the description focused on observable behavior and review decisions. Include an implementation detail only when it explains correctness, the choice of algorithm, compatibility, or a material risk. Public API names can identify the contract being changed. Leave variable inventories, file-by-file summaries, line-by-line mechanics, and the history of attempted approaches to the diff or linked investigation. Avoid unsupported claims such as "more robust", "cleaner", or "bit-identical"; describe the behavior or equivalence actually verified. Do not repeat the same point in the title, summary, and a separate list of changes.

For example, these short openings are adapted from FLA PRs [#1299](https://github.com/fla-org/flash-linear-attention/pull/1299) and [#1293](https://github.com/fla-org/flash-linear-attention/pull/1293):

```markdown
## Summary

KDA calls with negative eigenvalues or missing optional gate parameters now use the Triton implementation, preventing incorrect results or failures when FlashKDA is installed.
```

```markdown
## Summary

Adds optional sliding-window attention to single-step decoding. Existing calls keep full-context attention by default.
```

**Test plan.** State which regression or new behavior the tests protect, the commands actually run, and their results. For accelerator tests, include the hardware, relevant software versions, and backend or environment flags. Separate passed, failed, skipped, and unrun checks; a test count alone does not show that the modified path ran. Reproduce suspected pre-existing failures on the baseline and link the evidence. Do not describe an incomplete run or a smoke subset as a full pass.

Identify dependent tests with `python scripts/find_dependent_tests.py <changed_file.py> [more_files.py ...]`, then run the returned tests and any focused regression needed to exercise the changed path. Audit shared callers in layers and models as well as the operator. State the relevant coverage: forward/backward, dense/varlen, boundary shapes, dtypes, states, dispatch fallbacks, or context parallelism, as applicable. Explain any coverage gap; do not weaken tolerances to obtain a pass.

**Benchmark / NCU.** Follow [Benchmarking](#benchmarking): correctness must pass before reporting a performance gain. Record the baseline and candidate commits, hardware, software, shapes, dtypes, backend flags, and exact command or timing method. Use a compact table with units, for example:

| Workload (shape, dtype, mode)   | Measurement             | Before  | After   | Change  |
| ------------------------------- | ----------------------- | ------- | ------- | ------- |
| `<B, T, H, K, V; bf16; dense>`  | `<forward latency, us>` | `<...>` | `<...>` | `<...>` |
| `<B, T, H, K, V; bf16; varlen>` | `<fwd+bwd latency, us>` | `<...>` | `<...>` | `<...>` |

Replace the example cells with measured results and identify the supported workloads. Report regressions and relevant peak-memory trade-offs alongside gains. A microbenchmark supports a kernel claim; a training or generation throughput claim needs the corresponding end-to-end measurement. `neutral` describes a measured result and does not replace numbers. Use `N/A` with a reason when no kernel or performance-relevant code changed. NCU profiling is useful when it explains the result; it is not a requirement for every PR. Link or collapse full logs and profiles instead of pasting them into the description.

**Breaking changes.** State the affected API, shape/dtype/backend support, defaults, or checkpoint/state-dict compatibility and any migration needed. If none, say `None` and still identify the affected users or paths in the summary. Coordinate breaking changes before implementation, as required by [AGENTS.md](AGENTS.md#scope-and-direction).

**Checklist and exceptions.** Preserve the checklist wording. The first four boxes mean the work is complete or genuinely inapplicable, with the reason recorded in the relevant section. Leave the minor/cosmetic box unchecked for substantive work. Tick the large-PR box only when the PR exceeds 500 changed lines, and give a specific reason for keeping it together. Checkboxes attest to evidence; they do not substitute for it.

### Evidence by Change Type

Use the rows that apply to your change. Keep related tests and evidence in the same PR.

| Change                          | Evidence needed for review                                                                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Bug fix                         | Trigger or minimal reproducer, root cause, and a regression test that fails before the fix and passes after it.                          |
| Kernel or module implementation | Added or updated matching tests, passing reference comparisons for outputs/states and supported gradients, and same-hardware benchmarks. |
| Performance or backend change   | Passing correctness tests for the changed path and fallbacks; reproducible before/after measurements and any regressions.                |
| Shared-code refactor            | Caller audit and relevant operator/layer/model tests; evidence that supported behavior and numerics are preserved.                       |
| New operator, model, or API     | Linked design discussion, supported inputs and usage, reference and registration where applicable, tests, and compatibility impact.      |
| Documentation or CI             | Relevant link, command, lint, or workflow checks; regression tests for changed validation logic. Explain inapplicable GPU checks.        |

Historical examples illustrate particular practices: [#1299](https://github.com/fla-org/flash-linear-attention/pull/1299) describes dispatch contracts and regression coverage; [#1159](https://github.com/fla-org/flash-linear-attention/pull/1159) separates kernel, operator, and model performance. Their length and older checklist wording are not templates to copy.

For further examples of contribution guidance, see [PyTorch](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md), [torchtitan's proof of value](https://github.com/pytorch/torchtitan/blob/main/CONTRIBUTING.md#proof-of-value), [vLLM's PR template](https://github.com/vllm-project/vllm/blob/main/.github/PULL_REQUEST_TEMPLATE.md), [SGLang](https://docs.sglang.io/docs/developer_guide/contribution_guide), and [FlashInfer](https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md#pull-request-guidelines). FLA's commands, numerical standards, and review requirements are defined here.

### CI Pipeline

The PR checks include:

- **PR metadata** — Title prefix, the first four completed checklist items, a justification for checked minor PRs, and the acknowledgement and justification for PRs above 500 changed lines. Editing the PR body re-runs this check; missing evidence may leave a draft red.
- **Linting** — Ruff + autopep8 via pre-commit
- **License header check** — Ensures copyright headers are present
- **Accelerator tests** — H100 smoke tests give early feedback. The full H100 pipeline is gated by maintainer permissions or review approval for external contributions; other backends have their own workflows. Check the actual jobs and tested commit before claiming coverage.
- **Benchmarks** — Eligible benchmark jobs compare performance and publish results as a PR comment. They do not replace measurements for workloads or backends outside their coverage.

See [.github/workflows](.github/workflows) for current triggers and supported runners. For documentation-only changes, `[skip test]` in the commit message skips tests in workflows using the shared CI setup; it does not skip every accelerator workflow or benchmark. Do not use it to avoid validation of code changes.

### Responding to Review

Reply to each actionable comment with the change made or a concrete reason for keeping the current approach. Keep follow-up edits within the PR's stated purpose; open a separate PR for independent work. Re-run affected checks after code changes and update the description, results, and remaining limitations to match the current commit. If new changes make earlier evidence inapplicable, collect new evidence before presenting the PR as ready.

### Review Checklist

Before submitting, please go through the following checklist:

- The title and summary identify one concrete outcome.
- The diff is within 500 changed lines, or the large-PR acknowledgement and justification are filled in.
- The description records actual validation and affected users, with no unrelated changes or implementation inventory.
- Code follows the project's style conventions.
- Copyright header is present on new project Python source files.
- Changes to `fla/ops/` or `fla/modules/` add or update the matching test in `tests/`.
- Tests pass locally (`FLA_CI_ENV=0 pytest tests/ops/test_<your_op>.py`).
- New operators include a naive reference implementation.
- Outputs and final states are checked against a reference; training APIs also check backward gradients.
- Pre-commit hooks pass (`pre-commit run --files <your_files>`).

## Environment Variables

See [ENVs.md](ENVs.md) for a full list.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
