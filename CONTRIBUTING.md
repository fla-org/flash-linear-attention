# Contributing

Thank you for your interest in contributing to Flash Linear Attention! All pull requests are super welcomed and greatly appreciated.

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
* [Code Style](#code-style)
  * [Copyright Header](#copyright-header)
  * [Formatting and Linting](#formatting-and-linting)
  * [Docstrings and Comments](#docstrings-and-comments)
  * [Prose and Markdown](#prose-and-markdown)
  * [Naming Conventions](#naming-conventions)
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
  * [Commit Message Convention](#commit-message-convention)
  * [PR Description](#pr-description)
  * [CI Pipeline](#ci-pipeline)
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

1. **Match the reference numerically.** Every optimized kernel must agree with its naive reference within `assert_close` tolerance. Pure refactors and other non-computational changes (rewrites, fused paths, autotune tweaks) must leave outputs **and gradients** unchanged — verify before vs. after, don't assume.
2. **Find the root cause before patching.** Don't land band-aid fixes. If a change appears to help but you can't explain why, keep digging.
3. **Reuse over duplication.** Check `fla/ops/common/` and existing operators before writing new kernels; unify shared code paths instead of copying per-operator variants.
4. **Audit every callsite when touching shared code.** Renaming a symbol, changing a config field, or editing a common kernel/component means updating *all* of its uses in one pass — not one spot at a time. Changes in `fla/ops/` or `fla/modules/` ripple up to `fla/layers/` and `fla/models/`: check those consumers and decide explicitly whether the public interface needs to change. See [Triton Kernels](#triton-kernels) for the kernel-level checklist.
5. **Protect battle-tested paths; keep diffs minimal.** Changes to converged kernels or public APIs can silently break user code or checkpoints. Change only what the fix or feature needs, plus light incidental cleanups — don't revert or rewrite working code just because it could be cleaner (note it as optional in review instead). Flag risky changes, and when in doubt, ask.

## Setup Development Environment

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.7.0
- A GPU with Triton support (NVIDIA, AMD, or Intel)

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

    For non-CUDA backends, install the matching `torch` + `triton` flavor from the PyTorch index first (see [INSTALL.md](INSTALL.md)), then run the editable install with the matching extra (e.g. `.[rocm,test]`).

    > [!TIP]
    > If the install fails, double-check that your PyTorch version matches your local CUDA toolkit and that `nvcc` is available in your `PATH`.

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
pytest tests/
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
└── utils.py         # Global utilities and decorators

tests/
├── conftest.py      # Pytest config with NaN memory poisoning
├── ops/             # Operator tests
├── layers/          # Layer tests
├── models/          # Model tests
└── modules/         # Module tests
```

## Code Style

### Copyright Header

Every source file should begin with the following header:

```python
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
```

A CI workflow (`check-header.yml`) enforces this automatically.

### Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and [autopep8](https://github.com/hhatto/autopep8) for formatting. Pre-commit hooks run both automatically.

Key rules:
- **Max line length**: 127 characters
- **Target Python version**: 3.10+
- **Import sorting**: `isort`-compatible via Ruff (`fla` as first-party)
- **Type hints**: Use modern syntax (`X | None` instead of `Optional[X]`, `list[str]` instead of `List[str]`)
- Use `TYPE_CHECKING` for imports only needed at type-check time
- **Multi-line calls**: don't wrap a call that fits within the line limit. When a call does overflow, use a hanging indent with **one keyword argument per line**, not several per line.

### Docstrings and Comments

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

### Prose and Markdown

Don't hard-wrap prose at an arbitrary short column — this covers Markdown files, Python docstrings (including `Args:` / `Returns:` descriptions), and comment paragraphs. Either keep a paragraph on a single line, or break **only at sentence or clause boundaries** (after a `.`, `,`, `;`, or `—`), never mid-clause. In Python files the 127-character limit still applies, so wrap a docstring or comment at a clause boundary before it reaches the limit. Format Markdown tables with aligned columns so the `|` separators line up; table rows are exempt from the line limit.

### Naming Conventions

| Entity          | Convention         | Example                                   |
| --------------- | ------------------ | ----------------------------------------- |
| Classes         | PascalCase         | `GatedDeltaNet`, `LinearAttention`        |
| Functions       | snake_case         | `chunk_delta_rule`, `fused_recurrent_gla` |
| Constants       | UPPER_SNAKE_CASE   | `FLA_CI_ENV`, `SUPPORTS_AUTOTUNE_CACHE`   |
| Private helpers | Leading underscore | `_guarded_empty`, `_is_called_from_fla`   |

### Triton Kernels

- Kernel functions use `@triton.jit` with `do_not_specialize=['T']` for the sequence-length argument.
- Use `tl.constexpr` for compile-time constants (block sizes, flags like `USE_INITIAL_STATE`).
- Do not introduce new `tl.make_block_ptr` use; Triton marks it deprecated. Prefer
  `TensorDescriptor` / `tl.make_tensor_descriptor` when descriptor semantics are
  needed, or explicit `tl.load` / `tl.store` pointer arithmetic following an
  existing validated kernel pattern.
- Treat program IDs and grid-derived indices as potentially narrow integers.
  Cast them to `tl.int64` before multiplying by sizes, strides, or sequence
  offsets. This is especially important for non-first grid dimensions on NVIDIA
  and for non-NVIDIA backends where every grid dimension may be narrow.
- Keep all tensor address arithmetic in `tl.int64`: block bases, varlen offsets,
  strides, sequence positions, and element offsets must not rely on `int16` or
  `int32` overflow behavior.
- Gate autotune configs with `autotune_cache_kwargs` for cache support.
- Kernel naming: `<op>_fwd_kernel_<suffix>` / `<op>_bwd_kernel_<suffix>`.
- When renaming a symbol or adding/moving a parameter, sweep **every** site in one pass: the tensor, its `b_*` value, the `p_*` block pointer, comments, and — across the forward/backward kernels, host wrappers, and autograd `Function` — every signature, launch, return tuple, and `save_for_backward`/`saved_tensors` list. Keyword-argument order at each call site must match the parameter order in the signature.

### PyTorch Operators

- Wrap public-facing ops with the `@input_guard` decorator to ensure tensor contiguity.
- Use `@autocast_custom_fwd` / `@autocast_custom_bwd` for mixed-precision support.
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

Register your model in `fla/models/__init__.py` for auto-discovery.

## Testing

Every change to `fla/ops/` or `fla/modules/` must add or update the matching test under `tests/`, and a new operator must ship with a naive reference to compare against. Correctness is checked by **strict numerical comparison** against that reference — forward outputs *and* gradients — so a change that lacks a test, or only checks the forward pass, is not complete.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/ops/test_delta.py

# Run a specific test
pytest tests/ops/test_delta.py::test_chunk -v
```

### Writing Tests

Tests compare optimized (Triton) implementations against reference (naive/recurrent) implementations. Follow this pattern:

```python
import pytest
import torch

from fla.ops.your_op import chunk_your_op, fused_recurrent_your_op
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}".format(*test))
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

    # Triton implementation
    tri = chunk_your_op(q.clone(), k.clone(), v.clone())
    (tri * do).sum().backward()
    tri_dq, tri_dk, tri_dv = q.grad, k.grad, v.grad
    q.grad = k.grad = v.grad = None

    # Reference implementation
    ref = fused_recurrent_your_op(q.clone(), k.clone(), v.clone())
    (ref * do).sum().backward()
    ref_dq, ref_dk, ref_dv = q.grad, k.grad, v.grad

    assert_close('o', ref, tri, 0.006)
    assert_close('dq', ref_dq, tri_dq, 0.006)
    assert_close('dk', ref_dk, tri_dk, 0.006)
    assert_close('dv', ref_dv, tri_dv, 0.006)
```

Key guidelines:

- **Always use `torch.manual_seed(42)`** for reproducibility.
- **Use `assert_close`** from `fla.utils` for numerical comparison with relative tolerance.
- **Test both forward and backward** passes by computing gradients.
- **Use `device` from `fla.utils`** for device-agnostic tests.
- **Parametrize** with diverse shapes including non-power-of-2 sequence lengths (e.g., 63, 100, 2000).
- **Skip unsupported platforms** with `@pytest.mark.skipif(device_platform == 'intel', ...)` when needed.
- **Include test IDs** in parametrize for readable output.

**Naming and structure.** Name the file `tests/ops/test_<op>.py`, and name each test after the implementation entry point it exercises — `test_chunk`, `test_fused_recurrent`, `test_parallel` — mirroring the functions in `fla/ops/<op>/`. Distinguish a genuinely different code path with a short suffix (`test_chunk_varlen`, `test_fused_recurrent_state_v_first`). Prefer adding a new shape, dtype, or flag as a `@parametrize` case on an existing test rather than writing a new function; only add a new function when the path or purpose is clearly different — varlen vs. dense, a specific feature flag, or a separate entry point. See `tests/ops/test_gla.py` and `tests/ops/test_gdn.py` for the pattern.

### NaN Memory Poisoning

The test suite (`conftest.py`) automatically replaces `torch.empty` with NaN-filled tensors for `tests/ops/` and `tests/modules/`. This catches bugs where uninitialized memory is accidentally used. You don't need to do anything special — just be aware that your kernels must fully initialize all output tensors.

## Benchmarking

Any change that can affect performance — a new or rewritten kernel in `fla/ops/` or `fla/modules/`, an autotune or backend tweak — should come with before/after numbers in the PR, measured on the same hardware and workload. `[Perf]` PRs must include them.

Benchmark only against a **green test gate**. A kernel that runs faster but fails its `tests/ops/test_<op>.py` (forward, backward, and NaN-poisoned init) is not an improvement, so confirm correctness first — see [Testing](#testing).

**Op microbenchmark** — times forward and forward+backward across a shape sweep, and compares against a git ref (it builds a throwaway worktree, so your working tree is untouched):

```bash
python -m benchmarks.ops.run --op chunk_gla --base main   # one op vs. main
python -m benchmarks.ops.run --list                       # registered ops
```

New ops are registered in `benchmarks/ops/registry.py`.

**Correctness-gated driver** — runs the op's pytest as a frozen gate, then benchmarks, and refuses to report a speedup on a red gate. Use it as the per-iteration command when optimizing a kernel; the `fla-optimization-loop` agent skill drives the full loop:

```bash
python -m benchmarks.ops.verify --op chunk_gla --base main
```

**Model-level throughput and generation:**

```bash
python benchmarks/benchmark_training_throughput.py --name kda --batch_size 2 --seq_len 8192 [--varlen]
python benchmarks/benchmark_generation.py --name kda
```

For profiling (Nsight Compute, hot-instruction analysis), see the `fla-nvidia-performance` agent skill. Report throughput (tokens/s or iters/s) and, when relevant, peak memory, and flag any shape or backend that regressed and why.

## Submit Pull Requests

Once your change is implemented, tested, and (if it touches performance) benchmarked, open a pull request against `main`.

> [!NOTE]
> Please include tests with every pull request if applicable!

- **Keep the scope focused**: one PR should do one thing. If you have multiple unrelated changes, please split them into separate PRs.
- **Use Draft PRs**: feel free to open a draft early for design feedback or work-in-progress discussion.

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

### PR Description

Lead with what changed and why, at a high level — describe the behavior or capability, not a file-by-file walkthrough. Include:

- **Summary**: the change and its motivation, stated up front. Keep it concise; reviewers read the diff for details.
- **Test plan**: how you verified it (commands run, hardware used).
- **Breaking changes** (if any): list any API changes that are not backward compatible, and describe the migration path.

See [recent PRs](https://github.com/fla-org/flash-linear-attention/pulls?q=is%3Apr+is%3Amerged) for examples.

### CI Pipeline

When you submit a PR, the following checks run automatically:

- **Linting** — Ruff + autopep8 via pre-commit
- **License header check** — Ensures copyright headers are present
- **GPU tests** — On NVIDIA H100/A100/4090 and Intel B580 (when available)
- **Benchmarks** — Performance regression checks; results are posted automatically as a PR comment

Add `[skip test]` to your commit message to skip GPU tests for documentation-only changes. For `[Perf]` changes, include before/after numbers in the PR — see [Benchmarking](#benchmarking).

### Review Checklist

Before submitting, please go through the following checklist:

- Code follows the project's style conventions.
- Copyright header is present on all new files.
- Changes to `fla/ops/` or `fla/modules/` add or update the matching test in `tests/`.
- Tests pass locally (`pytest tests/ops/test_<your_op>.py`).
- New operators include a naive reference implementation.
- Both forward and backward passes are tested.
- Gradient correctness is verified against a reference implementation.
- Pre-commit hooks pass (`pre-commit run --files <your_files>`).

## Environment Variables

See [ENVs.md](ENVs.md) for a full list.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
