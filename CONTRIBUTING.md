# Contributing

Thank you for your interest in contributing to Flash Linear Attention! All pull requests are super welcomed and greatly appreciated.

## Table of Contents

* [Table of Contents](#table-of-contents)
* [Report Bugs](#report-bugs)
* [Ask Questions](#ask-questions)
* [Submit Pull Requests](#submit-pull-requests)
  * [Commit Message Convention](#commit-message-convention)
  * [PR Description](#pr-description)
  * [CI Pipeline](#ci-pipeline)
  * [Review Checklist](#review-checklist)
* [Setup Development Environment](#setup-development-environment)
  * [Prerequisites](#prerequisites)
  * [Setup](#setup)
  * [Lint Check](#lint-check)
  * [Test Locally](#test-locally)
* [Project Structure](#project-structure)
* [Code Style](#code-style)
  * [Copyright Header](#copyright-header)
  * [Formatting and Linting](#formatting-and-linting)
  * [Naming Conventions](#naming-conventions)
  * [Triton Kernels](#triton-kernels)
  * [PyTorch Operators](#pytorch-operators)
* [Adding a New Operator](#adding-a-new-operator)
* [Adding a New Model](#adding-a-new-model)
* [Testing](#testing)
  * [Running Tests](#running-tests)
  * [Writing Tests](#writing-tests)
  * [NaN Memory Poisoning](#nan-memory-poisoning)
* [Environment Variables](#environment-variables)
* [License](#license)

## Report Bugs

If you run into any weird behavior while using `fla`, feel free to open a new [issue](https://github.com/fla-org/flash-linear-attention/issues)! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open should include:

- A minimal code snippet that reproduces the bug.
- A clear explanation of what the issue is.

## Ask Questions

Please ask questions in [issues](https://github.com/fla-org/flash-linear-attention/issues) or on [Discord](https://discord.gg/vDaJTmKNcS). Check [FAQs.md](FAQs.md) first for common questions.

## Submit Pull Requests

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

Include a clear description with:

- **Summary**: What the PR does and why (bullet points preferred).
- **Test plan**: How the change is tested.
- **Breaking changes** (if any): List any API changes that are not backward compatible and describe the migration path.

See [recent PRs](https://github.com/fla-org/flash-linear-attention/pulls?q=is%3Apr+is%3Amerged) for examples.

### CI Pipeline

When you submit a PR, the following checks run automatically:

- **Linting** — Ruff + autopep8 via pre-commit
- **License header check** — Ensures copyright headers are present
- **GPU tests** — On NVIDIA H100/A100/4090 and Intel B580 (when available)
- **Benchmarks** — Performance regression checks

Add `[skip test]` to your commit message to skip GPU tests for documentation-only changes.

### Review Checklist

Before submitting, please go through the following checklist:

- Code follows the project's style conventions.
- Copyright header is present on all new files.
- Tests pass locally (`pytest tests/ops/test_<your_op>.py`).
- New operators include a naive reference implementation.
- Both forward and backward passes are tested.
- Gradient correctness is verified against a reference implementation.
- Pre-commit hooks pass (`pre-commit run --files <your_files>`).

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

2. Install in development mode:

    ```bash
    pip install -e '.[test]'
    ```

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
- Use `tl.make_block_ptr` for coalesced memory access.
- Gate autotune configs with `autotune_cache_kwargs` for cache support.
- Kernel naming: `<op>_fwd_kernel_<suffix>` / `<op>_bwd_kernel_<suffix>`.

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

### NaN Memory Poisoning

The test suite (`conftest.py`) automatically replaces `torch.empty` with NaN-filled tensors for `tests/ops/` and `tests/modules/`. This catches bugs where uninitialized memory is accidentally used. You don't need to do anything special — just be aware that your kernels must fully initialize all output tensors.

## Environment Variables

See [ENVs.md](ENVs.md) for a full list.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
