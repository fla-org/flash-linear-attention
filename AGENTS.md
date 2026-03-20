# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Flash Linear Attention (`fla`) is a Python GPU kernel library providing Triton-based implementations of linear attention models. It is **not** a web application—there are no servers, databases, or Docker services to run.

### Dependencies

Install in editable mode: `pip install -e ".[test,benchmark]"` plus `ruff autopep8` for linting. All deps are declared in `pyproject.toml`.

### Lint

```bash
ruff check .
```

There are 9 pre-existing `RUF022` (unsorted `__all__`) warnings in the repo—these are not regressions.

### Tests

```bash
pytest tests/ -v
```

**All 1142 tests require an NVIDIA GPU with CUDA + Triton.** The Cloud Agent VM has no GPU, so tests will fail with `RuntimeError: Found no NVIDIA driver`. Test collection (`pytest --collect-only`) works and is useful for verifying test infrastructure.

### Key gotchas

- **No CPU fallback for forward pass:** Model configs can be created and models instantiated on CPU, but `model(input_ids)` and `model.generate(...)` fail on CPU because the Triton kernels have no CPU implementation. The `fla/utils.py` warns "Triton is not supported on current platform, roll back to CPU" but this only affects utility functions, not the core ops.
- **Fused modules:** When creating models on CPU for inspection, set `fuse_cross_entropy=False`, `fuse_norm=False`, `fuse_swiglu=False` in the config to avoid Triton-dependent fused operations during model construction.
- **CI:** Tests run on self-hosted GPU runners (4090, A100, H100, Intel B580) via GitHub Actions. See `.github/workflows/reusable-ci-tests.yml`.
- **Pre-commit hooks:** Configured in `.pre-commit-config.yaml` using ruff and autopep8.
