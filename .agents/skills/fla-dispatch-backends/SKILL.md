---
name: fla-dispatch-backends
description: >
  Workflow for FLA backend dispatch decorators and backend implementations. Use
  when touching fla.ops.backends, @dispatch-decorated functions, BaseBackend
  subclasses, backend verifier methods, backend env vars, or backend tests.
---

# FLA Dispatch Backends Skill

Use this skill for the runtime backend dispatch system implemented in
`fla/ops/backends/__init__.py`.

## Core model

- Public functions opt in with `@dispatch('<operation>')`.
- First call lazily imports `fla.ops.<operation>.backends`, unless the operation
  has a custom module in `_OPERATION_BACKEND_MODULES` (for example `modules`).
- Backend modules create `BackendRegistry('<operation>')` and register
  `BaseBackend` subclasses.
- Dispatch tries registered backends sorted by `priority` where lower means
  higher priority.
- A backend is considered only when `is_available()` and `is_enabled()` are both
  true.
- Runtime dispatch checks `is_available()` and `is_enabled()` directly; do not
  rely on the cached `can_use()` path inside code that must be torch.compile
  friendly.
- If `<func_name>_verifier` exists, it must return `(True, None)` or
  `(False, reason)`. Rejected calls fall back to the next backend.
- If no backend handles the call, dispatch runs the original implementation.
- `FLA_DISABLE_BACKEND_DISPATCH=1` bypasses the decorator entirely.
- The dispatch wrapper is marked with `torch.compiler.disable`, so keep backend
  selection logic outside compiled graphs and keep compiled work inside the
  selected backend implementation.

## Backend implementation checklist

For a new backend:

1. Add a `BaseBackend` subclass under the operation's `backends/` package.
2. Set `backend_type`, `package_name`, `env_var`, `default_enable`, and `priority`.
3. Implement `<public_function_name>_verifier(...)` with the same public call
   surface as the decorated function.
4. Implement `<public_function_name>(...)` and keep return values identical to
   the default implementation.
5. Register the backend in the operation's `backends/__init__.py`.
6. Add tests that cover accepted dispatch, verifier rejection, and fallback.

## Verifier rules

- Verifiers must be cheap, deterministic, and side-effect free.
- Return a specific rejection reason; it is logged once and is useful in CI logs.
- Check dtype, shape, layout, inference/training mode, external package
  requirements, env flags, and unsupported options before calling backend code.
- Do not silently copy or normalize inputs in a verifier; do that in the backend
  implementation only when it is part of the backend contract.
- If a backend supports only inference, check `torch.is_grad_enabled()` or
  `torch.is_inference_mode_enabled()` as appropriate.
- Do not mutate global backend registries, environment variables, tensors, RNG
  state, or caches from a verifier.

## Decorator placement

- Decorate public operation entry points, not private helpers that are only used
  inside one backend.
- Keep the decorated function as the semantic fallback implementation. A user
  should be able to set `FLA_DISABLE_BACKEND_DISPATCH=1` and still get the same
  API behavior.
- Use the operation name that maps to the backend package. For normal ops,
  `@dispatch('kda')` maps to `fla.ops.kda.backends`; special cases belong in
  `_OPERATION_BACKEND_MODULES`.
- Do not add import-time side effects in backend packages beyond registering
  backends.

## Testing guidance

- Test the public decorated function, not only the backend helper.
- Force dispatch off with `FLA_DISABLE_BACKEND_DISPATCH=1` when comparing against
  the Triton/default path.
- Force or disable backend-specific env vars (`FLA_FLASH_KDA`, `FLA_TILELANG`,
  `FLA_INTRACARD_CP`) when testing route behavior.
- Include at least one rejection test for each verifier branch added or changed.
- For backend changes under `fla/ops/<op>/backends/`, ensure dependent op tests
  still run; `scripts/find_dependent_tests.py` maps backend changes back to the
  decorated op files.

## Style constraints

- Use platform helpers from `fla.utils` for hardware/platform decisions instead
  of adding new direct `torch.cuda` checks in public code or tests. If no helper
  covers the condition, add a small helper in `fla.utils` first.
- Keep backend imports lazy inside backend implementations when importing an
  optional package would otherwise break environments without that package.
- Keep error/rejection messages precise and user-facing; they appear in logs and
  tests may assert them.
