---
name: fla-dispatch-backends
description: Change FLA backend registration, dispatch, or verifiers while preserving routing and call contracts.
---

# Backend dispatch

Use this skill for registration, selection, or verifier changes in `fla/backends.py` and operation backends. [CONTRIBUTING.md](../../../CONTRIBUTING.md) defines style and public compatibility; [fla/backends.py](../../../fla/backends.py) defines the routing behavior.

## 1. Define the supported calls

Identify the default entry point and the calls the backend can handle: inputs, layouts, dtypes, gradient modes, and optional dependencies. Match the entry point's signature, parameter order, defaults, return structure, and supported gradients.

A verifier returns `(True, None)` or `(False, reason)`. It checks support without changing inputs or global state. Keep it beside the implementation; put helpers shared by several implementations in a utility module.

## 2. Register the implementation

Keep the default entry point backend-independent with bare `@dispatch`. Choose the registration form that matches the implementation:

| Implementation               | Registration                                                         |
| ---------------------------- | -------------------------------------------------------------------- |
| Function with the same API   | `@register(entry, backend=TritonAscendBackend, verifier=...)`        |
| Adapter with backend policy  | `@register(entry)` on a `BaseBackend` subclass                       |

An adapter defines methods named after the entry point, such as `chunk_kda` and `chunk_kda_verifier`. Register it where it is defined and import optional kernels only when selected.

Import the owning package's backends after its default entry points are defined. For implementations with platform-specific dependencies, guard the import with the shared backend's `is_available()`. Do not gate registration on enable switches; dispatch reads those at call time. Keep per-operation switches on `@register(..., env_var=...)` and implementation paths out of the shared dispatcher.

Existing owners using `@dispatch('<operation>')` use `@register('<operation>')` on adapter classes and load implementations through the owner's backend package. Both forms use the same registration and selection logic.

## 3. Preserve routing behavior

For each call, try backends in ascending priority order; ties retain registration order:

1. Check availability and enablement.
2. Run the verifier; rejection proceeds to the next candidate.
3. Call the selected implementation. Its exceptions propagate.
4. If no candidate accepts, call the default implementation.

Registration replaces an existing backend of the same type. Function registration keys the registry by the unwrapped entry point, so compiler decorators and public aliases share that registry. Legacy owner registries must load their backend package before lookup.

Keep selection outside `torch.compile` graphs. Do not cache availability or enablement in the dispatch wrapper: registered backends must remain eligible when their switches change. Python's import cache handles modules already loaded, and unused optional dependencies must not prevent package import.

## 4. Verify through the entry point

Run the decorated entry point for accepted calls, verifier rejection, disabled/unavailable backends, and fallback. Check optional arguments omitted as well as supplied, changed support boundaries, outputs, and supported gradients.

Use explicit backend flags in routing tests. For the default path, set `FLA_DISABLE_BACKEND_DISPATCH=1` before importing FLA. Use fresh processes for import-order checks, and verify public aliases and shared backend ownership after registration changes.

Find affected consumers with `scripts/find_dependent_tests.py` and run the relevant operator, layer, and model tests. Host routing tests establish dispatch behavior; accelerator reference tests establish numerical correctness.
