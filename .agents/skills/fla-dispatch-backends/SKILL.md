---
name: fla-dispatch-backends
description: Change FLA backend registration, dispatch, or verifiers while preserving routing and call contracts.
---

# Backend dispatch

Use this skill when changing `fla/backends.py` or an operation's backend. The [backend module](../../../fla/backends.py) defines registration and selection; [CONTRIBUTING.md](../../../CONTRIBUTING.md) defines style and public compatibility.

## Add or change a backend

1. Define the supported calls: inputs, layouts, dtypes, gradient modes, and required dependencies.
2. Put a `BaseBackend` subclass in the operation's `backends/` package and apply `@register_backend('<operation>')`. Set availability, enablement, and priority for that implementation; import optional kernels inside its methods.
3. Match the dispatched function's signature, parameter order, defaults, return structure, and gradients. A verifier accepts or rejects the same call without changing its inputs or global state.
4. Import the class in the operation's `backends/__init__.py`. Decorate the default entry point with `@dispatch('<operation>')` from `fla.backends`.
5. Test through the decorated entry point and use `scripts/find_dependent_tests.py` to identify affected consumers.

## Routing contract

Backends run in ascending priority order, with registration order breaking ties. Selection checks availability and enablement on each call. A verifier returns `(True, None)` or `(False, reason)`; rejection tries the next candidate. If none accepts, the default implementation runs. Exceptions from the selected implementation propagate.

Register each backend under its owning operation and keep its methods specific to that operation. Registration replaces an existing instance with the same backend type. Load the owning backend package before resolving its registry.

Keep selection outside `torch.compile` graphs and avoid cached availability or enablement checks in the dispatch wrapper. Optional dependencies must remain lazy so an unused backend cannot prevent package import.

## Validation

Cover accepted dispatch, rejection, unavailable and disabled backends, and default fallback. Exercise changed verifier boundaries and compare outputs and supported gradients with the default implementation. Include calls with omitted optional arguments to check defaults.

Set `FLA_DISABLE_BACKEND_DISPATCH=1` before importing FLA for the default path. Use explicit backend flags for routing tests, and fresh processes for import behavior. When changing registration, check shared backend ownership, import order, and existing public imports.
