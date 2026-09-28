# Backend dispatch

Each operation owns its registry and exports a bound `dispatch` decorator from its backend package. Shared code never resolves operation names into import paths.

| Location                                   | Responsibility                                                              |
| ------------------------------------------ | --------------------------------------------------------------------------- |
| `fla/backends/base.py`                     | Backend availability, enablement, and call verification                     |
| `fla/backends/registry.py`                 | Registration, priority ordering, selection, and fallback                    |
| `fla/ops/<operation>/backends/__init__.py` | Declare and register that operation's adapters; export its dispatcher       |
| `fla/modules/backends/__init__.py`         | Own the existing modules registry and its dispatcher                        |
| `fla/ops/backends/__init__.py`             | Temporary compatibility for the former global registry and string-based API |

Register adapters where they belong:

```python
from fla.backends import BackendRegistry
from fla.ops.kda.backends.flash_kda import FlashKDABackend

kda_registry = BackendRegistry('kda')
kda_registry.register(FlashKDABackend())
dispatch = kda_registry.dispatch
```

KDA entry points import `dispatch` from `fla.ops.kda.backends` and use bare `@dispatch`. The decorator is already bound to `kda_registry`, so callers supply no operation name.

Registration happens when the owning package is imported. `register()` records the adapter; it performs no availability probes or kernel execution. Optional implementation imports remain inside adapter methods where already supported; the existing optional Gluon import guard remains in AttnRes. A call checks availability, enablement, and its verifier in ascending priority order, preserving registration order for ties. If no backend handles the call, the decorated function runs. Registries are independent objects, even when their diagnostic names match.

Set `FLA_DISABLE_BACKEND_DISPATCH=1` before importing FLA to make decorators return their original functions. Backend-specific environment variables retain their existing meanings; see [ENVs.md](../../ENVs.md#operator-backend-dispatch).

The modules registry retains its existing eager adapter import, including the NPU GRPO compilation workaround. Module backend splitting and kernel renaming are separate work.

## Import compatibility

`fla.ops.backends` emits `DeprecationWarning`. Its legacy `BackendRegistry(name)` factory returns the directory-owned registry, and its `dispatch(name)` delegates to that registry. Import `BaseBackend` and `BackendRegistry` from `fla.backends`; import `dispatch` from the owning backend package. The former registry class's private internals and subclassing are not supported by the compatibility factory.

A bound dispatcher also accepts its own old operation string, such as `@dispatch('kda')`, with a deprecation warning. For another operation, import that operation's dispatcher. These compatibility forms will be removed in the next release after 0.6.0.

For module consumers, long-term API compatibility is guaranteed only for public exports from `fla.modules`. Backend and module implementation paths are internal.
