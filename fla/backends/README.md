# Backend dispatch

`fla.backends` provides the shared `BaseBackend`, `BackendRegistry`, and `dispatch` machinery used by operators and modules.

```python
from fla.backends import BackendRegistry, BaseBackend, dispatch
```

Backend implementations and registration stay in `fla.ops.<operation>.backends` and `fla.modules.backends`. Operation keys, selection priorities, verifiers, fallback behavior, and environment variables keep their existing meanings.

`fla.ops.backends` re-exports the same objects and emits `DeprecationWarning`. This compatibility path will be removed in the next release after 0.6.0; update dispatcher imports to `fla.backends`.
