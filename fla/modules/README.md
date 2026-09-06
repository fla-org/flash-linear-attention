# Module imports and backend development

## Public imports

Only the public symbols exported by `fla.modules` have long-term module API compatibility. Import components through this namespace:

```python
from fla.modules import FusedCrossEntropyLoss, RMSNorm, RotaryEmbedding, ShortConvolution
```

The public export list is `fla.modules.__all__`. Submodule paths such as `fla.modules.layernorm` and all backend paths are implementation details and have no long-term import compatibility guarantee. This policy concerns module imports; model configuration and checkpoint compatibility remain separate contracts.

## Deprecated backend imports

The following internal paths are deprecated in the 0.6.0 development line and will be removed in the next release after 0.6.0. They emit `DeprecationWarning` and forward to the current implementation during the transition. Normal imports from `fla.modules` do not emit these warnings.

| Deprecated path or usage                                           | Replacement                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `fla.modules.backends.dispatch`                                    | `fla.backends.dispatch`                                        |
| `fla.ops.backends`                                                 | `fla.backends`                                                 |
| `@dispatch('modules')`                                             | `@dispatch('modules.<operation>')`                              |
| `fla.modules.backends.modules_registry`                            | The registry in `fla.modules.backends.<operation>`              |
| `fla.modules.backends.triton_ascend.<operation>`                    | `fla.modules.backends.<operation>.triton_ascend`                 |
| `fla.modules.backends.triton_ascend.TritonAscendBackend`             | The backend class in `fla.modules.backends.<operation>`          |

Backend implementation function names omit the redundant `_npu` suffix, and backend autograd classes omit the `NPU` suffix. For example, the internal `fla.modules.backends.triton_ascend.layernorm.layer_norm_fwd_npu` becomes `fla.modules.backends.layernorm.triton_ascend.layer_norm_fwd`. These replacement paths are also internal APIs.

## Backend organization

The shared dispatch machinery lives in `fla.backends`. Module backends are organized by operation:

```text
fla/modules/
├── layernorm.py
└── backends/
    ├── layernorm/
    │   ├── __init__.py       # registry and lazy backend adapter
    │   └── triton_ascend.py  # kernels and launchers
    └── rotary/
        ├── __init__.py
        └── triton_ascend.py
```

Default entry points import `dispatch` from `fla.backends` and use an operation-specific key such as `modules.layernorm`. The first call imports the corresponding registry. Each registry owns its backend priorities and enablement policy. Backend adapters import their kernel implementation only when selected; importing the public module API does not load optional backend implementations or modify global compiler functions.

This organization follows the dispatch-based approach in the [NPU roadmap](https://github.com/fla-org/flash-linear-attention/issues/942). The production backend remains `triton_ascend`, with the same availability checks, numerical implementations, and compiler settings. GRPO's old-log-probability loss stays eager on NPU through a local compile setting. Future `token_shift` or `tilelang_ascend` implementations can register through the same machinery without adding methods to a modules-wide adapter.

When moving an implementation, update its registration, callers, dependent-test discovery, split-package contents, and CI path filters together. Keep forward/backward numerical tests and their tolerances unchanged.
