# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Operator specifications and discovery of definitions owned by benchmark objects."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.utils.ops.inputs import TensorSpec


@dataclass
class OpConfig:
    """Registry entry describing how to benchmark a single op.

    Args:
        name (str):
            Display and registry name, such as `chunk_gla`.
        import_path (str):
            Python module path, such as `fla.ops.gla`.
        inputs (dict[str, TensorSpec]):
            Mapping from function argument names to tensor specs.
        func_name (str, Optional):
            Function attribute name to import when it differs from `name`.
            Default: `None`.
        extra_kwargs (dict[str, Any], Optional):
            Constant keyword arguments passed to the op. Default: `{}`.
        output_is_tuple (bool):
            Whether the op returns a tuple whose first item is the tensor used for `.backward()`. Default: `True`.
        skip_backward (bool):
            Whether to skip forward-backward benchmark mode. Default: `False`.
        post_init (Callable, Optional):
            Callback invoked as `post_init(inputs, B=B, T=T, H=H, D=D, **kw)` for custom input mutation. Default: `None`.
        category (str):
            Grouping label used in reports. Default: `''`.
        dim_constraints (dict, Optional):
            Shape constraints, such as `{'D': [64, 128]}`. Shapes that do not match are skipped. Default: `None`.
        default_shapes (dict[str, dict[str, int]], Optional):
            Per-op shape configs used instead of the global `SHAPE_CONFIGS`.
            This is useful when the op's shape semantics differ from `B/T/H/D`,
            for example when AttnRes uses an extra `L` residual-source axis.
            Default: `None`.
        test_file (str, Optional):
            Path (relative to repo root) to the op's pytest file,
            used as the frozen correctness gate by `benchmarks/verify.py`.
            When None, the gate is derived from `import_path`'s last segment (e.g. `fla.ops.gla` -> `tests/ops/test_gla.py`).
            Set this only when the derived path is wrong. Default: `None`.
        backend_env (dict[str, str], Optional):
            Maps a `--backend <name>` value to the environment variable that enables that backend's
            dispatch (e.g. `{'gluon': 'FLA_ATTNRES_GLUON'}`). The runner sets it before launching the
            op; ops selected purely by a runtime verifier need no entry. Default: `None`.
    """
    name: str
    import_path: str
    inputs: dict[str, TensorSpec]
    func_name: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    output_is_tuple: bool = True
    skip_backward: bool = False
    post_init: Callable | None = None
    category: str = ''
    dim_constraints: dict | None = None
    default_shapes: dict[str, dict[str, int]] | None = None
    test_file: str | None = None
    backend_env: dict[str, str] | None = None


SHAPE_CONFIGS = {
    'B1_T8192_H96_D128':  {'B': 1,  'T': 8192,  'H': 96, 'D': 128},
    'B2_T16384_H16_D128': {'B': 2,  'T': 16384, 'H': 16, 'D': 128},
    'B4_T2048_H16_D128':  {'B': 4,  'T': 2048,  'H': 16, 'D': 128},
    'B4_T4096_H64_D128':  {'B': 4,  'T': 4096,  'H': 64, 'D': 128},
    'B8_T2048_H32_D256':  {'B': 8,  'T': 2048,  'H': 32, 'D': 256},
    'B8_T1024_H8_D64':    {'B': 8,  'T': 1024,  'H': 8,  'D': 64},
}


@cache
def _get_registry() -> dict[str, OpConfig]:
    registry = {}
    for path in sorted((Path(__file__).resolve().parents[2] / 'ops').glob('*.py')):
        if path.stem == '__init__':
            continue
        module = importlib.import_module(f'benchmarks.ops.{path.stem}')
        for config in module.OPERATORS:
            if config.name in registry:
                raise ValueError(f'Duplicate operator: {config.name}')
            registry[config.name] = config
    return registry


def get_op(name: str) -> OpConfig:
    return _get_registry()[name]


def list_ops() -> list[str]:
    return sorted(_get_registry())
