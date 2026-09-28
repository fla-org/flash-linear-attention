# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compatibility for the former global registry and string-based dispatcher."""

import importlib
import warnings

from fla.backends import BackendRegistry as _BackendRegistry
from fla.backends import BaseBackend

warnings.warn(
    "fla.ops.backends is deprecated and will be removed in the next release after 0.6.0. "
    "Import BackendRegistry and BaseBackend from fla.backends, "
    "and dispatch from the operation's backends package instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BackendRegistry:
    """Compatibility factory returning the registry owned by an operation's directory."""

    _registries: dict[str, _BackendRegistry] = {}

    def __new__(cls, operation_name: str) -> _BackendRegistry:
        if operation_name not in cls._registries:
            module_path = 'fla.modules.backends' if operation_name == 'modules' else f'fla.ops.{operation_name}.backends'
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as error:
                if error.name != module_path and not module_path.startswith(f'{error.name}.'):
                    raise
                registry = _BackendRegistry(operation_name)
            else:
                registry = module.dispatch.__self__
            cls._registries[operation_name] = registry
        return cls._registries[operation_name]

    @classmethod
    def ensure_initialized(cls, operation: str) -> None:
        cls(operation)


def dispatch(operation: str):
    """Resolve an old operation key and return its directory's dispatcher."""
    return BackendRegistry(operation).dispatch


__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch']
