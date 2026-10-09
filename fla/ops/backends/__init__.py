# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

from fla.backends import BackendRegistry as _BackendRegistry
from fla.backends import BaseBackend, _registries, _resolve_registry

warnings.warn(
    "fla.ops.backends is deprecated and will be removed in the next release after 0.6.0. "
    "Import BackendRegistry, BaseBackend and dispatch from fla.backends instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BackendRegistry:
    """Return the shared registry for an operation."""

    _registries = _registries

    def __new__(cls, operation_name: str) -> _BackendRegistry:
        return _resolve_registry(operation_name, allow_unknown=True)

    @classmethod
    def ensure_initialized(cls, operation: str) -> None:
        cls(operation)


def dispatch(operation: str):
    return BackendRegistry(operation).dispatch


__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch']
