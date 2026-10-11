# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration and runtime selection."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from functools import cache, wraps
from inspect import unwrap
from typing import Any, ClassVar, TypeVar

import torch

from fla.utils import find_spec_cached, has_usable_nvcc

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable)

_DISPATCH_DISABLED = os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
if _DISPATCH_DISABLED:
    logger.info("[FLA Backend] FLA_DISABLE_BACKEND_DISPATCH=1 — all dispatch bypassed")


class BaseBackend:
    """Base class for operation-specific backends.

    Attributes:
        backend_type (str, Optional):
            Identifier for the backend type, used to distinguish different backend implementations.
            `FLA_<BACKEND_TYPE>` enables all backends of this type, overriding individual switches set to `0`.
            Default: `"base"`.
        package_name (str, Optional):
            Name of the external package required by the backend.
            `None` indicates no external dependency. Default: `None`.
        env_var (str, Optional):
            Environment variable name that controls whether the backend is enabled.
            `None` means always enabled. Default: `None`.
        default_enable (bool, Optional):
            Whether the backend is enabled by default when `env_var` is not set.
            Set to `False` to require explicit user opt-in. Default: `True`.
        priority (int, Optional):
            Backend priority. Lower values indicate higher priority. Default: 5.
    """

    backend_type: ClassVar[str] = "base"
    package_name: ClassVar[str | None] = None
    env_var: str | None = None
    default_enable: ClassVar[bool] = True
    priority: ClassVar[int] = 5
    implementation: Callable | None = None
    verifier: Callable | None = None

    def __init__(self, *, env_var: str | None = None):
        if env_var is not None:
            self.env_var = env_var

    def get_implementation(self, function_name: str) -> Callable | None:
        if self.implementation is not None:
            return self.implementation
        return getattr(self, function_name, None)

    @classmethod
    def is_available(cls) -> bool:
        if cls.package_name is None:
            return True
        return find_spec_cached(cls.package_name) is not None

    def is_enabled(self) -> bool:
        if self.env_var is None:
            return True
        if os.environ.get(f"FLA_{self.backend_type.upper()}", "0") != "0":
            return True
        default_value = "1" if self.default_enable else "0"
        return os.environ.get(self.env_var, default_value) != "0"

    @classmethod
    @cache
    def can_use(cls) -> bool:
        return cls.is_available() and cls().is_enabled()

    def verify(self, function_name: str, *args, **kwargs) -> tuple[bool, str | None]:
        """Check if backend can handle the function call."""
        verifier_name = f"{function_name}_verifier"
        verifier = self.verifier or getattr(self, verifier_name, None)
        if verifier is None:
            return True, None

        try:
            return verifier(*args, **kwargs)
        except Exception as error:
            return False, str(error)


class TritonAscendBackend(BaseBackend):
    """Shared availability and selection policy for Ascend implementations."""

    backend_type = 'triton_ascend'
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU


class TileLangBackend(BaseBackend):
    """Shared availability and selection policy for TileLang implementations."""

    backend_type = 'tilelang'
    package_name = 'tilelang'
    env_var = 'FLA_TILELANG'

    @classmethod
    def is_available(cls) -> bool:
        return super().is_available() and has_usable_nvcc()


class GluonBackend(BaseBackend):
    """NVIDIA GPU backend using Gluon kernels."""

    backend_type = 'gluon'
    package_name = 'triton.experimental.gluon'
    env_var = 'FLA_GLUON'
    default_enable = False

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NVIDIA
        return IS_NVIDIA and super().is_available()


class BackendRegistry:
    """Ordered backend candidates for an operation or entry point."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self._backends: dict[str, BaseBackend] = {}
        self._logged: set[str] = set()

    def register(self, backend: BaseBackend) -> None:
        """Register or replace a backend by its type."""
        self._backends[backend.backend_type] = backend

    def _get_sorted_backends(self) -> list[BaseBackend]:
        """Lower priorities come first; ties retain registration order."""
        return sorted(self._backends.values(), key=lambda backend: backend.priority)

    def get_active(self) -> BaseBackend | None:
        """Return the first available and enabled backend, before call verification."""
        for backend in self._get_sorted_backends():
            if backend.is_available() and backend.is_enabled():
                return backend
        return None

    def dispatch(self, default_function: F) -> F:
        """Select the first eligible backend, falling back to the decorated function."""
        if _DISPATCH_DISABLED:
            return default_function
        function_name = default_function.__name__

        @wraps(default_function)
        def wrapper(*args, **kwargs) -> Any:
            for backend in self._get_sorted_backends():
                # avoid backend.can_use(): its @cache wrapper breaks torch.compile tracing.
                if not (backend.is_available() and backend.is_enabled()):
                    continue

                accepted, reason = backend.verify(function_name, *args, **kwargs)
                if not accepted:
                    rejection_key = f"{self.operation_name}:{function_name}:{backend.backend_type}:fail"
                    if rejection_key not in self._logged:
                        self._logged.add(rejection_key)
                        logger.info(
                            f"[FLA Backend] {self.operation_name}.{function_name} -> {backend.backend_type} "
                            f"rejected: {reason}"
                        )
                    continue

                implementation = backend.get_implementation(function_name)
                if implementation is None:
                    continue

                result = implementation(*args, **kwargs)

                log_key = f"{self.operation_name}:{function_name}:{backend.backend_type}"
                if log_key not in self._logged:
                    self._logged.add(log_key)
                    logger.info(f"[FLA Backend] {self.operation_name}.{function_name} -> {backend.backend_type}")

                return result

            return default_function(*args, **kwargs)

        # runtime backend selection must stay outside torch.compile graphs.
        return torch.compiler.disable(wrapper)


_operation_registries: dict[str, BackendRegistry] = {}
_function_registries: dict[Callable, BackendRegistry] = {}


def _get_operation_registry(operation: str) -> BackendRegistry:
    if operation not in _operation_registries:
        _operation_registries[operation] = BackendRegistry(operation)
    return _operation_registries[operation]


def _load_operation_registry(operation: str) -> BackendRegistry:
    module_path = f'fla.ops.{operation}.backends'
    # an existing registry does not guarantee that all of the owner's backends are loaded.
    importlib.import_module(module_path)
    return _get_operation_registry(operation=operation)


def register(
    entry_point: Callable | str,
    *,
    backend: type[BaseBackend] | None = None,
    verifier: Callable | None = None,
    env_var: str | None = None,
) -> Callable[[F], F]:
    """Register an implementation or adapter for a dispatched function or operation name."""
    def decorator(implementation: F) -> F:
        if _DISPATCH_DISABLED and not isinstance(entry_point, str):
            return implementation
        if isinstance(implementation, type):
            registered_backend = implementation()
            if backend is not None and backend.backend_type != registered_backend.backend_type:
                raise ValueError('The registration backend must match the adapter backend_type')
        else:
            registered_backend = backend() if backend is not None else BaseBackend()
            registered_backend.implementation = implementation
        registered_backend.verifier = verifier
        if env_var is not None:
            registered_backend.env_var = env_var
        if isinstance(entry_point, str):
            registry = _get_operation_registry(operation=entry_point)
        else:
            registry = _function_registries[unwrap(entry_point)]
        registry.register(registered_backend)
        return implementation

    return decorator


def dispatch(entry_point: F | str) -> F | Callable[[F], F]:
    """Allow backends to register replacements while retaining the original fallback."""
    if isinstance(entry_point, str):
        return _load_operation_registry(operation=entry_point).dispatch
    if _DISPATCH_DISABLED:
        return entry_point
    registry = BackendRegistry(entry_point.__module__)
    _function_registries[unwrap(entry_point)] = registry

    return registry.dispatch(entry_point)


__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch', 'register']
