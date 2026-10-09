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
from typing import Any, ClassVar, TypeVar

import torch

from fla.utils import find_spec_cached

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable)
B = TypeVar('B', bound='BaseBackend')

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
    env_var: ClassVar[str | None] = None
    default_enable: ClassVar[bool] = True
    priority: ClassVar[int] = 5

    @classmethod
    def is_available(cls) -> bool:
        if cls.package_name is None:
            return True
        return find_spec_cached(cls.package_name) is not None

    @classmethod
    def is_enabled(cls) -> bool:
        if cls.env_var is None:
            return True
        if os.environ.get(f"FLA_{cls.backend_type.upper()}", "0") != "0":
            return True
        default_value = "1" if cls.default_enable else "0"
        return os.environ.get(cls.env_var, default_value) != "0"

    @classmethod
    @cache
    def can_use(cls) -> bool:
        return cls.is_available() and cls.is_enabled()

    def verify(self, func_name: str, *args, **kwargs) -> tuple[bool, str | None]:
        """Check if backend can handle the function call."""
        verifier_name = f"{func_name}_verifier"
        verifier = getattr(self, verifier_name, None)
        if verifier is None:
            return True, None

        try:
            return verifier(*args, **kwargs)
        except Exception as e:
            return False, str(e)


class BackendRegistry:
    """Backends owned by one operation directory."""

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

    def dispatch(self, func: F) -> F:
        """Select the first eligible backend, falling back to the decorated function."""
        if _DISPATCH_DISABLED:
            return func
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            backends_list = self._get_sorted_backends()

            for be in backends_list:
                # avoid be.can_use(): its @cache wrapper breaks torch.compile tracing.
                if not (be.is_available() and be.is_enabled()):
                    continue

                can_use, reason = be.verify(func_name, *args, **kwargs)
                if not can_use:
                    fail_key = f"{self.operation_name}:{func_name}:{be.backend_type}:fail"
                    if fail_key not in self._logged:
                        self._logged.add(fail_key)
                        logger.info(
                            f"[FLA Backend] {self.operation_name}.{func_name} -> {be.backend_type} "
                            f"rejected: {reason}"
                        )
                    continue

                impl = getattr(be, func_name, None)
                if impl is None:
                    continue

                result = impl(*args, **kwargs)

                log_key = f"{self.operation_name}:{func_name}:{be.backend_type}"
                if log_key not in self._logged:
                    self._logged.add(log_key)
                    logger.info(f"[FLA Backend] {self.operation_name}.{func_name} -> {be.backend_type}")

                return result

            return func(*args, **kwargs)

        # runtime backend selection must stay outside torch.compile graphs.
        wrapper = torch.compiler.disable(wrapper)

        return wrapper


_registries: dict[str, BackendRegistry] = {}


def _registry_for(operation: str) -> BackendRegistry:
    if operation not in _registries:
        _registries[operation] = BackendRegistry(operation)
    return _registries[operation]


def register_backend(operation: str) -> Callable[[type[B]], type[B]]:
    """Register a backend instance for an operation and return its class."""
    def decorator(backend_class: type[B]) -> type[B]:
        _registry_for(operation).register(backend_class())
        return backend_class

    return decorator


def _resolve_registry(operation: str, *, allow_unknown: bool = False) -> BackendRegistry:
    if operation == 'modules':
        module_path = 'fla.modules.backends'
    elif operation == 'modules.conv':
        module_path = 'fla.modules.causal_conv.backends'
    elif operation.startswith('modules.norm.'):
        module_path = 'fla.modules.norm.triton_ascend'
    elif operation.startswith('modules.'):
        module_path = f'fla.{operation}.backends'
    else:
        module_path = f'fla.ops.{operation}.backends'
    # an existing registry does not guarantee that all of the owner's backends are loaded.
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if not allow_unknown or (error.name != module_path and not module_path.startswith(f'{error.name}.')):
            raise
    return _registry_for(operation)


def dispatch(operation: str) -> Callable[[F], F]:
    """Load an operation's backends and select from the shared registry."""
    return _resolve_registry(operation).dispatch


__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch', 'register_backend']
