"""Generic backend dispatch system for FLA operations."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import Any, ClassVar, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable)


class BaseBackend:
    """Base class for operation-specific backends."""

    backend_type: ClassVar[str] = "base"
    package_name: ClassVar[str | None] = None
    env_var: ClassVar[str | None] = None

    @classmethod
    def is_available(cls) -> bool:
        if cls.package_name is None:
            return True
        return find_spec(cls.package_name) is not None

    @classmethod
    def is_enabled(cls) -> bool:
        if cls.env_var is None:
            return True
        return os.environ.get(cls.env_var, "1") != "0"

    @classmethod
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
    """Per-operation backend registry."""

    _registries: ClassVar[dict[str, BackendRegistry]] = {}
    _initialized: ClassVar[set[str]] = set()
    _init_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self._backends: dict[str, BaseBackend] = {}
        self._active: BaseBackend | None = None
        self._lock = threading.RLock()
        self._logged: set[str] = set()
        BackendRegistry._registries[operation_name] = self

    def register(self, backend: BaseBackend) -> None:
        """Register a backend."""
        with self._lock:
            self._backends[backend.backend_type] = backend
            if self._active is None and backend.can_use():
                self._active = backend

    def get_active(self) -> BaseBackend | None:
        """Get active backend."""
        return self._active

    @classmethod
    def ensure_initialized(cls, operation: str) -> None:
        """Lazy-load backends on first use."""
        if operation in cls._initialized:
            return

        with cls._init_lock:
            if operation in cls._initialized:
                return

            # Import backend module to trigger registration
            with contextlib.suppress(ImportError):
                __import__(f'fla.ops.{operation}.backends', fromlist=[''])

            cls._initialized.add(operation)


def dispatch(operation: str):
    """Dispatch decorator with verifier support.

    Iterates through all registered backends and selects the first one
    that passes the verifier for the given function call.
    """
    def decorator(func: F) -> F:
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Lazy initialization of backends
            BackendRegistry.ensure_initialized(operation)

            registry = BackendRegistry._registries.get(operation)
            if registry is None:
                return func(*args, **kwargs)

            # Iterate through all registered backends to find one that can handle this call
            # Note: _backends is only modified during registration, which happens at import time
            # After initialization, it's safe to read without lock
            backends_list = registry._backends.values()

            for be in backends_list:
                if not be.can_use():
                    continue

                can_use, _ = be.verify(func_name, *args, **kwargs)
                if not can_use:
                    continue

                impl = getattr(be, func_name, None)
                if impl is None:
                    continue

                result = impl(*args, **kwargs)

                log_key = f"{operation}:{func_name}:{be.backend_type}"
                if log_key not in registry._logged:
                    registry._logged.add(log_key)
                    logger.info(f"[FLA Backend] {operation}.{func_name} -> {be.backend_type}")

                return result

            # No backend can handle this call, use default implementation
            return func(*args, **kwargs)

        return wrapper
    return decorator


__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch']
