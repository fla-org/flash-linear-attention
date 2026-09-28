# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration and runtime selection."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

import torch

if TYPE_CHECKING:
    from fla.backends.base import BaseBackend

logger = logging.getLogger(__name__)
F = TypeVar('F', bound=Callable)

_DISPATCH_DISABLED = os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
if _DISPATCH_DISABLED:
    logger.info("[FLA Backend] FLA_DISABLE_BACKEND_DISPATCH=1 — all dispatch bypassed")


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

    def dispatch(self, func: F | str) -> F | Callable[[F], F]:
        """Select the first eligible backend, falling back to the decorated function."""
        if isinstance(func, str):
            if func != self.operation_name:
                raise ValueError(f'This dispatcher belongs to {self.operation_name!r}, not {func!r}.')
            warnings.warn(
                'Operation strings are deprecated and will be removed in the next release after 0.6.0. '
                'Import dispatch from the owning backends package and use @dispatch without arguments.',
                DeprecationWarning,
                stacklevel=2,
            )
            return self.dispatch
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
