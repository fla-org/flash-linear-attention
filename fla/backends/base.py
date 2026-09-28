# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend availability and call verification."""

import os
from functools import cache
from typing import ClassVar

from fla.utils import find_spec_cached


class BaseBackend:
    """Base class for operation-specific backends.

    Attributes:
        backend_type (str, Optional):
            Identifier for the backend type, used to distinguish different backend implementations.
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
