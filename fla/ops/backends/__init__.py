# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deprecated import path for the shared backend dispatcher."""

import warnings

from fla.backends import BackendRegistry, BaseBackend, dispatch

warnings.warn(
    "fla.ops.backends is deprecated and will be removed in the next release after 0.6.0. "
    "Import BackendRegistry, BaseBackend, and dispatch from fla.backends instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch']
