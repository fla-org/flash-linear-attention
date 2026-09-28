# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared backend contracts and registry mechanics."""

from fla.backends.base import BaseBackend
from fla.backends.registry import BackendRegistry

__all__ = ['BackendRegistry', 'BaseBackend']
