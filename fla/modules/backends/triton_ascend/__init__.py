# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deprecated platform-wide module backend import path."""

import warnings

warnings.warn(
    "fla.modules.backends.triton_ascend is deprecated and will be removed in the next release after 0.6.0. "
    "Use fla.modules.backends.<operation>.triton_ascend for internal implementation access. "
    "Only public exports from fla.modules have long-term module API compatibility.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    if name != 'TritonAscendBackend':
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from fla.modules.backends._legacy import TritonAscendBackend
    return TritonAscendBackend
