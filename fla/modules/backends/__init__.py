# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Operation-specific module backends; implementation imports are internal."""

import warnings

__all__ = ['dispatch', 'modules_registry']


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    replacement = 'fla.backends.dispatch' if name == 'dispatch' else 'an operation-specific backend registry'
    warnings.warn(
        f"{__name__}.{name} is deprecated and will be removed in the next release after 0.6.0. "
        f"Use {replacement} instead. Only public exports from fla.modules have long-term module API compatibility.",
        DeprecationWarning,
        stacklevel=2,
    )
    if name == 'dispatch':
        from fla.backends import dispatch
        return dispatch
    from fla.modules.backends._legacy import modules_registry
    return modules_registry
