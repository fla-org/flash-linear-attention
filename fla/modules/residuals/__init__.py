# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from inspect import isabstract

from fla.modules.residuals.attnres import AttentionResidual
from fla.modules.residuals.base import BaseResidual
from fla.modules.residuals.mhc import ManifoldHyperConnection
from fla.modules.residuals.standard import StandardResidual

__all__ = [
    'AttentionResidual',
    'BaseResidual',
    'ManifoldHyperConnection',
    'StandardResidual',
    'get_residual_class',
    'register_residual',
]


_RESIDUAL_CLASSES: dict[str, type[BaseResidual]] = {
    'standard': StandardResidual,
    'attnres': AttentionResidual,
    'mhc': ManifoldHyperConnection,
}


def register_residual(name: str, residual_cls: type[BaseResidual]) -> None:
    """Register a residual before constructing or loading its model in each process.

    Custom classes implement `pre`, `post`, and `reset_parameters`, including initialization of owned children.
    Checkpoints store the registered name and parameters; they do not store the class implementation.
    """
    if not isinstance(name, str) or not name or any(char.isspace() for char in name):
        raise ValueError('Residual name must be a non-empty string without whitespace')
    if not isinstance(residual_cls, type) or not issubclass(residual_cls, BaseResidual) or isabstract(residual_cls):
        raise TypeError('residual_cls must be a concrete BaseResidual subclass')
    if name in _RESIDUAL_CLASSES:
        raise ValueError(f'Residual {name!r} is already registered')
    _RESIDUAL_CLASSES[name] = residual_cls


def get_residual_class(name: str) -> type[BaseResidual]:
    try:
        return _RESIDUAL_CLASSES[name]
    except KeyError:
        raise ValueError(f'Unknown residual_mode: {name!r}. Call register_residual before loading the model.') from None
