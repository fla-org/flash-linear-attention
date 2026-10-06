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

_RESIDUAL_CLASSES: dict[str, type[BaseResidual]] = {}


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError('Residual name must be a string')
    if not name or any(char.isspace() for char in name):
        raise ValueError('Residual name must be non-empty and contain no whitespace')


def register_residual(name: str, residual_cls: type[BaseResidual]) -> None:
    """Register a concrete residual class under a unique, case-sensitive config name.

    Args:
        name (str):
            Value used by `residual_mode`. A project-qualified name such as `my_project.scaled` avoids collisions.
        residual_cls (type[BaseResidual]):
            Concrete residual implementation. Its constructor must accept the model's residual arguments
            and the options supplied in `residual_kwargs`. Custom parameter initialization must follow
            `BaseResidual.reset_parameters`; constructor-only initial values may not survive model loading.

    Registration is local to the Python process. Import and register custom implementations in every
    process before constructing or loading a model; checkpoints store the name and weights, not the class code.
    Existing names cannot be overwritten, including by registering the same class again.
    """
    _validate_name(name)
    if not isinstance(residual_cls, type) or not issubclass(residual_cls, BaseResidual):
        raise TypeError('residual_cls must be a BaseResidual subclass')
    if isabstract(residual_cls):
        raise TypeError('residual_cls must be concrete; implement all abstract methods')
    if name in _RESIDUAL_CLASSES:
        raise ValueError(f'Residual {name!r} is already registered')
    _RESIDUAL_CLASSES[name] = residual_cls


def get_residual_class(name: str) -> type[BaseResidual]:
    """Resolve a registered residual name without constructing a module or importing user code."""
    _validate_name(name)
    if name not in _RESIDUAL_CLASSES:
        raise ValueError(
            f'Unknown residual_mode: {name!r}. Registered names: {sorted(_RESIDUAL_CLASSES)}. '
            'Import your custom implementation and call register_residual before constructing or loading the model.'
        )
    return _RESIDUAL_CLASSES[name]


register_residual('standard', StandardResidual)
register_residual('attnres', AttentionResidual)
register_residual('mhc', ManifoldHyperConnection)
