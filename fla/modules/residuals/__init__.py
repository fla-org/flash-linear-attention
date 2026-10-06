# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.modules.residuals.attnres import AttentionResidual
from fla.modules.residuals.base import BaseResidual
from fla.modules.residuals.mhc import ManifoldHyperConnection
from fla.modules.residuals.registry import get_residual_class, register_residual
from fla.modules.residuals.standard import StandardResidual

__all__ = [
    'AttentionResidual',
    'BaseResidual',
    'ManifoldHyperConnection',
    'StandardResidual',
    'get_residual_class',
    'register_residual',
]
