# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.modules.residuals.attnres import AttentionResidual
from fla.modules.residuals.base import BaseResidual
from fla.modules.residuals.standard import StandardResidual

__all__ = [
    'AttentionResidual',
    'BaseResidual',
    'StandardResidual',
    'create_residual',
]


def create_residual(
    hidden_size: int,
    norm_eps: float = 1e-6,
    fuse_norm: bool = True,
    block_size: int | None = None,
) -> BaseResidual:
    """Create standard residuals for `block_size=None`, or attention residuals with the given block size."""
    if block_size is None:
        return StandardResidual(hidden_size=hidden_size, norm_eps=norm_eps, fuse_norm=fuse_norm, fuse_residual=True)
    return AttentionResidual(hidden_size=hidden_size, block_size=block_size, norm_eps=norm_eps, fuse_norm=fuse_norm)
