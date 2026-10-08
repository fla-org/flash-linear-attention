# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.modules.fused_linear_cross_entropy.module import (
    FusedLinearCrossEntropyLoss,
    LinearLossParallel,
)
from fla.modules.fused_linear_cross_entropy.ops import (
    MAX_FUSED_SIZE,
    STATIC_WARPS,
    FusedLinearCrossEntropyFunction,
    elementwise_mul_kernel,
    fused_linear_cross_entropy_backward,
    fused_linear_cross_entropy_bwd,
    fused_linear_cross_entropy_forward,
    fused_linear_cross_entropy_fwd,
    fused_linear_cross_entropy_loss,
    logsumexp_fwd,
    logsumexp_fwd_kernel,
)

__all__ = [
    'MAX_FUSED_SIZE',
    'STATIC_WARPS',
    'FusedLinearCrossEntropyFunction',
    'FusedLinearCrossEntropyLoss',
    'LinearLossParallel',
    'elementwise_mul_kernel',
    'fused_linear_cross_entropy_backward',
    'fused_linear_cross_entropy_bwd',
    'fused_linear_cross_entropy_forward',
    'fused_linear_cross_entropy_fwd',
    'fused_linear_cross_entropy_loss',
    'logsumexp_fwd',
    'logsumexp_fwd_kernel',
]
