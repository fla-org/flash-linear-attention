# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.modules import _deprecated_getattr
from fla.modules.fused_cross_entropy.ops import FusedCrossEntropyLoss, cross_entropy_loss

__all__ = ['FusedCrossEntropyLoss', 'cross_entropy_loss']


if TritonAscendBackend.is_available():
    from fla.modules.fused_cross_entropy import triton_ascend  # noqa: F401


__getattr__ = _deprecated_getattr(
    module_name=__name__,
    targets=('fla.modules.fused_cross_entropy.ops',),
    aliases={
        'fused_cross_entropy_forward': 'cross_entropy_fwd',
        'CrossEntropyLossFunction': 'FusedCrossEntropyFunction',
    },
)
