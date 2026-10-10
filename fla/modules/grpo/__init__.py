# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import TritonAscendBackend
from fla.modules._compat import _deprecated_getattr
from fla.modules.grpo.ops import fused_grpo_loss, grpo_loss_with_old_logps

__all__ = ['fused_grpo_loss', 'grpo_loss_with_old_logps']


if TritonAscendBackend.is_available():
    from fla.modules.grpo import triton_ascend  # noqa: F401


__getattr__ = _deprecated_getattr(
    module_name=__name__,
    targets=('fla.modules.grpo.ops',),
)
