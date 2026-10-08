# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from fla.backends import BaseBackend, register_backend


@register_backend('modules.fused_cross_entropy')
class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend using triton-ascend kernels."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def cross_entropy_loss(
        self,
        logits,
        target,
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        logit_softcapping=None,
        ignore_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        from fla.modules.fused_cross_entropy.backends.triton_ascend.ops import (
            cross_entropy_loss_npu,
        )
        return cross_entropy_loss_npu(
            logits,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            logit_softcapping,
            ignore_index,
            inplace_backward,
            process_group,
        )
