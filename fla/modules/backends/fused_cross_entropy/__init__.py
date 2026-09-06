# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for fused_cross_entropy."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendFusedCrossEntropyBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        logit_softcapping: float = None,
        ignore_index=-100,
        inplace_backward: bool = False,
        process_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.modules.backends.fused_cross_entropy.triton_ascend import cross_entropy_loss
        return cross_entropy_loss(
            logits=logits,
            target=target,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            lse_square_scale=lse_square_scale,
            logit_softcapping=logit_softcapping,
            ignore_index=ignore_index,
            inplace_backward=inplace_backward,
            process_group=process_group,
        )


fused_cross_entropy_registry = BackendRegistry("modules.fused_cross_entropy")
fused_cross_entropy_registry.register(TritonAscendFusedCrossEntropyBackend())
