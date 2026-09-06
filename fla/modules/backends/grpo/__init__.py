# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for grpo."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendGRPOBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def fused_grpo_loss(
        self,
        logits,
        ref_logp,
        input_ids,
        advantages,
        beta=0.1,
        completion_mask=None,
        save_kl=False,
        inplace=False,
    ) -> torch.Tensor:
        from fla.modules.backends.grpo.triton_ascend import fused_grpo_loss
        return fused_grpo_loss(
            logits=logits,
            ref_logp=ref_logp,
            input_ids=input_ids,
            advantages=advantages,
            beta=beta,
            completion_mask=completion_mask,
            save_kl=save_kl,
            inplace=inplace,
        )


grpo_registry = BackendRegistry("modules.grpo")
grpo_registry.register(TritonAscendGRPOBackend())
