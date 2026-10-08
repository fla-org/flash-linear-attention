# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register_backend


@register_backend('modules.grpo')
class TritonAscendBackend(BaseBackend):
    """Ascend implementation of this operation."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
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
    ):
        from fla.modules.grpo.backends.triton_ascend.ops import fused_grpo_loss_npu
        return fused_grpo_loss_npu(
            logits,
            ref_logp,
            input_ids,
            advantages,
            beta,
            completion_mask,
            save_kl,
            inplace,
        )


__all__ = ['TritonAscendBackend']
