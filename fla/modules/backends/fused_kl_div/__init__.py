# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for fused_kl_div."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendFusedKLDivBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def fused_kl_div_forward(
        self,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
        reduction: str = 'batchmean',
        accumulate_grad_in_fp32: bool = True,
    ):
        from fla.modules.backends.fused_kl_div.triton_ascend import fused_kl_div_forward
        return fused_kl_div_forward(
            x=x,
            target_x=target_x,
            weight=weight,
            target_weight=target_weight,
            reduction=reduction,
            accumulate_grad_in_fp32=accumulate_grad_in_fp32,
        )

    def fused_kl_div_backward(self, do: torch.Tensor, dx: torch.Tensor, dw: torch.Tensor):
        from fla.modules.backends.fused_kl_div.triton_ascend import fused_kl_div_backward
        return fused_kl_div_backward(do=do, dx=dx, dw=dw)


fused_kl_div_registry = BackendRegistry("modules.fused_kl_div")
fused_kl_div_registry.register(TritonAscendFusedKLDivBackend())
