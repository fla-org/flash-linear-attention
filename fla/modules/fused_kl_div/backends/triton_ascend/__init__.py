# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from fla.backends import BaseBackend, register_backend


@register_backend('modules.fused_kl_div')
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

    def fused_kl_div_fwd(
        self,
        x,
        target_x,
        weight,
        target_weight,
        reduction='batchmean',
        accumulate_grad_in_fp32=True,
        use_dx=True,
        use_dw=True,
    ):
        from fla.modules.fused_kl_div.backends.triton_ascend.ops import fused_kl_div_fwd_npu
        return fused_kl_div_fwd_npu(
            x=x,
            target_x=target_x,
            weight=weight,
            target_weight=target_weight,
            reduction=reduction,
            accumulate_grad_in_fp32=accumulate_grad_in_fp32,
            use_dx=use_dx,
            use_dw=use_dw,
        )

    def fused_kl_div_bwd(self, do, dx, dw):
        from fla.modules.fused_kl_div.backends.triton_ascend.ops import fused_kl_div_bwd_npu
        return fused_kl_div_bwd_npu(do=do, dx=dx, dw=dw)
