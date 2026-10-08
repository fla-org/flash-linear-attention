# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register_backend


@register_backend('modules.norm.l2norm')
class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend for l2norm."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def l2norm_fwd(
        self,
        x,
        eps=1e-6,
        output_dtype=None,
    ):
        from fla.modules.norm.l2norm.backends.triton_ascend.ops import l2norm_fwd_npu
        return l2norm_fwd_npu(x, eps, output_dtype)

    def l2norm_bwd(
        self,
        y,
        rstd,
        dy,
        eps=1e-6,
    ):
        from fla.modules.norm.l2norm.backends.triton_ascend.ops import l2norm_bwd_npu
        return l2norm_bwd_npu(y, rstd, dy)
