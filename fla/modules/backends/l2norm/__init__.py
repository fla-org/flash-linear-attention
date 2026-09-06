# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for l2norm."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendL2NormBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def l2norm_fwd(self, x: torch.Tensor, eps: float = 1e-06, output_dtype: torch.dtype | None = None):
        from fla.modules.backends.l2norm.triton_ascend import l2norm_fwd
        return l2norm_fwd(x=x, eps=eps, output_dtype=output_dtype)

    def l2norm_bwd(self, y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-06):
        from fla.modules.backends.l2norm.triton_ascend import l2norm_bwd
        return l2norm_bwd(y=y, rstd=rstd, dy=dy)


l2norm_registry = BackendRegistry("modules.l2norm")
l2norm_registry.register(TritonAscendL2NormBackend())
