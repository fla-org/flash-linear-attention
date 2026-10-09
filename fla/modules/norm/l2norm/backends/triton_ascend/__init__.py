# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

from fla.backends import BaseBackend, register_backend

if TYPE_CHECKING:
    import torch


@register_backend('modules.norm.l2norm')
class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend for l2norm."""

    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def l2norm_fwd(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.modules.norm.l2norm.backends.triton_ascend.ops import l2norm_fwd_npu
        return l2norm_fwd_npu(x=x, eps=eps, output_dtype=output_dtype)

    def l2norm_bwd(self, y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        from fla.modules.norm.l2norm.backends.triton_ascend.ops import l2norm_bwd_npu
        return l2norm_bwd_npu(y=y, rstd=rstd, dy=dy)
