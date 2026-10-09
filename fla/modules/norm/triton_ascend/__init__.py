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


@register_backend('modules.norm.layernorm')
class LayerNormBackend(BaseBackend):
    """Ascend NPU backend for layernorm."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def layer_norm_fwd(
        self,
        x,
        weight,
        bias,
        eps=1e-5,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        is_rms_norm=False,
        num_groups=1,
    ):
        from fla.modules.norm.triton_ascend.layernorm import layer_norm_fwd_npu
        return layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            residual,
            out_dtype,
            residual_dtype,
            is_rms_norm,
            num_groups,
        )

    def layer_norm_bwd(
        self,
        dy,
        x,
        weight,
        bias,
        mean=None,
        rstd=None,
        dres=None,
        has_residual=False,
        is_rms_norm=False,
        x_dtype=None,
        recompute_output=False,
        num_groups=1,
    ):
        from fla.modules.norm.triton_ascend.layernorm import layer_norm_bwd_npu
        return layer_norm_bwd_npu(
            dy,
            x,
            weight,
            bias,
            mean,
            rstd,
            dres,
            has_residual,
            is_rms_norm,
            x_dtype,
            recompute_output,
            num_groups,
        )


@register_backend('modules.norm.l2norm')
class L2NormBackend(BaseBackend):
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
        from fla.modules.norm.triton_ascend.l2norm import l2norm_fwd_npu
        return l2norm_fwd_npu(x=x, eps=eps, output_dtype=output_dtype)

    def l2norm_bwd(self, y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        from fla.modules.norm.triton_ascend.l2norm import l2norm_bwd_npu
        return l2norm_bwd_npu(y=y, rstd=rstd, dy=dy)


@register_backend('modules.norm.fused_norm_gate')
class FusedNormGateBackend(BaseBackend):
    """Ascend NPU backend for fused_norm_gate."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def layer_norm_gated_fwd(
        self,
        x,
        g,
        weight,
        bias,
        activation="swish",
        eps=1e-5,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        is_rms_norm=False,
    ):
        from fla.modules.norm.triton_ascend.fused_norm_gate import layer_norm_gated_fwd_npu
        return layer_norm_gated_fwd_npu(
            x,
            g,
            weight,
            bias,
            activation,
            eps,
            residual,
            out_dtype,
            residual_dtype,
            is_rms_norm,
        )

    def layer_norm_gated_bwd(
        self,
        dy,
        x,
        g,
        weight,
        bias,
        activation="swish",
        eps=1e-5,
        mean=None,
        rstd=None,
        dresidual=None,
        has_residual=False,
        is_rms_norm=False,
        x_dtype=None,
        recompute_output=False,
    ):
        from fla.modules.norm.triton_ascend.fused_norm_gate import layer_norm_gated_bwd_npu
        return layer_norm_gated_bwd_npu(
            dy,
            x,
            g,
            weight,
            bias,
            activation,
            eps,
            mean,
            rstd,
            dresidual,
            has_residual,
            is_rms_norm,
            x_dtype,
            recompute_output,
        )
