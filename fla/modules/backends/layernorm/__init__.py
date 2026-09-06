# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for layernorm."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendLayerNormBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def layer_norm_fwd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-05,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        residual_dtype: torch.dtype | None = None,
        is_rms_norm: bool = False,
        num_groups: int = 1,
    ):
        from fla.modules.backends.layernorm.triton_ascend import layer_norm_fwd
        return layer_norm_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            num_groups=num_groups,
        )

    def layer_norm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mean: torch.Tensor | None = None,
        rstd: torch.Tensor | None = None,
        dres: torch.Tensor | None = None,
        has_residual: bool = False,
        is_rms_norm: bool = False,
        x_dtype: torch.dtype | None = None,
        recompute_output: bool = False,
        num_groups: int = 1,
    ):
        from fla.modules.backends.layernorm.triton_ascend import layer_norm_bwd
        return layer_norm_bwd(
            dy=dy,
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            dres=dres,
            has_residual=has_residual,
            is_rms_norm=is_rms_norm,
            x_dtype=x_dtype,
            recompute_output=recompute_output,
            num_groups=num_groups,
        )


layernorm_registry = BackendRegistry("modules.layernorm")
layernorm_registry.register(TritonAscendLayerNormBackend())
