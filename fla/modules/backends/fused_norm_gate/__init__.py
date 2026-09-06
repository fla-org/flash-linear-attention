# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for fused_norm_gate."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendFusedNormGateBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def layer_norm_gated_fwd(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str = 'swish',
        eps: float = 1e-05,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        residual_dtype: torch.dtype | None = None,
        is_rms_norm: bool = False,
    ):
        from fla.modules.backends.fused_norm_gate.triton_ascend import layer_norm_gated_fwd
        return layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )

    def layer_norm_gated_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str = 'swish',
        eps: float = 1e-05,
        mean: torch.Tensor | None = None,
        rstd: torch.Tensor | None = None,
        dresidual: torch.Tensor | None = None,
        has_residual: bool = False,
        is_rms_norm: bool = False,
        x_dtype: torch.dtype | None = None,
        recompute_output: bool = False,
    ):
        from fla.modules.backends.fused_norm_gate.triton_ascend import layer_norm_gated_bwd
        return layer_norm_gated_bwd(
            dy=dy,
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            mean=mean,
            rstd=rstd,
            dresidual=dresidual,
            has_residual=has_residual,
            is_rms_norm=is_rms_norm,
            x_dtype=x_dtype,
            recompute_output=recompute_output,
        )


fused_norm_gate_registry = BackendRegistry("modules.fused_norm_gate")
fused_norm_gate_registry.register(TritonAscendFusedNormGateBackend())
