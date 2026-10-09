# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register_backend


@register_backend('modules.norm.fused_norm_gate')
class TritonAscendBackend(BaseBackend):
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
        from fla.modules.norm.fused_norm_gate.backends.triton_ascend.ops import layer_norm_gated_fwd_npu
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
        from fla.modules.norm.fused_norm_gate.backends.triton_ascend.ops import layer_norm_gated_bwd_npu
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
