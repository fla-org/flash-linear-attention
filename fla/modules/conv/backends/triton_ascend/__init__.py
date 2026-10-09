# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register_backend


@register_backend('modules.conv')
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

    def causal_conv1d_fwd(
        self,
        x,
        weight,
        bias,
        residual,
        initial_state=None,
        output_final_state=False,
        activation=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        chunk_size=64,
        layout_fallback=False,
    ):
        from fla.modules.conv.backends.triton_ascend.ops import causal_conv1d_fwd_npu
        return causal_conv1d_fwd_npu(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
            chunk_size,
            layout_fallback,
        )

    def causal_conv1d_bwd(
        self,
        x,
        dy,
        dht,
        weight=None,
        bias=None,
        residual=None,
        initial_state=None,
        activation=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        chunk_size=64,
        layout_fallback=False,
    ):
        from fla.modules.conv.backends.triton_ascend.ops import causal_conv1d_bwd_npu
        return causal_conv1d_bwd_npu(
            x,
            dy,
            dht,
            weight,
            bias,
            residual,
            initial_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
            chunk_size,
            layout_fallback,
        )

    def compute_dh0_triton(
        self,
        dy,
        y,
        weight,
        initial_state,
        activation,
        cu_seqlens,
        dht=None,
    ):
        from fla.modules.conv.backends.triton_ascend.ops import compute_dh0_npu
        return compute_dh0_npu(
            dy,
            y,
            weight,
            initial_state,
            activation,
            cu_seqlens,
            dht,
        )

    def causal_conv1d_update_states(
        self,
        x,
        state_len,
        initial_state=None,
        cu_seqlens=None,
    ):
        from fla.modules.conv.backends.triton_ascend.ops import causal_conv1d_update_states_npu
        return causal_conv1d_update_states_npu(
            x,
            state_len,
            initial_state,
            cu_seqlens,
        )

    def causal_conv1d_update(
        self,
        x,
        cache,
        residual=None,
        weight=None,
        bias=None,
        activation=None,
    ):
        from fla.modules.conv.backends.triton_ascend.ops import causal_conv1d_update_npu
        return causal_conv1d_update_npu(
            x,
            cache,
            residual,
            weight,
            bias,
            activation,
        )


__all__ = ['TritonAscendBackend']
