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
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        residual: torch.Tensor | None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        layout_fallback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.modules.causal_conv.triton_ascend.causal_conv import causal_conv1d_fwd_npu

        return causal_conv1d_fwd_npu(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            BT=chunk_size,
            layout_fallback=layout_fallback,
        )

    def causal_conv1d_bwd(
        self,
        x: torch.Tensor,
        dy: torch.Tensor,
        dht: torch.Tensor | None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        layout_fallback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        from fla.modules.causal_conv.triton_ascend.causal_conv import causal_conv1d_bwd_npu

        return causal_conv1d_bwd_npu(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            BT=chunk_size,
            layout_fallback=layout_fallback,
        )

    def compute_dh0_triton(
        self,
        dy: torch.Tensor,
        y: torch.Tensor | None,
        weight: torch.Tensor,
        initial_state: torch.Tensor,
        activation: str | None,
        cu_seqlens: torch.Tensor | None,
        dht: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from fla.modules.causal_conv.triton_ascend.causal_conv import compute_dh0_npu

        return compute_dh0_npu(
            dy=dy,
            y=y,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            dht=dht,
        )

    def causal_conv1d_update_states(
        self,
        x: torch.Tensor,
        state_len: int,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from fla.modules.causal_conv.triton_ascend.causal_conv import causal_conv1d_update_states_npu

        return causal_conv1d_update_states_npu(x=x, state_len=state_len, initial_state=initial_state, cu_seqlens=cu_seqlens)

    def causal_conv1d_update(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        residual: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.modules.causal_conv.triton_ascend.causal_conv import causal_conv1d_update_npu

        return causal_conv1d_update_npu(x=x, cache=cache, residual=residual, weight=weight, bias=bias, activation=activation)


__all__ = ['TritonAscendBackend']
