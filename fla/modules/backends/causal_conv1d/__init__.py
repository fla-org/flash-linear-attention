# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for causal_conv1d."""

import torch

from fla.backends import BackendRegistry, BaseBackend


def _has_non_standard_layout(x: torch.Tensor) -> bool:
    """QKV-style views (stride_t != D) break triton-ascend masked kernels."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if x.dim() != 3:
        return not x.is_contiguous()
    _, stride_t, stride_d = x.stride()
    return stride_d == 1 and stride_t != x.shape[-1]


class TritonAscendCausalConv1dBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def causal_conv1d_fwd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        BT: int = 64,
        layout_fallback: bool = False,
    ) -> torch.Tensor:
        from fla.modules.backends.causal_conv1d.triton_ascend import causal_conv1d_fwd
        return causal_conv1d_fwd(
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
            BT=BT,
            layout_fallback=layout_fallback or _has_non_standard_layout(x),
        )

    def causal_conv1d_bwd(
        self,
        x: torch.Tensor,
        dy: torch.Tensor,
        dht: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        BT: int = 64,
        layout_fallback: bool = False,
    ):
        from fla.modules.backends.causal_conv1d.triton_ascend import causal_conv1d_bwd
        return causal_conv1d_bwd(
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
            BT=BT,
            layout_fallback=layout_fallback or _has_non_standard_layout(x),
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
        from fla.modules.backends.causal_conv1d.triton_ascend import compute_dh0
        return compute_dh0(
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
        from fla.modules.backends.causal_conv1d.triton_ascend import causal_conv1d_update_states
        return causal_conv1d_update_states(x=x, state_len=state_len, initial_state=initial_state, cu_seqlens=cu_seqlens)

    def causal_conv1d_update(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        residual: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ) -> torch.Tensor:
        from fla.modules.backends.causal_conv1d.triton_ascend import causal_conv1d_update
        return causal_conv1d_update(x=x, cache=cache, residual=residual, weight=weight, bias=bias, activation=activation)


causal_conv1d_registry = BackendRegistry("modules.causal_conv1d")
causal_conv1d_registry.register(TritonAscendCausalConv1dBackend())
