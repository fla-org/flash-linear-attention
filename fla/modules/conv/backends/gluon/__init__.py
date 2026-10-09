# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Gluon backend for FLA modules."""

import torch

from fla.backends import BaseBackend, register_backend
from fla.utils import IS_NVIDIA


@register_backend('modules.conv')
class ConvGluonBackend(BaseBackend):
    """NVIDIA GPU backend using Gluon kernels."""

    backend_type = "gluon"
    package_name = "triton.experimental.gluon"
    env_var = "FLA_CONV_GLUON"
    default_enable = False
    priority = 5

    @classmethod
    def is_available(cls) -> bool:
        return IS_NVIDIA and super().is_available()

    def causal_conv1d_fwd_verifier(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        layout_fallback: bool = False,
    ) -> tuple[bool, str | None]:
        if torch.distributed.is_initialized():
            return False, "Gluon convolution does not support distributed execution"
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, "Gluon convolution supports float16, bfloat16, and float32"
        if x.ndim != 3 or x.stride(-1) != 1:
            return False, "Gluon convolution requires [B, T, D] with contiguous channels"
        if weight is None or weight.shape[0] != x.shape[-1] or weight.shape[1] not in (2, 3, 4):
            return False, "Gluon convolution requires a width of 2, 3, or 4"
        if cu_seqlens is not None and x.shape[0] != 1:
            return False, "Gluon packed convolution requires batch size 1"
        if chunk_size != 64:
            return False, "Gluon convolution requires 64-token chunk indices"
        return True, None

    def causal_conv1d_fwd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        layout_fallback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.modules.conv.backends.gluon.ops import causal_conv1d_fwd
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
            chunk_size=chunk_size,
        )

    def causal_conv1d_bwd_verifier(
        self,
        x: torch.Tensor,
        dy: torch.Tensor,
        dht: torch.Tensor | None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        activation: str | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        layout_fallback: bool = False,
    ) -> tuple[bool, str | None]:
        if initial_state is not None or dht is not None:
            return False, "Gluon convolution does not support state gradients"
        return self.causal_conv1d_fwd_verifier(x=x, weight=weight, cu_seqlens=cu_seqlens, chunk_size=chunk_size)

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
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        layout_fallback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, None]:
        from fla.modules.conv.backends.gluon.ops import causal_conv1d_bwd
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
            chunk_size=chunk_size,
        )


GluonBackend = ConvGluonBackend

__all__ = ['ConvGluonBackend', 'GluonBackend']
