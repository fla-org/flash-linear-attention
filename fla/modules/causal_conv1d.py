# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import warnings

import torch
import torch.nn as nn
from einops import rearrange

from fla.ops.causal_conv1d import (
    CausalConv1dFunction,
    causal_conv1d,
    causal_conv1d_bwd,
    causal_conv1d_fwd,
    causal_conv1d_update,
    causal_conv1d_update_states,
)
from fla.ops.causal_conv1d.ops import causal_conv1d_fn_cuda, causal_conv1d_update_cuda


class ShortConvolution(nn.Conv1d):
    """Depthwise causal convolution with optional activation and state caching.

    Args:
        hidden_size (int):
            Number of input and output channels.
        kernel_size (int):
            Width of the convolution kernel.
        bias (bool, Optional):
            Whether to use a channel bias. Default: `False`.
        activation (str, Optional):
            Activation applied to the convolution output. Default: `"silu"`.
        backend (str, Optional):
            Backend selected unless overridden by `FLA_CONV_BACKEND`. Default: `"triton"`.
        device (torch.device, Optional):
            Device for the parameters. Default: `None`.
        dtype (torch.dtype, Optional):
            Parameter dtype. Default: `None`.
        **kwargs:
            Accepts the deprecated `use_fast_conv1d` option.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if 'use_fast_conv1d' in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead.",
            )
        self.backend = os.environ.get('FLA_CONV_BACKEND', backend)
        if self.backend not in ['cuda', 'triton']:
            raise ValueError(f"Invalid backend: {self.backend}, must be one of ['cuda', 'triton']")
        if self.backend == 'cuda':
            if causal_conv1d_fn_cuda is None:
                warnings.warn(
                    "The `backend` parameter is set to `cuda`, but `causal_conv1d_fn` is not available. "
                    "Switching to the Triton implementation instead. "
                    "Consider installing `causal_conv1d` to enable the CUDA backend.",
                )
                self.backend = 'triton'

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        s += f', backend={self.backend}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x (torch.Tensor):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (torch.Tensor, Optional):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (torch.Tensor, Optional):
                Attention mask for padded positions. Default: `None`.
            cache (torch.Tensor, Optional):
                Previous cache of shape `[N, D, W]`, where `W` is the kernel size. Default: `None`.
                Single-token calls without autograd update it **inplace**.
                Other calls return the updated state when `output_final_state=True`.
            output_final_state (bool, Optional):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (torch.Tensor, Optional):
                Cumulative sequence lengths of shape `[N+1]`. Default: `None`.
            chunk_indices (torch.Tensor, Optional):
                Chunk indices for variable-length sequences. Default: `None`.

        Returns:
            y (torch.Tensor):
                Convolution output of shape `[B, T, D]`.
            final_state (torch.Tensor, Optional):
                Cache of shape `[N, D, W]` when requested, otherwise `None`.
        """
        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # the in-place decode kernel requires autograd off and exactly one token per sequence, including packed inputs.
        if not torch.is_grad_enabled() and B * T == N and (cu_seqlens is None or bool((cu_seqlens.diff() == 1).all())):
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return y, cache

        # use Triton for stateful calls unsupported by the CUDA backend.
        if self.backend == 'cuda' and cache is not None:
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = 'triton'

        return causal_conv1d(
            x=x,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs,
        )

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        # the update kernel requires a cache even when the final state is not requested.
        if cache is None:
            cache = x.new_zeros(N, D, W)
        if self.backend == 'triton':
            y, cache = causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
            return y, (cache if output_final_state else None)

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # the CUDA update mutates cache in place.
        y = causal_conv1d_update_cuda(
            x=x,
            conv_state=cache,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            activation=self.activation,
        )
        y = y.view(shape)
        if residual is not None:
            y.add_(residual)
        return y, (cache if output_final_state else None)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]


__all__ = [
    'CausalConv1dFunction',
    'ShortConvolution',
    'causal_conv1d',
    'causal_conv1d_bwd',
    'causal_conv1d_fwd',
    'causal_conv1d_update',
    'causal_conv1d_update_states',
]
