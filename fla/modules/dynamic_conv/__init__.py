# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules.dynamic_conv.ops import dynamic_conv1d


class DynamicShortConvolution(nn.Module):
    """Depthwise causal convolution with a low-rank, input-dependent kernel generator.

    Args:
        hidden_size (int):
            Number of convolution channels.
        kernel_size (int):
            Causal window width, between 1 and 8.
        bias (bool, Optional):
            Whether to use a channel bias. Default: `False`.
        activation (str, Optional):
            Output activation, accepting None, silu, or swish. Default: `"silu"`.
        backend (str, Optional):
            Convolution backend. Only `"triton"` is supported. Default: `"triton"`.
        device (torch.device, Optional):
            Device for the parameters. Default: `None`.
        dtype (torch.dtype, Optional):
            Parameter dtype. Default: `None`.
        rank (int, Optional):
            Kernel-generator bottleneck width, between 1 and 128. Default: 16.
        generator_input_size (int, Optional):
            Generator input width. Default: `hidden_size`.
        generator_activation (str, Optional):
            Optional SiLU/swish between the generator projections. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        backend: str = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        rank: int = 16,
        generator_input_size: int | None = None,
        generator_activation: str | None = None,
    ) -> None:
        super().__init__()
        generator_input_size = hidden_size if generator_input_size is None else generator_input_size
        if hidden_size < 1 or generator_input_size < 1 or not 1 <= kernel_size <= 8 or not 1 <= rank <= 128:
            raise ValueError('positive channel sizes, 1 <= kernel_size <= 8, and 1 <= rank <= 128 are required')
        if generator_activation not in (None, 'silu', 'swish') or activation not in (None, 'silu', 'swish'):
            raise ValueError('activation and generator_activation must be None, silu, or swish')
        if backend != 'triton':
            raise ValueError('DynamicShortConvolution only supports the triton backend')
        self.hidden_size = hidden_size
        self.kernel_size = (kernel_size,)
        self.activation = activation
        self.backend = backend
        self.generator_activation = generator_activation
        self.in_proj = nn.Linear(generator_input_size, rank, bias=False, device=device, dtype=dtype)
        self.kernel_basis = nn.Parameter(torch.zeros(rank, kernel_size, hidden_size, device=device, dtype=dtype))
        self.weight = nn.Parameter(torch.empty(kernel_size, hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype)) if bias else None
        nn.init.normal_(self.in_proj.weight, std=0.02)
        bound = 1 / math.sqrt(kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        *,
        generator_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convolve x using the current generator input, which defaults to x.

        Inputs and residuals have shape [B, T, D]; generator_input has shape [B, T, generator_input_size].
        An optional [B, T] mask zeros convolution and generator inputs; it cannot be combined with cu_seqlens.
        Cache stores raw input history [N, D, W]. Single-token calls without autograd update it in place.
        Returns (output, final_state), with final_state=None unless output_final_state=True.
        Packed chunk_indices must use a chunk size of 16.
        """
        generator_input = x if generator_input is None else generator_input
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError('mask and cu_seqlens cannot be provided at the same time')
            x = x * mask.unsqueeze(-1).to(x.dtype)
            generator_input = generator_input * mask.unsqueeze(-1).to(generator_input.dtype)
        z = self.in_proj(generator_input)
        if self.generator_activation is not None:
            z = F.silu(z)
        B, T, _ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        update_cache = not torch.is_grad_enabled() and B * T == N
        if update_cache and cu_seqlens is not None:
            update_cache = bool((cu_seqlens.diff() == 1).all())
        y, final_state = dynamic_conv1d(
            x=x,
            z=z,
            weight=self.kernel_basis,
            kernel_bias=self.weight,
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state or update_cache,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if update_cache and cache is not None:
            cache.copy_(final_state)
            final_state = cache
        return y, final_state if output_final_state else None

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]


__all__ = ['DynamicShortConvolution', 'dynamic_conv1d']
