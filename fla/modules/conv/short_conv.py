# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Short convolution implementation for efficient causal convolutions."""

import warnings

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.l2norm import l2norm
from fla.utils import IS_NVIDIA

try:
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_cuda
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update_cuda
except ImportError:
    causal_conv1d_fn_cuda = None
    causal_conv1d_update_cuda = None


class ShortConvolution(nn.Conv1d):
    """Short convolution layer for efficient causal convolution operations.

    This class implements a depthwise 1D convolution with causal padding,
    designed for efficient sequence processing. It supports multiple backends (Triton/CUDA)
    and optional activation functions.

    Args:
        hidden_size (int): Number of input/output channels (must be equal for depthwise conv)
        kernel_size (int): Size of the convolution kernel
        bias (bool, optional): Whether to include learnable bias. Defaults to False.
        activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to 'silu'.
        backend (Optional[str], optional): Backend implementation ('triton' or 'cuda'). Defaults to 'triton'.
        device (Optional[torch.device], optional): Device to place the layer on. Defaults to None.
        dtype (Optional[torch.dtype], optional): Data type for layer parameters. Defaults to None.
        norm (str, Optional):
            Head-wise normalization applied after convolution, activation, and residual. Default: `None`.
            Supports `'l2'`.
        norm_eps (float, Optional):
            Epsilon added to the squared L2 norm. Default: 1e-6.
        **kwargs: Additional keyword arguments (deprecated 'use_fast_conv1d' supported for compatibility)

    Attributes:
        hidden_size (int): Number of channels
        activation (Optional[str]): Selected activation function
        backend (str): Actual backend being used (may differ from input due to availability)

    Note:
        - Uses depthwise convolution (groups=hidden_size) for efficiency
        - Applies causal padding (kernel_size-1) to ensure no future information leakage
        - Falls back to Triton backend if CUDA backend is unavailable
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
        norm: str | None = None,
        norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if norm not in (None, 'l2'):
            raise ValueError(f"Unsupported normalization: {norm}")
        if norm is not None and not norm_eps > 0:
            raise ValueError("`norm_eps` must be positive when normalization is enabled.")
        self.norm = norm
        self.norm_eps = norm_eps

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if 'use_fast_conv1d' in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead.",
            )
        import os
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

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
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
        if self.norm is not None:
            s += ', norm={norm}'
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
        head_dim: int | None = None,
        use_norm: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            chunk_indices (Optional[torch.LongTensor]):
                Chunk indices for variable-length sequences. Default: `None`.
            head_dim (int, Optional):
                Channels per head, required when normalization is enabled. Default: `None`.
            use_norm (bool, Optional):
                Whether to apply the configured normalization for this call. Default: `True`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """
        # Import here to avoid circular dependency
        from fla.modules.conv.causal_conv1d import causal_conv1d

        use_norm = self.norm is not None and use_norm
        if use_norm and (isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0 or x.shape[-1] % head_dim):
            raise ValueError("`head_dim` must be a positive divisor of the channel dimension when normalization is enabled.")

        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        # For packed varlen inputs, decode only when every sequence has exactly one token:
        # a zero-length or multi-token sequence makes the shape check misfire.
        if (
            B * T == N and
            (cu_seqlens is None or bool((cu_seqlens.diff() == 1).all())) and
            not (use_norm and kwargs.get('cp_context') is not None)
        ):
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                head_dim=head_dim,
                use_norm=use_norm,
            )
            return y, cache

        # cuda backend do not support:
        # 1. both `cu_seqlens` and `cache` being provided
        # 2. both `cu_seqlens` and `output_final_state` being provided
        # and other small issues
        # to simplify the implementation, we just switch to triton backend
        if self.backend == 'cuda' and cache is not None:
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = 'triton'

        if use_norm and self.backend == 'triton' and IS_NVIDIA and kwargs.get('cp_context') is None:
            from fla.ops.convolution import fused_short_conv

            return fused_short_conv(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                residual=residual,
                initial_state=cache,
                output_final_state=output_final_state,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_norm=True,
                norm_eps=self.norm_eps,
                head_dim=head_dim,
                cu_seqlens_cpu=kwargs.get('cu_seqlens_cpu'),
            )

        y, final_state = causal_conv1d(
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
        if use_norm:
            y = l2norm(x=y.reshape(*y.shape[:-1], y.shape[-1] // head_dim, head_dim), eps=self.norm_eps).reshape_as(y)
        return y, final_state

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        head_dim: int | None = None,
        use_norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.modules.conv.triton.ops import causal_conv1d_update

        use_norm = self.norm is not None and use_norm
        if use_norm and (isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0 or x.shape[-1] % head_dim):
            raise ValueError("`head_dim` must be a positive divisor of the channel dimension when normalization is enabled.")

        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        # Always initialise cache when None so the Triton kernel never
        # receives a None tensor. Return value still respects output_final_state
        # to maintain consistency with the non-step path in forward().
        if cache is None:
            cache = x.new_zeros(N, D, W)
        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == 'triton':
            y, cache = causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
            if use_norm:
                y = l2norm(x=y.reshape(*y.shape[:-1], y.shape[-1] // head_dim, head_dim), eps=self.norm_eps).reshape_as(y)
            return y, (cache if output_final_state else None)

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # equivalent to:
        # cache.copy_(cache.roll(shifts=-1, dims=-1))
        # cache[:, :, -1] = x
        # y = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
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
        if use_norm:
            y = l2norm(x=y.reshape(*y.shape[:-1], y.shape[-1] // head_dim, head_dim), eps=self.norm_eps).reshape_as(y)
        return y, (cache if output_final_state else None)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]
