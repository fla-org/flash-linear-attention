# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fla.modules.activations.ops import swish_bwd
from fla.modules.causal_conv1d.ops import ShortConvolution, causal_conv1d_update_states
from fla.ops.utils import prepare_chunk_indices
from fla.utils import input_guard


@triton.jit
def _dynamic_conv1d_offsets(
    cu_seqlens,
    chunk_indices,
    T,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64) // NB
    i_n = tl.program_id(1).to(tl.int64)
    if cu_seqlens is not None:
        i_n = tl.load(chunk_indices + 2 * i_t).to(tl.int64)
        i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        length = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    else:
        bos, length = i_n * T, T
    return i_n, bos, length, i_t * BT


@triton.jit
def _dynamic_conv1d_load_x(
    x,
    initial_state,
    i_n,
    bos,
    length,
    t,
    d,
    D: tl.constexpr,
    W: tl.constexpr,
):
    values = tl.load(
        pointer=x + (bos + t[:, None]) * D + d[None, :],
        mask=(t[:, None] >= 0) & (t[:, None] < length) & (d[None, :] < D),
        other=0,
    ).to(tl.float32)
    if initial_state is not None:
        values += tl.load(
            pointer=initial_state + (i_n * D + d[None, :]) * W + t[:, None] + W,
            mask=(t[:, None] >= -W) & (t[:, None] < 0) & (d[None, :] < D),
            other=0,
        ).to(tl.float32)
    return values


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_fwd_kernel(
    x,
    weight,
    residual,
    initial_state,
    y,
    pre_activation,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_n, bos, length, start = _dynamic_conv1d_offsets(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NB=tl.cdiv(D, BD),
        BT=BT,
    )
    t = start + tl.arange(0, BT).to(tl.int64)
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    offset = (bos + t[:, None]) * D + d[None, :]
    mask = (t[:, None] < length) & (d[None, :] < D)
    out = tl.full((BT, BD), 0, tl.float32)
    for j in range(W):
        kernel = tl.load(weight + offset * W + j, mask=mask, other=0).to(tl.float32)
        values = _dynamic_conv1d_load_x(
            x=x,
            initial_state=initial_state,
            i_n=i_n,
            bos=bos,
            length=length,
            t=t - W + 1 + j,
            d=d,
            D=D,
            W=W,
        )
        out += kernel * values
    if pre_activation is not None:
        tl.store(pre_activation + offset, out, mask=mask)
        out *= tl.sigmoid(out)
    if residual is not None:
        out += tl.load(residual + offset, mask=mask, other=0).to(tl.float32)
    tl.store(y + offset, out, mask=mask)


@triton.jit(do_not_specialize=['T'])
def dynamic_conv1d_bwd_kernel(
    x,
    weight,
    initial_state,
    dy,
    dht,
    dx,
    dw,
    cu_seqlens,
    chunk_indices,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    STATE: tl.constexpr,
):
    if STATE:
        i_n = tl.program_id(1).to(tl.int64)
        if cu_seqlens is not None:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            length = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
        else:
            bos, length = i_n * T, T
        start = tl.program_id(0).to(tl.int64) // tl.cdiv(D, BD) * BT
        t = start + tl.arange(0, BT).to(tl.int64) - W
        valid = t < 0
    else:
        i_n, bos, length, start = _dynamic_conv1d_offsets(
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            NB=tl.cdiv(D, BD),
            BT=BT,
        )
        t = start + tl.arange(0, BT).to(tl.int64)
        valid = t < length
    d = (tl.program_id(0).to(tl.int64) % tl.cdiv(D, BD)) * BD + tl.arange(0, BD)
    offset = (bos + t[:, None]) * D + d[None, :]
    mask = valid[:, None] & (d[None, :] < D)
    if not STATE:
        dout = tl.load(dy + offset, mask=mask, other=0).to(tl.float32)
    grad = tl.full((BT, BD), 0, tl.float32)
    for j in range(W):
        future = t + W - 1 - j
        future_offset = (bos + future[:, None]) * D + d[None, :]
        future_mask = (future[:, None] >= 0) & (future[:, None] < length) & (d[None, :] < D)
        future_dy = tl.load(dy + future_offset, mask=future_mask, other=0).to(tl.float32)
        kernel = tl.load(weight + future_offset * W + j, mask=future_mask, other=0).to(tl.float32)
        grad += future_dy * kernel
        if not STATE:
            values = _dynamic_conv1d_load_x(
                x=x,
                initial_state=initial_state,
                i_n=i_n,
                bos=bos,
                length=length,
                t=t - W + 1 + j,
                d=d,
                D=D,
                W=W,
            )
            tl.store(dw + offset * W + j, dout * values, mask=mask)
    if dht is not None:
        slot = t - length + W
        state_mask = mask & (slot[:, None] >= 0) & (slot[:, None] < W)
        grad += tl.load(dht + (i_n * D + d[None, :]) * W + slot[:, None], mask=state_mask, other=0).to(tl.float32)
    if STATE:
        tl.store(dx + (i_n * D + d[None, :]) * W + t[:, None] + W, grad, mask=mask)
    else:
        tl.store(dx + offset, grad, mask=mask)


class DynamicConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, weight, residual, initial_state, output_final_state, activation, cu_seqlens, chunk_indices):
        B, T, D = x.shape
        W = weight.shape[-1]
        BT, BD = 64, 32
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens=cu_seqlens, chunk_size=BT)
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        y = torch.empty_like(x)
        pre_activation = torch.empty_like(x, dtype=torch.float32) if activation is not None else None
        if NT:
            dynamic_conv1d_fwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                weight=weight,
                residual=residual,
                initial_state=initial_state,
                y=y,
                pre_activation=pre_activation,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                D=D,
                W=W,
                BT=BT,
                BD=BD,
            )
        final_state = None
        if output_final_state:
            final_state = causal_conv1d_update_states(x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(x, weight, residual, initial_state, pre_activation, cu_seqlens, chunk_indices)
        return y, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dy, dht):
        x, weight, residual, initial_state, pre_activation, cu_seqlens, chunk_indices = ctx.saved_tensors
        B, T, D = x.shape
        W = weight.shape[-1]
        BT, BD = 64, 32
        NT = len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)
        dr = dy.to(residual) if residual is not None else None
        if pre_activation is not None and x.numel():
            dy = swish_bwd(x=pre_activation, dy=dy, output_contiguous=True)
        dx, dw = torch.empty_like(x), torch.empty_like(weight)
        dh0 = torch.empty_like(initial_state) if initial_state is not None else None
        if NT:
            dynamic_conv1d_bwd_kernel[(triton.cdiv(D, BD) * NT, B)](
                x=x,
                weight=weight,
                initial_state=initial_state,
                dy=dy,
                dht=dht,
                dx=dx,
                dw=dw,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                D=D,
                W=W,
                BT=BT,
                BD=BD,
                STATE=False,
            )
        if dh0 is not None:
            dynamic_conv1d_bwd_kernel[(triton.cdiv(D, BD) * triton.cdiv(W, BT), len(dh0))](
                x=x,
                weight=weight,
                initial_state=initial_state,
                dy=dy,
                dht=dht,
                dx=dh0,
                dw=None,
                cu_seqlens=cu_seqlens,
                chunk_indices=None,
                T=T,
                D=D,
                W=W,
                BT=BT,
                BD=BD,
                STATE=True,
            )
        return dx, dw, dr, dh0, None, None, None, None


class DynamicShortConvolution(ShortConvolution):
    r"""Depthwise causal convolution with input-dependent weights.

    Let D = hidden_size, r = dynamic_rank_dim, and W = kernel_size.
    For conditioning input h_t (default: x_t), the kernel generator computes

    .. math::
        z_t = \phi(A h_t), \qquad
        K_t = \operatorname{reshape}(B z_t + b, [D, W]),

    where A has shape [r, D], B has shape [D * W, r], and b has shape [D * W].
    The convolution applies the generated weights from oldest to newest:

    .. math::
        y_t = \psi\!\left(\sum_{j=0}^{W-1} K_t[:, j] \odot x_{t-W+1+j}\right)
              + \mathrm{residual}_t.

    Phi is dynamic_activation and psi is activation; None selects the identity.
    Only the second generator projection has a bias, which is always enabled.
    The convolution has no additional bias; an absent residual is zero.

    Args:
        hidden_size (int):
            Number of conditioning, input, and output channels D.
        kernel_size (int):
            Width W of the causal convolution kernel.
        dynamic_rank_dim (int, Optional):
            Rank r of the kernel generator. Default: 16.
        dynamic_activation (str, Optional):
            Activation between projections: `None`, `"silu"`, or `"swish"`. Default: `None`.
        activation (str, Optional):
            Convolution output activation: `None`, `"silu"`, or `"swish"`. Default: `None`.
        backend (str, Optional):
            Convolution backend; dynamic weights support `"triton"` on CUDA/HIP. Default: `"triton"`.
        device (torch.device, Optional):
            Device for the parameters. Default: `None`.
        dtype (torch.dtype, Optional):
            Parameter dtype. Default: `None`.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dynamic_rank_dim: int = 16,
        dynamic_activation: str | None = None,
        activation: str | None = None,
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if min(hidden_size, kernel_size, dynamic_rank_dim) < 1:
            raise ValueError('hidden_size, kernel_size, and dynamic_rank_dim must be positive')
        if dynamic_activation not in (None, 'silu', 'swish'):
            raise ValueError('dynamic_activation must be None, silu, or swish')
        super().__init__(
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            bias=False,
            activation=activation,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        if self.backend != 'triton':
            raise ValueError('DynamicShortConvolution requires the triton backend')
        self.dynamic_rank_dim = dynamic_rank_dim
        del self.weight
        self.weight = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=dynamic_rank_dim, bias=False, device=device, dtype=dtype),
            nn.Identity() if dynamic_activation is None else nn.SiLU(),
            nn.Linear(
                in_features=dynamic_rank_dim,
                out_features=hidden_size * kernel_size,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not isinstance(self.weight, nn.Sequential):
            super().reset_parameters()
            return
        nn.init.normal_(self.weight[0].weight, std=0.02)
        nn.init.zeros_(self.weight[2].weight)
        bound = self.kernel_size[0] ** -0.5
        nn.init.uniform_(self.weight[2].bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        *,
        generator_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply dynamic convolution, returning the output and optional final input history.

        Args:
            x (torch.Tensor):
                Convolution input [B, T, D]. Packed inputs require B=1.
            residual (torch.Tensor, Optional):
                Tensor [B, T, D] added after the output activation. Default: `None`.
            mask (torch.Tensor, Optional):
                Attention mask [B, T], mutually exclusive with cu_seqlens. Default: `None`.
            cache (torch.Tensor, Optional):
                Input history [N, D, W], ordered oldest to newest. Default: `None`.
                Single-token calls without autograd update it in place; other calls leave it unchanged.
            output_final_state (bool, Optional):
                Whether to return the final input history. Default: `False`.
            cu_seqlens (torch.Tensor, Optional):
                Packed sequence boundaries [N+1]. Default: `None`.
            chunk_indices (torch.Tensor, Optional):
                Packed chunk indices for 64-token chunks. Default: `None`.
            generator_input (torch.Tensor, Optional):
                Conditioning input [B, T, D]. Default: `x`.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                Output [B, T, D] and final history [N, D, W] when requested, otherwise `None`.
        """
        if x.device.type != 'cuda':
            raise ValueError('DynamicShortConvolution requires CUDA/HIP tensors')
        if kwargs.get('cp_context') is not None:
            raise ValueError('DynamicShortConvolution does not support context parallelism')
        B, T, D = x.shape
        W = self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if cu_seqlens is not None and B != 1:
            raise ValueError('Packed inputs require batch size 1')
        if cache is not None and cache.shape != (N, D, W):
            raise ValueError('cache must have shape [N, D, W]')
        if residual is not None and residual.shape != x.shape:
            raise ValueError('residual must have the same shape as x')
        generator_input = x if generator_input is None else generator_input
        if generator_input.shape != x.shape:
            raise ValueError('generator_input must have the same shape as x')
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError('mask and cu_seqlens cannot be provided at the same time')
            x = x * mask.unsqueeze(-1).to(x.dtype)
            generator_input = generator_input * mask.unsqueeze(-1).to(generator_input.dtype)
        weight = self.weight(generator_input).unflatten(dim=-1, sizes=(D, W))
        update_cache = not torch.is_grad_enabled() and B * T == N
        if update_cache and cu_seqlens is not None:
            update_cache = bool((cu_seqlens.diff() == 1).all())
        y, final_state = DynamicConv1dFunction.apply(
            x, weight, residual, cache, output_final_state or update_cache, self.activation, cu_seqlens, chunk_indices,
        )
        if update_cache and cache is not None:
            cache.copy_(final_state)
            final_state = cache
        return y, final_state if output_final_state else None

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        *,
        generator_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward(
            x=x,
            residual=residual,
            cache=cache,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            generator_input=generator_input,
        )
